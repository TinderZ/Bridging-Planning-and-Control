# harvest_v2/harvest_env.py
import functools
import random
from copy import deepcopy
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec


from .harvest_agent import HarvestAgent
from .llm_social_hub_harvest import LLMSocialHubHarvest
from .constants import (
    HARVEST_MAP, DEFAULT_COLOURS, HARVEST_VIEW_SIZE, VIEW_PADDING,
    ACTION_MEANING, NUM_ACTIONS, MOVE_ACTIONS, TURN_ACTIONS, SPECIAL_ACTIONS,
    ORIENTATIONS, ORIENTATION_VECTORS, ROTATION_MAP, STAY_ACTION_INDEX,
    WALL, AGENT_START, APPLE_SPAWN, EMPTY, APPLE, PENALTY_BEAM, AGENT_CHARS,
    NON_WALKABLE, FIRE_BLOCKING_CELLS,
    APPLE_REWARD, PENALTY_HIT, PENALTY_FIRE,
    FIRE_BEAM_LENGTH, FIRE_BEAM_WIDTH,
    SPAWN_PROB, APPLE_RADIUS, APPLE_RESPAWN_COOLDOWN,
    LLM_COMMAND_ENCODING
)

class HarvestEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "harvest_v2"}

    def __init__(
        self,
        num_agents: int = 5,
        render_mode: str | None = None,
        max_cycles: int = 1000,
        use_llm: bool = False,
        llm_f_step: int = 50,
        llm_type: str = "rule-based",
        llm_provider: str = "vllm", # Add llm_provider parameter
        **kwargs
    ):
        super().__init__()

        if not (0 < num_agents <= len(AGENT_CHARS)):
            raise ValueError(f"Number of agents must be between 1 and {len(AGENT_CHARS)}.")

        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agent_id_map = {i: agent_id for i, agent_id in enumerate(self.possible_agents)}

        self.render_mode = render_mode
        self.max_cycles = max_cycles

        self.use_llm = use_llm
        if self.use_llm:
            self.llm_f_step = llm_f_step
            self.llm_social_hub = LLMSocialHubHarvest(self.possible_agents, llm_type, llm_provider=llm_provider) # Pass llm_provider
        else:
            self.llm_social_hub = None
        self.llm_task_commands: Dict[str, str | None] = {}
        self.llm_state_commands: Dict[str, str | None] = {} # Initialize llm_state_commands


        self.base_map = self._ascii_to_numpy(HARVEST_MAP)
        self.map_height, self.map_width = self.base_map.shape

        self.spawn_points = self._find_points(AGENT_START)
        self.apple_spawn_points = self._find_points(APPLE_SPAWN)
        self.wall_points = self._find_points(WALL)

        self._agents: Dict[str, HarvestAgent] = {}
        self.num_cycles = 0
        self.beam_pos: List[Tuple[int, int, bytes]] = []
        self.apple_cooldowns: Dict[Tuple[int, int], int] = {}

        # --- Statistics Tracking ---
        self._agent_step_stats: Dict[str, Dict[str, int]] = {}
        self._step_sustainability:Dict[str, int] = {} # Initialize step_sustainability
        self._agent_cumulative_apples: Dict[str, int] = {}
        self.current_apple_density = 0.0 # Initialize current_apple_density

        self.fig = None
        self.ax = None
        self.render_im = None

        self.observation_spaces = {
            agent_id: spaces.Box(low=0, high=255, shape=(2 * HARVEST_VIEW_SIZE + 1, 2 * HARVEST_VIEW_SIZE + 1, 3), dtype=np.uint8)
            for agent_id in self.possible_agents
        }
        self.action_spaces = {
            agent_id: spaces.Discrete(NUM_ACTIONS) for agent_id in self.possible_agents
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.agents = self.possible_agents[:]
        self._agents = {}
        self.world_map = deepcopy(self.base_map)

        available_spawn_points = deepcopy(self.spawn_points)
        random.shuffle(available_spawn_points)

        for i, agent_id in enumerate(self.agents):
            spawn_pos = np.array(available_spawn_points.pop())
            orientation = random.choice(list(ORIENTATIONS.keys()))
            self._agents[agent_id] = HarvestAgent(i, spawn_pos, orientation, HARVEST_VIEW_SIZE)
            self.world_map[spawn_pos[0], spawn_pos[1]] = EMPTY

        self.num_cycles = 0
        self.beam_pos = []
        self.apple_cooldowns = {}

        if self.use_llm and self.llm_social_hub:
            self.llm_social_hub.reset_game() # Reset the social hub for a new game
        self.llm_task_commands = {agent_id: None for agent_id in self.agents}
        self.llm_state_commands = {agent_id: "default" for agent_id in self.agents} # Reset llm_state_commands
        self.current_apple_density = 0.0 # Reset apple density

        # Initialize statistics
        self._agent_step_stats = {agent_id: {"apples_collected_step": 0} for agent_id in self.agents}
        self._step_sustainability = {agent_id: 0 for agent_id in self.agents}
        self._agent_cumulative_apples = {agent_id: 0 for agent_id in self.agents}

        # Populate initial apples
        self._update_environment_state()

        observations = {agent_id: self._get_observation(agent_id) for agent_id in self.agents}
        infos = {agent_id: {} for agent_id in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        self.num_cycles += 1
        self.beam_pos = []
        agents_at_step_start = self.agents[:]

        # Reset step-specific stats
        for agent_id in agents_at_step_start:
            self._agent_step_stats[agent_id] = {"apples_collected_step": 0}
            self._step_sustainability[agent_id] = 0

        if self.use_llm and self.num_cycles % self.llm_f_step == 1:
            self._get_llm_commands(agents_at_step_start)

        agent_action_map, agent_new_positions = self._process_agent_movements(actions)
        self._resolve_movement_conflicts(agent_new_positions)
        self._handle_consumption_and_special_actions(agent_action_map)
        self._update_environment_state()

        rewards = {agent_id: self._agents[agent_id].consume_reward() for agent_id in agents_at_step_start}
        for agent_id, reward in rewards.items():
            self._agents[agent_id].add_cumulative_reward(reward)

        is_truncated = self.num_cycles >= self.max_cycles
        terminations = {agent_id: False for agent_id in agents_at_step_start}
        truncations = {agent_id: is_truncated for agent_id in agents_at_step_start}

        observations = {agent_id: self._get_observation(agent_id) for agent_id in agents_at_step_start}

        # Build rich infos dictionary
        infos = {}
        llm_stats = None
        if self.use_llm and self.llm_social_hub:
            llm_stats = self.llm_social_hub.get_decision_stats()

        for agent_id in agents_at_step_start:
            command_str = self.llm_task_commands.get(agent_id, "NONE")

            agent_info = {
                "apples_collected_step": self._agent_step_stats[agent_id]["apples_collected_step"],
                "cumulative_reward": self._agents[agent_id].get_cumulative_reward(),
                "llm_command_encoded": LLM_COMMAND_ENCODING.get(command_str, 0),
                "step_sustainability":self._step_sustainability[agent_id]
            }
            if llm_stats:
                agent_info["llm_total_farm_decisions"] = llm_stats.get("total_farm_decisions", 0)
                agent_info["llm_total_conserve_decisions"] = llm_stats.get("total_conserve_decisions", 0)
                agent_info["llm_farm_proportion"] = llm_stats.get("farm_proportion", 0.0)
            infos[agent_id] = agent_info

        if is_truncated:
            self.agents = []

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self, mode=None) -> np.ndarray | None:
        """Renders the environment."""
        rgb_map = self._map_to_colors()

        if self.render_mode == "human":
            if self.fig is None or self.ax is None:
                plt.ion()
                self.fig, self.ax = plt.subplots(1, 1)
                plt.title("Harvest Social Dilemma")
                self.render_im = None

            if self.render_im is None:
                self.render_im = self.ax.imshow(rgb_map, interpolation='nearest')
            else:
                self.render_im.set_data(rgb_map)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            return None
        elif self.render_mode == "rgb_array":
            return rgb_map
        return None

    def _get_llm_commands(self, agents_active: List[str]):
        if not self.use_llm or not self.llm_social_hub: return

        # Global apple density
        apple_density = np.count_nonzero(self.world_map == APPLE) / len(self.apple_spawn_points)
        self.current_apple_density = apple_density

        # Agent-specific status
        agent_status = {}
        for agent_id in agents_active:
            agent = self._agents[agent_id]
            local_density = self._get_local_apple_density(agent.get_pos())
            agent_status[agent_id] = {
                "apples_collected": self._agent_cumulative_apples.get(agent_id, 0),
                "local_density": local_density
            }

        env_state = {
            "apple_density": apple_density,
            "agent_status": agent_status  # New key with richer info
        }

        # Pass the current step count and new env_state to the controller
        all_decisions = self.llm_social_hub.get_commands(env_state, self.num_cycles)
        for agent_id, decision in all_decisions.items():
            if decision:
                self.llm_task_commands[agent_id] = decision.get("action")
                self.llm_state_commands[agent_id] = decision.get("state", "default")

    def _get_local_apple_density(self, agent_pos: np.ndarray) -> float:
        """Calculates the apple density within an agent's view."""
        view_size = HARVEST_VIEW_SIZE
        r_start = max(0, agent_pos[0] - view_size)
        r_end = min(self.map_height, agent_pos[0] + view_size + 1)
        c_start = max(0, agent_pos[1] - view_size)
        c_end = min(self.map_width, agent_pos[1] + view_size + 1)

        view_area = self.world_map[r_start:r_end, c_start:c_end]

        apple_count = np.count_nonzero(view_area == APPLE)

        # The denominator should be the number of tiles that *can* have apples.
        # Let's consider all non-wall tiles as potential spawn areas for simplicity in the local view.
        non_wall_tiles = np.count_nonzero(view_area != WALL)

        if non_wall_tiles == 0:
            return 0.0

        return apple_count / non_wall_tiles

    def _process_agent_movements(self, actions: Dict[str, int]):
        agent_action_map = {}
        agent_new_positions = {}
        for agent_id in self.agents:
            agent = self._agents[agent_id]

            # --- Handle Immobilization ---
            if agent.is_immobilized():
                action_code = STAY_ACTION_INDEX # Override action
                agent.decrement_immobilization()
            else:
                action_code = actions.get(agent_id, STAY_ACTION_INDEX)

            action_str = ACTION_MEANING[action_code]
            agent_action_map[agent_id] = action_str

            if action_str in TURN_ACTIONS:
                agent.set_orientation(ROTATION_MAP[(agent.get_orientation(), action_str)])
            elif action_str in MOVE_ACTIONS:
                move_vec = self._rotate_vector(MOVE_ACTIONS[action_str], agent.get_orientation())
                agent_new_positions[agent_id] = agent.get_pos() + move_vec
        return agent_action_map, agent_new_positions

    def _resolve_movement_conflicts(self, intended_positions: dict):
        final_positions = {aid: agent.get_pos() for aid, agent in self._agents.items() if aid in self.agents}

        target_cells = {}
        for agent_id, pos in intended_positions.items():
            if not self._is_position_valid(pos) or self.world_map[pos[0], pos[1]] in NON_WALKABLE:
                continue
            pos_tuple = tuple(pos)
            if pos_tuple not in target_cells: target_cells[pos_tuple] = []
            target_cells[pos_tuple].append(agent_id)

        for pos, agent_ids in target_cells.items():
            if len(agent_ids) > 1:
                winner = random.choice(agent_ids)
                final_positions[winner] = np.array(pos)
            else:
                final_positions[agent_ids[0]] = np.array(pos)

        for agent_id, pos in final_positions.items():
            self._agents[agent_id].set_pos(pos)

    def _handle_consumption_and_special_actions(self, agent_action_map: Dict[str, str]):
        shuffled_agents = random.sample(self.agents, len(self.agents))

        for agent_id in shuffled_agents:
            agent = self._agents[agent_id]
            pos = agent.get_pos()

            command = self.llm_task_commands.get(agent_id)
            # Handle apple consumption based on controller command
            if self.world_map[pos[0], pos[1]] == APPLE:
                near_apples_cnt = self._count_nearby_apples(pos[0], pos[1]) - 1
                if self.use_llm and command == "CONSERVE":

                    if near_apples_cnt >= 4: agent.add_reward(APPLE_REWARD)
                    else: agent.add_reward(0 * APPLE_REWARD)
                else:
                    agent.add_reward(APPLE_REWARD)


                self._step_sustainability[agent_id] = (min(near_apples_cnt, 4) - 1) / 4
                self.world_map[pos[0], pos[1]] = EMPTY
                self._agent_step_stats[agent_id]["apples_collected_step"] += 1
                self._agent_cumulative_apples[agent_id] += 1
                # Set cooldown for this apple spawn point
                self.apple_cooldowns[tuple(pos)] = APPLE_RESPAWN_COOLDOWN
            # else:
            #     if self.use_llm and command == "CONSERVE":
            #         agent.add_reward()
                self._step_sustainability[agent_id] = 1

            # Handle special actions like FIRE
            action = agent_action_map.get(agent_id)
            if action == "FIRE":
                agent.add_reward(PENALTY_FIRE)
                self._fire_beam(agent)

    def _fire_beam(self, agent: HarvestAgent):
        firing_direction = ORIENTATION_VECTORS[agent.get_orientation()]
        agent_positions = {tuple(a.get_pos()): a for a_id, a in self._agents.items() if a_id in self.agents}

        curr_pos = agent.get_pos() + firing_direction
        for _ in range(FIRE_BEAM_LENGTH):
            if not self._is_position_valid(curr_pos) or self.world_map[curr_pos[0], curr_pos[1]] in FIRE_BLOCKING_CELLS:
                break

            self.beam_pos.append((curr_pos[0], curr_pos[1], PENALTY_BEAM))

            hit_agent = agent_positions.get(tuple(curr_pos))
            if hit_agent and hit_agent.agent_id != agent.agent_id:
                hit_agent.hit(PENALTY_BEAM)
                break

            curr_pos += firing_direction

    def _update_environment_state(self):
        # Decrement cooldowns
        cooldowns_to_remove = []
        for pos, cooldown in self.apple_cooldowns.items():
            self.apple_cooldowns[pos] -= 1
            if self.apple_cooldowns[pos] <= 0:
                cooldowns_to_remove.append(pos)
        for pos in cooldowns_to_remove:
            del self.apple_cooldowns[pos]

        agent_positions = {tuple(agent.get_pos()) for agent in self._agents.values()}
        for r, c in self.apple_spawn_points:
            if self.world_map[r, c] == EMPTY and (r, c) not in agent_positions and (r, c) not in self.apple_cooldowns:
                num_apples = self._count_nearby_apples(r, c)
                if random.random() < SPAWN_PROB[min(num_apples, 4)]:
                    self.world_map[r, c] = APPLE

    def _count_nearby_apples(self, row, col):
        count = 0
        for r_offset in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
            for c_offset in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                if r_offset**2 + c_offset**2 > APPLE_RADIUS**2: continue
                n_r, n_c = row + r_offset, col + c_offset
                if self._is_position_valid([n_r, n_c]) and self.world_map[n_r, n_c] == APPLE:
                    count += 1
        return count

    def _get_observation(self, agent_id: str) -> np.ndarray:
        agent = self._agents[agent_id]
        task_command = self.llm_task_commands.get(agent_id)

        mask_apples = False

        if self.use_llm:
            if task_command == "CONSERVE":
                mask_apples = True
            # elif task_command == "ATTACK":
            #     mask_apples = True
            #     highlight_agents = True

        map_rgb = self._map_to_colors(
            mask_apples,
            agent_id
        )

        return self._get_agent_view(agent, map_rgb)

    def _get_agent_view(self, agent, full_rgb_map):
        pos = agent.get_pos()
        padded_map = np.pad(full_rgb_map, ((VIEW_PADDING, VIEW_PADDING), (VIEW_PADDING, VIEW_PADDING), (0, 0)), 'constant')
        padded_r, padded_c = pos[0] + VIEW_PADDING, pos[1] + VIEW_PADDING
        view = padded_map[padded_r - HARVEST_VIEW_SIZE : padded_r + HARVEST_VIEW_SIZE + 1,
                          padded_c - HARVEST_VIEW_SIZE : padded_c + HARVEST_VIEW_SIZE + 1]
        return self._rotate_view(agent.get_orientation(), view)

    def _map_to_colors(self, mask_apples=False, requesting_agent_id=None):
        rgb_map = np.zeros((self.map_height, self.map_width, 3), dtype=np.uint8)
        empty_color = DEFAULT_COLOURS[EMPTY]
        conserved_apple_color = DEFAULT_COLOURS[b'C']

        # Pre-calculate agent positions and their conservation status
        # agent_positions_and_commands = {}
        # for agent_id in self.agents:
        #     agent = self._agents[agent_id]
        #     pos = tuple(agent.get_pos())
        #     command = None
        #     if self.use_llm:
        #         command = self.llm_task_commands.get(agent_id)
        map_with_agents = self._get_map_with_agents()

        for r in range(self.map_height):
            for c in range(self.map_width):
                current_pos = (r, c)
                base_char = map_with_agents[r, c] # Get the underlying map character

                # Default color is based on the base character
                color_to_use = DEFAULT_COLOURS.get(base_char, empty_color)

                # Check if an apple is at this position
                if base_char == APPLE and mask_apples:
                    nearby_apples = self._count_nearby_apples(r, c) - (1 if base_char == APPLE else 0)
                    if nearby_apples > 1:
                        color_to_use = DEFAULT_COLOURS[b'A'] # Green
                    else:
                        color_to_use = DEFAULT_COLOURS[b' '] # Black
                    # color_to_use = conserved_apple_color


                rgb_map[r, c, :] = color_to_use

        # Overlay beams (beams are temporary and always on top)
        for r, c, beam_char in self.beam_pos:
            rgb_map[r, c, :] = DEFAULT_COLOURS.get(beam_char, empty_color)

        return rgb_map

    def _get_map_with_agents(self):
        map_view = np.copy(self.world_map)
        for agent in self._agents.values():
            if agent.agent_id in self.agents:
                pos = agent.get_pos()
                map_view[pos[0], pos[1]] = agent.get_agent_char()
        for r, c, beam_char in self.beam_pos:
            map_view[r, c] = beam_char
        return map_view

    def _ascii_to_numpy(self, ascii_map):
        return np.array([[c.encode('ascii') for c in row] for row in ascii_map])

    def _find_points(self, char_to_find):
        return np.argwhere(self.base_map == char_to_find).tolist()

    def _rotate_vector(self, vector, orientation):
        if orientation == "UP": return vector
        if orientation == "DOWN": return -vector
        if orientation == "LEFT": return np.array([-vector[1], vector[0]])
        if orientation == "RIGHT": return np.array([vector[1], -vector[0]])

    def _is_position_valid(self, pos):
        return 0 <= pos[0] < self.map_height and 0 <= pos[1] < self.map_width

    def _rotate_view(self, orientation, view):
        if orientation == "UP": return view
        if orientation == "RIGHT": return np.rot90(view, k=3)
        if orientation == "DOWN": return np.rot90(view, k=2)
        if orientation == "LEFT": return np.rot90(view, k=1)


def env(render_mode='human', **kwargs):
    """
    Creates a PettingZoo AEC environment.

    Args:
        render_mode: The rendering mode ('human', 'rgb_array', or None).
        **kwargs: Other arguments to pass to the CleanupEnv constructor
                  (e.g., num_agents, max_cycles).
    """
    parallel_env = HarvestEnv(render_mode=render_mode, **kwargs)
    aec_env = parallel_to_aec(parallel_env)
    #aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env) # Good for debugging
    #aec_env = wrappers.OrderEnforcingWrapper(aec_env)   # Ensures order
    return aec_env