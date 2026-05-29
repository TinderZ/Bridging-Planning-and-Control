# envs/harvest/llm_social_hub_harvest.py
import json
import os
import random
from typing import Dict, List
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from .controller_module_harvest import ControllerModuleHarvest
from .llm_module_harvest import LLMModuleHarvest

# --- API/Model Constants ---
QWEN_API_KEYS = [
    key.strip()
    for key in os.getenv("QWEN_API_KEYS", os.getenv("QWEN_API_KEY", "")).split(",")
    if key.strip()
]
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen2.5-14b-instruct")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:6006/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-4B")

class LLMSocialHubHarvest:
    def __init__(self, agent_ids: List[str], llm_type: str, llm_provider: str = "vllm"):
        self.agent_ids = agent_ids
        self.llm_type = llm_type
        self.llm_provider = llm_provider

        # --- Sliding window for game summaries ---
        self.game_summaries = deque(maxlen=3)
        self.start_conserving_density = None
        self.apples_ran_out_step = None
        self.first_conserve_command_issued = False

        # --- Initialize API client and Controller Modules for each agent ---
        self.client, self.model_name = self._initialize_client()
        self.controller_modules: Dict[str, ControllerModuleHarvest] = {
            agent_id: ControllerModuleHarvest(agent_id, self.client, self.model_name)
            for agent_id in self.agent_ids
        }

        # --- Initialize rule-based modules ---
        self.rule_based_modules: Dict[str, LLMModuleHarvest] = {
            agent_id: LLMModuleHarvest(agent_id, self.client, self.model_name, llm_provider=llm_provider) # Pass client and model_name
            for agent_id in self.agent_ids
        }

        # --- Round-table meeting related variables ---
        self.game_prompt = \
"""You are in a 5-player game. The goal is to collect as many apples as possible while ensuring long-term sustainability.

Game Rules:
- All players are in an orchard to collect apples.
- Collecting apples earns rewards; the more you collect, the higher the reward.
- Apples respawn over time. If too many apples are collected, the respawn rate slows down.
- It is recommended that players who have collected fewer apples focus on collecting apples, while players who have collected more apples focus on conserving apples (i.e., not collecting).
- Each round has two phases:
- Phase 1: You will receive environmental information (apple density, and all players' apple collection status), and then discuss strategies with other players.
- Phase 2: You must make a personal choice between "FARM" (collect apples) and "CONSERVE" (do not collect apples) based on the discussion and your own thoughts.
"""
        self.game_rounds = []  # Format: [{"round": 1, ...]
        self.game_history_info = "No game history yet.\n" # String for prompt
        self.speaking_order = list(range(len(agent_ids)))  # [0, 1, 2, 3, 4]
        self.current_first_speaker = 0  # Index of the first speaker for the current round
        self.current_round = 0 # Initialize current round
        self.decision_history = [] # Store decisions for probability calculation
        self.total_farm_decisions = 0
        self.total_conserve_decisions = 0

    def _initialize_client(self):
        """Initializes the OpenAI client based on the provider."""
        if self.llm_provider == "vllm":
            model_name = VLLM_MODEL_NAME
            client = OpenAI(api_key=os.getenv("VLLM_API_KEY", "EMPTY"), base_url=VLLM_BASE_URL)
            print(f"LLM Hub: Initialized for LOCAL vLLM. Model: {model_name}")
        elif self.llm_provider == "api":
            if not QWEN_API_KEYS:
                raise ValueError("Set QWEN_API_KEY or QWEN_API_KEYS before using llm_provider='api'.")
            model_name = QWEN_MODEL_NAME
            client = OpenAI(api_key=random.choice(QWEN_API_KEYS), base_url=QWEN_BASE_URL)
            print(f"LLM Hub: Initialized for EXTERNAL Qwen API. Model: {model_name}")
        else:
            raise ValueError(f"Unsupported llm_provider: '{self.llm_provider}'.")
        return client, model_name

    def reset_game(self):
        """Called at the end of a game to generate a summary and reset social hub state."""
        # Existing game summary logic
        if self.start_conserving_density is not None or self.apples_ran_out_step is not None:
            density_info = f"{self.start_conserving_density:.0%}" if self.start_conserving_density is not None else "N/A"
            depletion_info = f"became scarce around step {self.apples_ran_out_step}" if self.apples_ran_out_step is not None else "did not become scarce"
            summary = f"- In the last game, conservation started when apple density was around {density_info}, and apples {depletion_info}."
            self.game_summaries.append(summary)
        self.start_conserving_density = None
        self.apples_ran_out_step = None
        self.first_conserve_command_issued = False

        # Reset round-table meeting specific states
        self.game_rounds = []
        self.game_history_info = "No game history yet.\n"
        self.speaking_order = list(range(len(self.agent_ids)))
        self.current_first_speaker = random.randint(0, len(self.agent_ids) - 1)
        self.current_round = 0 # Reset current round
        self.decision_history = [] # Reset decision history
        self.total_farm_decisions = 0 # Reset farm decisions count
        self.total_conserve_decisions = 0 # Reset conserve decisions count
        print("Social hub reset, starting new game")

    def get_commands(self, env_state: Dict, current_step: int) -> Dict[str, Dict[str, str]]:
        """Gets decisions for all agents, either via rules or parallel LLM calls."""
        if self.llm_type == "rule-based":
            decisions = {}
            for agent_id in self.agent_ids:
                module = self.rule_based_modules[agent_id]
                # The rule-based command needs the current step/cycle number
                decisions[agent_id] = module.get_rule_based_command(env_state, current_step)
            return decisions
        elif self.llm_type == "real-llm":
            return self._process_real_llm_discussion(env_state, current_step)
        elif self.llm_type == "random":
            decisions = {}
            for agent_id in self.agent_ids:
                decisions[agent_id] = {"action": random.choice(["FARM", "CONSERVE"]), "state": "default"}
            return decisions
        else:
            print(f"Warning: Unimplemented LLM type '{self.llm_type}', using default decision")
            return self._get_default_decisions()

    def _process_real_llm_discussion(self, env_state: Dict, current_step: int) -> Dict[str, Dict[str, str]]:
        """
        Uses real LLM for social discussion and decision making.
        """
        print(f"=== Social hub round {current_step} discussion begins ===")
        self.current_round = current_step # Update current round

        # Build environment information
        player_apples = {}
        for agent_id, status in env_state.get("agent_status", {}).items():
            player_num = int(agent_id.split('_')[1]) + 1
            player_apples[f"Player {player_num}"] = status["apples_collected"]

        # Generate language description of player status
        sorted_players = sorted(player_apples.items(), key=lambda item: item[1])
        player_desc_map = {}

        for i, (player_name, apple_count) in enumerate(sorted_players):
            if i == 0:
                desc = "collected the most apples"
            elif i == 1:
                desc = "collected more apples"
            elif i == 2:
                desc = "collected a medium amount of apples"
            elif i == 3:
                desc = "collected fewer apples"
            else:  # i == 4
                desc = "collected the fewest apples"
            player_desc_map[player_name] = desc

        player_descriptions = []
        for i in range(1, len(self.agent_ids) + 1):
            player_name = f"Player {i}"
            if player_name in player_desc_map:
                player_descriptions.append(f"{player_name}: {player_desc_map[player_name]}")
        player_apples_desc = ", ".join(player_descriptions)

        # --- Build the detailed agent status summary for the prompt ---
        agent_status = env_state.get("agent_status", {})
        agent_status_lines = []
        # Sort agents by apples collected to identify who is falling behind
        sorted_agents_for_summary = sorted(agent_status.items(), key=lambda item: item[1]['apples_collected'])

        for agent_id, status in sorted_agents_for_summary:
            line = (
                f"- Player {agent_id}: "
                f"Sees local density of {status['local_density']:.1%}, "
                f"has collected {status['apples_collected']} apples."
            )
            agent_status_lines.append(line)
        agent_status_summary = "\n".join(agent_status_lines)

        # Convert numerical values to linguistic expressions
        apple_percentage = env_state.get("apple_density", 0.0) * 100

        if apple_percentage > 70:
            apple_desc = "very abundant"
        elif apple_percentage > 50:
            apple_desc = "abundant"
        elif apple_percentage > 30:
            apple_desc = "normal amount"
        elif apple_percentage > 15:
            apple_desc = "scarce"
        else:
            apple_desc = "very scarce"

        env_info = f"""
Current environment status:
- Mature apples: {apple_desc}
- Player collection status: {player_apples_desc}
"""
        print(env_info)

        # Generate speaking order for the current round
        speaking_order = self._generate_speaking_order()
        print(f"Speaking order for this round: {[f'Player {i+1}' for i in speaking_order]}")

        # 1. Discussion phase, agents speak in order
        current_round_utterances = []  # Store utterances for the current round
        print(f"--- Discussion ---")

        for speaker_idx in speaking_order:
            agent_id = self.agent_ids[speaker_idx]
            player_num = speaker_idx + 1

            # Build discussion prompt, including utterances already made in this round
            discussion_prompt = self._build_discussion_prompt(
                player_num, env_info, self.game_history_info, current_round_utterances, current_step, agent_status_summary
            )

            # Call LLM to get utterance
            LLMmodule = self.rule_based_modules[agent_id] # Use rule_based_modules for discussion (LLMModuleHarvest)
            response = LLMmodule.call_llm_api(
                prompt=discussion_prompt,
                max_tokens=100,
                temperature=0.7
            )

            utterance = f"Player {player_num}: {response}"
            current_round_utterances.append(utterance)
            print(utterance)

        # Build current round discussion string
        current_round_discussion_str = f"\nDiscussion for round {current_step}:\n" + "\n".join(current_round_utterances)

        # 2. Final decision phase (in parallel)
        print("--- Final Decision Phase ---")

        def get_decision(agent_id: str) -> tuple[str, Dict[str, str]]:
            """Gets decision for a single agent"""
            player_num = int(agent_id.split('_')[1]) + 1
            decision_prompt = self._build_decision_prompt(
                player_num, env_info, current_round_discussion_str, agent_status_summary, player_apples, agent_status[agent_id]["local_density"]
            )

            # Call LLM to get final decision
            LLMmodule = self.rule_based_modules[agent_id] # Use rule_based_modules for decision (LLMModuleHarvest)
            response = LLMmodule.call_llm_api(
                prompt=decision_prompt,
                max_tokens=10,
                temperature=0.4
            )

            # Parse decision
            action = self._parse_decision(response)
            print(f"Player {player_num}'s decision: {action}")
            return agent_id, {"action": action, "state": "default"}

        with ThreadPoolExecutor(max_workers=len(self.agent_ids)) as executor:
            # Execute decisions in parallel
            results = executor.map(get_decision, self.agent_ids)
            decisions = dict(results)

        # 3. Save complete game round to history
        self._save_game_round_to_history(env_info, current_round_utterances, decisions)

        # Record decisions for probability calculation
        self.decision_history.append(decisions)
        for agent_id, decision_data in decisions.items():
            action = decision_data.get("action")
            if action == "FARM":
                self.total_farm_decisions += 1
            elif action == "CONSERVE":
                self.total_conserve_decisions += 1

        # --- Track key events for end-of-game summary (moved here) ---
        was_conserve_issued = any(d.get("action") == "CONSERVE" for d in decisions.values())
        if was_conserve_issued and not self.first_conserve_command_issued:
            self.first_conserve_command_issued = True
            self.start_conserving_density = env_state.get("apple_density", 0.0)

        # Track if apples become scarce (e.g., density < 15%)
        if env_state.get("apple_density", 0.0) < 0.15 and self.apples_ran_out_step is None:
            self.apples_ran_out_step = current_step

        return decisions

    def _generate_speaking_order(self) -> List[int]:
        """
        Generates the speaking order for the current round.

        Returns:
            List of agent indices in speaking order.
        """
        # Start from the current first speaker, then cycle through
        order = []
        for i in range(len(self.agent_ids)):
            speaker_idx = (self.current_first_speaker + i) % len(self.agent_ids)
            order.append(speaker_idx)

        # Update the first speaker for the next round (cycle forward)
        self.current_first_speaker = (self.current_first_speaker + 1) % len(self.agent_ids)

        return order

    def _save_game_round_to_history(self, env_info: str, current_round_utterances: List[str],
                                   decisions: Dict[str, Dict[str, str]]):
        """
        Saves the complete game round to history.

        Args:
            env_info: Environment information.
            current_round_utterances: List of utterances from the current round.
            decisions: Decisions of all agents.
        """
        # Extract action decisions
        action_decisions = {
            agent_id: decision["action"]
            for agent_id, decision in decisions.items()
        }

        # Build action string
        action_str = "Decision results:"
        for agent_id, action in sorted(action_decisions.items()):
            player_num = int(agent_id.split('_')[1]) + 1
            action_str += f" Player {player_num}: {action};"

        # Save to game history
        round_data = {
            "round": self.current_round,
            "content": f"\nRound {self.current_round}:\n" + f"\n{env_info}\n" + "\n".join(current_round_utterances) + f"\n{action_str}\n"
        }

        self.game_rounds.append(round_data)

        # Keep only the last 4 rounds
        if len(self.game_rounds) > 4:
            self.game_rounds = self.game_rounds[-4:]

        # Rebuild game history string for prompt
        if len(self.game_rounds) == 0:
            self.game_history_info = "No game history yet.\n"
        else:
            self.game_history_info = "".join([round_data["content"] for round_data in self.game_rounds]) + "\n"

    def get_decision_stats(self) -> Dict[str, float]:
        """
        Calculates the total counts of FARM and CONSERVE decisions, and the proportion of FARM decisions.

        Returns:
            A dictionary with 'total_farm_decisions', 'total_conserve_decisions', and 'farm_proportion'.
        """
        total_decisions = self.total_farm_decisions + self.total_conserve_decisions

        farm_proportion = 0.0
        if total_decisions > 0:
            farm_proportion = self.total_farm_decisions / total_decisions

        return {
            "total_farm_decisions": self.total_farm_decisions,
            "total_conserve_decisions": self.total_conserve_decisions,
            "farm_proportion": farm_proportion
        }

    def _get_default_decisions(self) -> Dict[str, Dict[str, str]]:
        """
        Gets default decisions (all agents FARM).

        Returns:
            Default decision dictionary.
        """
        return {
            agent_id: {"action": "FARM", "state": "default"}
            for agent_id in self.agent_ids
        }

    def _build_discussion_prompt(self, player_num: int, env_info: str, game_history_info: str,
                               current_round_utterances: List[str], discussion_round: int, agent_status_summary: str) -> str:
        """
        Builds the prompt for the discussion phase.

        Args:
            player_num: Player number (1-5).
            env_info: Environment information.
            game_history_info: Game history information.
            current_round_utterances: List of utterances already made in the current round.
            discussion_round: Discussion round number.
            agent_status_summary: Summary of all agents' status including local density.

        Returns:
            Discussion prompt.
        """
        # Build other players' statements for the current round
        current_round_discussion = ""
        if current_round_utterances:
            current_round_discussion = f"\nOther players' statements in this round:\n" + "\n".join(current_round_utterances) + "\n"

        prompt = f"""{self.game_prompt}

You are Player {player_num}. It is now Phase 1, group discussion. Below is the game history, the current environment status, and the statements from other players in this round. Be aware that other players' statements may not be accurate.

{game_history_info}

Here is the current environment status:
{env_info}

All Agents' Status:
{agent_status_summary}

{current_round_discussion}

Hello Player {player_num}, it is now your turn to speak. Your statement should be fluent and concise, without any line breaks.

"""

        return prompt

    def _build_decision_prompt(self, player_num: int, env_info: str, current_round_discussion: str, agent_status_summary: str, all_player_apples: Dict[str, int], current_agent_local_density: float) -> str:
        """
        Builds the prompt for the decision phase.

        Args:
            player_num: Player number (1-5).
            env_info: Environment information.
            current_round_discussion: Discussion content from the current round.
            agent_status_summary: Summary of all agents' status including local density.
            all_player_apples: Dictionary of all players' collected apples.
            current_agent_local_density: The local apple density observed by the current agent.

        Returns:
            Decision prompt.
        """
        # Extract current player's utterance from discussion content
        player_utterance = ""
        lines = current_round_discussion.split('\n')
        for line in lines:
            if line.strip().startswith(f"Player {player_num}:"):
                player_utterance = line.split(":", 1)[1].strip()
                break

        # Determine specific guidance based on relative apple count
        sorted_apples = sorted(all_player_apples.values())
        current_player_apple_count = all_player_apples[f"Player {player_num}"]

        guidance = ""
        if current_player_apple_count == sorted_apples[0]: # Collected the fewest apples
            guidance = "Given that you have collected the fewest apples among all players, it is highly recommended that you FARM to catch up and contribute to the team's overall score."
        elif current_player_apple_count == sorted_apples[-1]: # Collected the most apples
            guidance = "Given that you have collected the most apples among all players, it is highly recommended that you CONSERVE to ensure long-term sustainability of the orchard."
        else:
            # For middle players, encourage conservation unless local apples are abundant
            if current_agent_local_density > 0.8: # Only if local apples are very abundant, middle players can consider farming
                guidance = "Your local area has very abundant apples. You can consider FARMING, but remember to prioritize overall sustainability."
            else: # Otherwise, encourage conservation
                guidance = "To ensure long-term sustainability, it is recommended that you CONSERVE. Only FARM if absolutely necessary, especially if your local area is not very abundant."


        prompt = f"""\
{self.game_prompt}

{env_info}

All Agents' Status:
{agent_status_summary}

You are Player {player_num}, and this is the discussion from the current round:
{current_round_discussion}

Your statement was: "{player_utterance}"

It is now Phase 2, individual decision. As Player {player_num}, make your decision based on the current environment status, your cumulative apple count, and the discussion in this round.
{guidance}

Please choose from the following two options and provide no other text:
- FARM
- CONSERVE
"""

        return prompt

    def _parse_decision(self, response: str) -> str:
        """
        Parses the LLM's decision response.

        Args:
            response: LLM's response content.

        Returns:
            Parsed action ("FARM" or "CONSERVE").
        """
        response_lower = response.lower().strip()

        if "farm" in response_lower:
            return "FARM"
        elif "conserve" in response_lower:
            return "CONSERVE"
        else:
            print(f"Could not parse decision response: {response}, defaulting to FARM")
            return "FARM" # Default to FARM if parsing fails
