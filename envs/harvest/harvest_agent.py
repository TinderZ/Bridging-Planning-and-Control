# harvest_agent.py
import numpy as np
from .constants import ORIENTATIONS, AGENT_CHARS

class HarvestAgent:
    """Represents an agent in the Harvest environment."""

    def __init__(self, agent_id_num: int, start_pos: np.ndarray, start_orientation: str, view_len: int):
        """
        Initializes a Harvest Agent.

        Args:
            agent_id_num: The numerical ID of the agent (e.g., 0, 1, 2...).
            start_pos: The starting position (row, col) as a numpy array.
            start_orientation: The starting orientation ('UP', 'DOWN', 'LEFT', 'RIGHT').
            view_len: The agent's view distance (determines observation size).
        """
        if agent_id_num < 0 or agent_id_num >= len(AGENT_CHARS):
             raise ValueError(f"agent_id_num must be between 0 and {len(AGENT_CHARS)-1}")

        self.agent_id = f"agent_{agent_id_num}"
        self.agent_char = AGENT_CHARS[agent_id_num]

        self.pos = np.array(start_pos, dtype=int)
        if start_orientation not in ORIENTATIONS:
            raise ValueError(f"Invalid start_orientation: {start_orientation}")
        self.orientation = start_orientation

        self.row_size = view_len
        self.col_size = view_len

        self.reward_this_turn = 0.0
        self.cumulative_reward = 0.0
        self.immobilized_steps = 0

        self.terminated = False
        self.truncated = False

    def is_immobilized(self) -> bool:
        """Checks if the agent is currently immobilized."""
        return self.immobilized_steps > 0

    def decrement_immobilization(self):
        """Decrements the immobilization counter by one step."""
        if self.immobilized_steps > 0:
            self.immobilized_steps -= 1

    def immobilize(self, steps: int):
        """
        Immobilizes the agent for a given number of steps.
        The agent will be forced to take the STAY action.
        """
        self.immobilized_steps = steps

    def get_pos(self) -> np.ndarray:
        return self.pos

    def set_pos(self, new_pos: np.ndarray):
        self.pos = np.array(new_pos, dtype=int)

    def get_orientation(self) -> str:
        return self.orientation

    def set_orientation(self, new_orientation: str):
        if new_orientation not in ORIENTATIONS:
            raise ValueError(f"Invalid orientation: {new_orientation}")
        self.orientation = new_orientation

    def get_agent_char(self) -> bytes:
        return self.agent_char

    def add_reward(self, reward: float):
        self.reward_this_turn += reward

    def consume_reward(self) -> float:
        reward = self.reward_this_turn
        self.reward_this_turn = 0.0
        return reward

    def add_cumulative_reward(self, reward: float):
        self.cumulative_reward += reward

    def get_cumulative_reward(self) -> float:
        return self.cumulative_reward

    def hit(self, char: bytes):
        from .constants import PENALTY_HIT, IMMOBILIZE_DURATION_HIT, PENALTY_BEAM
        if char == PENALTY_BEAM:
            self.add_reward(PENALTY_HIT)
            self.immobilize(IMMOBILIZE_DURATION_HIT)

    def reset(self, start_pos: np.ndarray, start_orientation: str):
        self.pos = np.array(start_pos, dtype=int)
        self.orientation = start_orientation
        self.reward_this_turn = 0.0
        self.cumulative_reward = 0.0
        self.terminated = False
        self.truncated = False
