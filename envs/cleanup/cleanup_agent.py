# cleanup_agent.py
import numpy as np
from envs.cleanup.cleanup_constants import ORIENTATIONS, AGENT_CHARS

class SSDAgent:
    """Represents an agent in the Cleanup environment."""

    def __init__(self, agent_id_num: int, start_pos: np.ndarray, start_orientation: str, view_len: int):
        """
        Initializes a Cleanup Agent.

        Args:
            agent_id_num: The numerical ID of the agent (e.g., 0, 1, 2...).
            start_pos: The starting position (row, col) as a numpy array.
            start_orientation: The starting orientation ('UP', 'DOWN', 'LEFT', 'RIGHT').
            view_len: The agent's view distance (determines observation size).
        """
        if agent_id_num < 0 or agent_id_num >= len(AGENT_CHARS):
             raise ValueError(f"agent_id_num must be between 0 and {len(AGENT_CHARS)-1}")

        self.agent_id = f"agent_{agent_id_num}"
        self.agent_char = AGENT_CHARS[agent_id_num] # e.g., b'1', b'2'

        self.pos = np.array(start_pos, dtype=int)
        if start_orientation not in ORIENTATIONS:
            raise ValueError(f"Invalid start_orientation: {start_orientation}")
        self.orientation = start_orientation # String 'UP', 'DOWN', 'LEFT', 'RIGHT'

        # Observation dimensions
        self.row_size = view_len
        self.col_size = view_len

        # State managed by environment, but agent might track rewards internally if needed
        self.reward_this_turn = 0.0
        self.cumulative_reward = 0.0

        # Required by PettingZoo API (though often managed by env)
        self.terminated = False
        self.truncated = False

        self.immobilized_steps_remaining = 0


    def get_pos(self) -> np.ndarray:
        """Returns the current position."""
        return self.pos

    def set_pos(self, new_pos: np.ndarray):
        """Sets the agent's position."""
        self.pos = np.array(new_pos, dtype=int)

    def get_orientation(self) -> str:
        """Returns the current orientation string."""
        return self.orientation

    def set_orientation(self, new_orientation: str):
        """Sets the agent's orientation."""
        if new_orientation not in ORIENTATIONS:
            raise ValueError(f"Invalid orientation: {new_orientation}")
        self.orientation = new_orientation

    def get_agent_char(self) -> bytes:
        """Returns the byte character representing the agent on the map."""
        return self.agent_char

    def add_reward(self, reward: float):
        """Adds reward earned in the current step."""
        self.reward_this_turn += reward

    def consume_reward(self) -> float:
        """Returns the accumulated reward and resets the internal counter."""
        # print(f"Agent {self.agent_id} --- reward this turn {self.reward_this_turn}")
        reward = self.reward_this_turn
        # print(f"agent $ {reward}")
        self.reward_this_turn = 0.0
        return reward

    # Methods for cumulative reward ---
    def add_cumulative_reward(self, reward: float):
        """Adds the reward from the current step to the cumulative total."""
        self.cumulative_reward += reward

    def get_cumulative_reward(self) -> float:
        """Returns the total cumulative reward collected by the agent."""
        return self.cumulative_reward


    def set_terminated(self, terminated: bool = True):
        """Sets the terminated status."""
        self.terminated = terminated

    def set_truncated(self, truncated: bool = True):
        """Sets the truncated status."""
        self.truncated = truncated
    
    def immobilize(self, duration: int):
        """Sets the immobilization duration for the agent."""
        self.immobilized_steps_remaining = duration

    def decrement_immobilization(self):
        """Decrements the remaining immobilization steps."""
        if self.immobilized_steps_remaining > 0:
            self.immobilized_steps_remaining -= 1

    def is_immobilized(self) -> bool:
        """Checks if the agent is currently immobilized."""
        return self.immobilized_steps_remaining > 0



    def reset(self, start_pos: np.ndarray, start_orientation: str):
        """Resets the agent's state."""
        self.pos = np.array(start_pos, dtype=int)
        self.orientation = start_orientation
        self.reward_this_turn = 0.0
        self.cumulative_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.immobilized_steps_remaining = 0
