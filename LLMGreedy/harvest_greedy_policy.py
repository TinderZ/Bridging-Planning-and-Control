import numpy as np
from envs.harvest.constants import DEFAULT_COLOURS

class HarvestGreedyPolicy:
    """
    A simple greedy policy for the Harvest environment.
    - Always targets apples.
    - Uses egocentric view for navigation.
    - Avoids getting stuck by turning.
    """
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # --- Actions ---
        self.ACTION_MOVE_LEFT = 0
        self.ACTION_MOVE_RIGHT = 1
        self.ACTION_MOVE_UP = 2
        self.ACTION_MOVE_DOWN = 3
        self.ACTION_STAY = 4
        self.ACTION_TURN_CLOCKWISE = 5
        self.ACTION_TURN_COUNTERCLOCKWISE = 6

        self.EXPLORE_ACTIONS = [
            self.ACTION_MOVE_LEFT, self.ACTION_MOVE_RIGHT,
            self.ACTION_MOVE_UP, self.ACTION_MOVE_DOWN
        ]

        # --- Target Color ---
        self.apple_color = DEFAULT_COLOURS[b'A']

        # --- Opposite Actions ---
        self.opposites = {
            self.ACTION_MOVE_UP: self.ACTION_MOVE_DOWN,
            self.ACTION_MOVE_DOWN: self.ACTION_MOVE_UP,
            self.ACTION_MOVE_LEFT: self.ACTION_MOVE_RIGHT,
            self.ACTION_MOVE_RIGHT: self.ACTION_MOVE_LEFT,
        }

    def compute_action(self, obs: np.ndarray, command: str | None, last_action: int | None, is_stuck: bool) -> int:
        # 0. Handle commands first
        if command == "CONSERVE":
            return self.ACTION_STAY

        # 1. Highest priority: If stuck, turn to un-stuck.
        if is_stuck:
            return np.random.choice([self.ACTION_TURN_CLOCKWISE, self.ACTION_TURN_COUNTERCLOCKWISE])

        # 2. Find apple pixels in the observation
        target_pixels_r, target_pixels_c = np.where(np.all(obs == self.apple_color, axis=-1))

        # 3. If no target in view, explore (avoiding going back)
        if not target_pixels_r.any():
            valid_explore_actions = list(self.EXPLORE_ACTIONS)
            if last_action is not None:
                opposite = self.opposites.get(last_action)
                if opposite in valid_explore_actions:
                    valid_explore_actions.remove(opposite)
            return np.random.choice(valid_explore_actions)

        # 4. Find the closest target in view
        agent_pos_in_view = (obs.shape[0] // 2, obs.shape[1] // 2)
        min_dist = float('inf')
        closest_target_in_view = None

        for r, c in zip(target_pixels_r, target_pixels_c):
            dist = abs(r - agent_pos_in_view[0]) + abs(c - agent_pos_in_view[1])
            if dist < min_dist:
                # If standing on an apple, ignore it to find the next one
                if dist == 0:
                    continue
                min_dist = dist
                closest_target_in_view = (r, c)

        # If all apples are underfoot, explore
        if closest_target_in_view is None:
             return np.random.choice(self.EXPLORE_ACTIONS)

        # 5. Navigate to the closest target (turn to break deadlocks)
        target_r, target_c = closest_target_in_view
        center_r, center_c = agent_pos_in_view
        row_diff = target_r - center_r
        col_diff = target_c - center_c

        best_move = None
        # Determine the single best move direction
        if abs(row_diff) > abs(col_diff):
            best_move = self.ACTION_MOVE_UP if row_diff < 0 else self.ACTION_MOVE_DOWN
        else: # Also handles diagonal case
            best_move = self.ACTION_MOVE_LEFT if col_diff < 0 else self.ACTION_MOVE_RIGHT
        
        # Check if the best move is the opposite of the last one
        if last_action is not None and best_move == self.opposites.get(last_action):
            # If so, we're in a deadlock. Turn randomly to break it.
            return np.random.choice([self.ACTION_TURN_CLOCKWISE, self.ACTION_TURN_COUNTERCLOCKWISE])
        else:
            # Otherwise, execute the best move.
            return best_move 