# constants.py
import numpy as np

# 地图定义 (Map Definition)
CLEANUP_MAP = [
    "@@@@@@@@@@@@@@@@@@",
    "@RRRRRR     BBBBB@",
    "@HHHHHH      BBBB@",
    "@RRRRRR p   BBBBB@",
    "@RRRRR       BBBB@",
    "@RRRRR    P BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRRR   p  BBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH   P  BBBBB@",
    "@RRRRRR    P BBBB@",
    "@HHHHHH P   BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH    P  BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH  P   BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@RRRRR   P   BBBB@",
    "@HHHHH      BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@@@@@@@@@@@@@@@@@@",
]  # 23 *  16

# 智能体动作映射 (Agent Action Mapping)
# 0: left, 1: right, 2: up, 3: down, 4: stay
# 5: turn_clockwise, 6: turn_counter_clockwise
# 7: fire (penalty), 8: clean
ACTION_MEANING = {
    0: "MOVE_LEFT",
    1: "MOVE_RIGHT",
    2: "MOVE_UP",
    3: "MOVE_DOWN",
    4: "STAY",
    5: "TURN_CLOCKWISE",
    6: "TURN_COUNTERCLOCKWISE",
    7: "FIRE",
    8: "CLEAN",
}
NUM_ACTIONS = len(ACTION_MEANING)

# 动作对应的位移和旋转 (Action Effects)
# Basic movements
MOVE_ACTIONS = {
    "MOVE_LEFT": np.array([0, -1]),
    "MOVE_RIGHT": np.array([0, 1]),
    "MOVE_UP": np.array([-1, 0]),
    "MOVE_DOWN": np.array([1, 0]),
    "STAY": np.array([0, 0]),
}

# Rotations (Matrices for clockwise and counter-clockwise)
# Note: Original code used matrices, here we'll manage orientation directly.
TURN_ACTIONS = {"TURN_CLOCKWISE", "TURN_COUNTERCLOCKWISE"}
# Special actions
SPECIAL_ACTIONS = {"FIRE", "CLEAN"}

# 智能体朝向 (Agent Orientations)
ORIENTATIONS = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
ORIENTATION_VECTORS = {
    "UP": np.array([-1, 0]),
    "RIGHT": np.array([0, 1]),
    "DOWN": np.array([1, 0]),
    "LEFT": np.array([0, -1]),
}
# Rotation mapping: (current_orientation, action) -> new_orientation
ROTATION_MAP = {
    ("UP", "TURN_CLOCKWISE"): "RIGHT",
    ("UP", "TURN_COUNTERCLOCKWISE"): "LEFT",
    ("RIGHT", "TURN_CLOCKWISE"): "DOWN",
    ("RIGHT", "TURN_COUNTERCLOCKWISE"): "UP",
    ("DOWN", "TURN_CLOCKWISE"): "LEFT",
    ("DOWN", "TURN_COUNTERCLOCKWISE"): "RIGHT",
    ("LEFT", "TURN_CLOCKWISE"): "UP",
    ("LEFT", "TURN_COUNTERCLOCKWISE"): "DOWN",
}


# 颜色定义 (Color Definitions)
DEFAULT_COLOURS = {
    b' ': np.array([205, 190, 112], dtype=np.uint8),    # Black background
    b'0': np.array([0, 0, 0], dtype=np.uint8),          # Black background beyond map walls
    b'': np.array([180, 180, 180], dtype=np.uint8),     # Grey board walls
    b'@': np.array([180, 180, 180], dtype=np.uint8),     # Grey board walls
    b'A': np.array([255, 0, 0], dtype=np.uint8),        # red apples
    b'B': np.array([124, 252, 0], dtype=np.uint8),      # green lawns
    b'F': np.array([255, 255, 0], dtype=np.uint8),      # Yellow firing beam (penalty)
    # Cleanup specific colors
    b'C': np.array([100, 255, 255], dtype=np.uint8),     # Cyan cleaning beam
    b'S': np.array([135, 206, 250], dtype=np.uint8),  # Stream cell (original seemed mixed, using one color)
    b'H': np.array([46, 139, 87], dtype=np.uint8),      # Brown waste cells
    b'R': np.array([135, 206, 250], dtype=np.uint8),         # Dark Blue river cell (differentiated from stream)
    
    b'P': np.array([205, 190, 112], dtype=np.uint8),     # Generic agent (any player) - Will be overridden by agent ID
    # Default agent colors (can be extended)
    b'1': np.array([0, 0, 255], dtype=np.uint8),         # Blue
    b'2': np.array([254, 151, 0], dtype=np.uint8),       # Orange
    b'3': np.array([105, 89, 205], dtype=np.uint8),      
    b'4': np.array([204, 0, 204], dtype=np.uint8),       
    b'5': np.array([238, 223, 255], dtype=np.uint8),     
    b'6': np.array([100, 255, 255], dtype=np.uint8),     # Cyan
    b'7': np.array([99, 99, 255], dtype=np.uint8),       # Lavender
    b'8': np.array([250, 204, 16], dtype=np.uint8),      # Yellow
    # Add more agent colors if needed
}

# Cleanup 环境特定参数 (Cleanup Environment Specific Parameters)
CLEANUP_VIEW_SIZE = 7  # Agent's view range (original was 7x7)

FIRE_BEAM_LENGTH = 5
CLEAN_BEAM_LENGTH = 3 
CLEAN_BEAM_LENGTH_VALID = 1
FIRE_BEAM_WIDTH = 1   
CLEAN_BEAM_WIDTH = 3  


PENALTY_HIT = 50      # Penalty for being hit by a fire beam
PENALTY_FIRE = 1      # Cost for firing a penalty beam
CLEAN_REWARD = 0      # Reward/cost for firing a cleaning beam (original was 0)
APPLE_REWARD = 1      # Reward for collecting an apple
GET_AREA_REWARD = 0  # Reward for being in river area when LLM command is "clean up"
STAY_AREA_PENALTY = 0 #-0.1  # Penalty for being in grass area when LLM command is "clean up"

# 概率和阈值 (Probabilities and Thresholds)
THRESHOLD_DEPLETION = 0.4
THRESHOLD_RESTORATION = 0.0
WASTE_SPAWN_PROBABILITY = 0.5
APPLE_RESPAWN_PROBABILITY = 0.05

# --- Constants for Immobilization ---
IMMOBILIZE_DURATION_HIT = 0  # Steps agent is immobilized after being hit
IMMOBILIZE_DURATION_FIRE = 0   # Steps agent is immobilized after firing
STAY_ACTION_INDEX = 4         # Index corresponding to the "STAY" action in ACTION_MEANING



# 地图字符 (Map Characters)
WALL = b'@'
AGENT_START = b'P'
APPLE_SPAWN = b'B'  # grass
WASTE_INIT = b'H'   # Initial waste
RIVER = b'R'        # River tiles (can also hold waste)
STREAM = b'S'       # Stream tiles

EMPTY = b' '
EMPTY1 = b'P'
APPLE = b'A'
WASTE = b'H'
PENALTY_BEAM = b'F'
CLEAN_BEAM = b'C'

AGENT_CHARS = [str(i).encode('ascii') for i in range(1, 7)] # b'1' to b'6'

# 不可通行的地块 (Non-walkable Tiles)
# Original included 'D', 'w', 'W' for switch, not relevant here
NON_WALKABLE = [WALL] #, WASTE] # Agents cannot walk *into* waste  # 污染也可以通行，不然改变观察之后，agent出不去了

# 清洁射线可交互的地块 (Tiles Clean Beam Interacts With)
CLEANABLE_TILES = [WASTE]
CLEANED_TILE_RESULT = [RIVER] # Waste becomes River tile after cleaning

# 阻挡射线的地块 (Tiles That Block Beams)
# Original code had 'blocking_cells=[b"H"]' for CLEAN beam
# and used default 'P' (agent) for FIRE beam. Let's clarify:
FIRE_BLOCKING_CELLS = [WALL] # Penalty beam stops at walls
CLEAN_BLOCKING_CELLS = [WALL]  #, WASTE] # Clean beam stops at walls ! not waste

# 用于计算视野的填充 (Padding for view calculation)
VIEW_PADDING = CLEANUP_VIEW_SIZE

# # 任务完成条件常量 (Task Completion Condition Constants)
# TASK_CLEANUP_WASTE_THRESHOLD = 40      # cleanup任务需要清理的waste数量
# TASK_CLEANUP_APPLE_DENSITY_THRESHOLD = 0.70  # cleanup任务完成的apple密度阈值 (80%)
# TASK_COLLECT_APPLE_DENSITY_THRESHOLD = 0.05  # collect apples任务完成的apple密度阈值 (10%)