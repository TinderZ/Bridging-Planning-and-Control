# harvest_v2/constants.py
import numpy as np

# --- 地图定义 (Map Definition) ---
HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

# --- 智能体动作映射 (Agent Action Mapping) ---
ACTION_MEANING = {
    0: "MOVE_LEFT",
    1: "MOVE_RIGHT",
    2: "MOVE_UP",
    3: "MOVE_DOWN",
    4: "STAY",
    5: "TURN_CLOCKWISE",
    6: "TURN_COUNTERCLOCKWISE",
    7: "FIRE",
}
NUM_ACTIONS = len(ACTION_MEANING)

# --- 动作效果 (Action Effects) ---
MOVE_ACTIONS = {
    "MOVE_LEFT": np.array([0, -1]),
    "MOVE_RIGHT": np.array([0, 1]),
    "MOVE_UP": np.array([-1, 0]),
    "MOVE_DOWN": np.array([1, 0]),
    "STAY": np.array([0, 0]),
}
TURN_ACTIONS = {"TURN_CLOCKWISE", "TURN_COUNTERCLOCKWISE"}
SPECIAL_ACTIONS = {"FIRE"}
STAY_ACTION_INDEX = 4

# --- 智能体朝向 (Agent Orientations) ---
ORIENTATIONS = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}
ORIENTATION_VECTORS = {
    "UP": np.array([-1, 0]),
    "RIGHT": np.array([0, 1]),
    "DOWN": np.array([1, 0]),
    "LEFT": np.array([0, -1]),
}
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

# --- 颜色定义 (Color Definitions) ---
DEFAULT_COLOURS = {
    b' ': np.array([0, 0, 0], dtype=np.uint8),
    b'0': np.array([0, 0, 0], dtype=np.uint8),
    b'': np.array([180, 180, 180], dtype=np.uint8),
    b'@': np.array([180, 180, 180], dtype=np.uint8),
    b'A': np.array([0, 255, 0], dtype=np.uint8),
    b'F': np.array([255, 255, 0], dtype=np.uint8),
    b'P': np.array([0, 0, 0], dtype=np.uint8),
    b'1': np.array([0, 0, 255], dtype=np.uint8),
    b'2': np.array([254, 151, 0], dtype=np.uint8),
    b'3': np.array([186, 85, 211], dtype=np.uint8),
    b'4': np.array([204, 0, 204], dtype=np.uint8),
    b'5': np.array([238, 223, 255], dtype=np.uint8),
    b'C': np.array([128, 0, 128], dtype=np.uint8), # Conserved Apple Color
}

# --- 环境参数 (Environment Parameters) ---
HARVEST_VIEW_SIZE = 7
VIEW_PADDING = HARVEST_VIEW_SIZE

# --- 射线参数 (Beam Parameters) ---
FIRE_BEAM_LENGTH = 5
FIRE_BEAM_WIDTH = 1

# --- 奖励和惩罚 (Rewards and Penalties) ---
APPLE_REWARD = 1
PENALTY_HIT = -50
PENALTY_FIRE = -1

# --- 定身参数 (Immobilization Parameters) ---
IMMOBILIZE_DURATION_HIT = 0


# --- 苹果再生参数 (Apple Respawn Parameters) ---
APPLE_RADIUS = 2
SPAWN_PROB = [0.0, 0.005, 0.01, 0.02, 0.05]
APPLE_RESPAWN_COOLDOWN = 25

# --- LLM 命令和状态编码 ---
LLM_COMMAND_ENCODING = {"NONE": 0, "FARM": 1, "CONSERVE": 2}

# --- 地图字符 (Map Characters) ---
WALL = b'@'
AGENT_START = b'P'
# In Harvest, 'A' is both the apple and its spawn point.
# We use EMPTY to represent a depleted spawn point.
APPLE_SPAWN = b'A'
EMPTY = b' '
APPLE = b'A'
PENALTY_BEAM = b'F'
AGENT_CHARS = [str(i).encode('ascii') for i in range(1, 10)]

# --- 不可通行的地块 (Non-walkable Tiles) ---
NON_WALKABLE = [WALL]

# --- 阻挡射线的地块 (Tiles That Block Beams) ---
# Agents also block beams, handled in env logic
FIRE_BLOCKING_CELLS = [WALL]
