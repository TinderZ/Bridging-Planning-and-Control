import numpy as np
from envs.cleanup.cleanup_constants import DEFAULT_COLOURS, RIVER, WASTE

class CleanupGreedyPolicy:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id

        # --- 可选动作 ---
        # 注意：这里的上下左右是相对于智能体朝向的
        self.ACTION_MOVE_LEFT = 0          # 相对左移
        self.ACTION_MOVE_RIGHT = 1         # 相对右移
        self.ACTION_MOVE_UP = 2            # 相对前进
        self.ACTION_MOVE_DOWN = 3          # 相对后退
        self.ACTION_TURN_CLOCKWISE = 5     # 顺时针旋转
        self.ACTION_TURN_COUNTERCLOCKWISE = 6 # 逆时针旋转
        self.ACTION_CLEAN = 8              # 发射清理光束

        self.EXPLORE_ACTIONS = [
            self.ACTION_MOVE_LEFT, self.ACTION_MOVE_RIGHT,
            self.ACTION_MOVE_UP, self.ACTION_MOVE_DOWN
        ]
        self.CLEANUP_IDLE_ACTIONS = [
            self.ACTION_CLEAN, self.ACTION_MOVE_LEFT, self.ACTION_MOVE_RIGHT,
            self.ACTION_MOVE_UP, self.ACTION_MOVE_DOWN, self.ACTION_TURN_CLOCKWISE,
            self.ACTION_TURN_COUNTERCLOCKWISE
        ]
        self.CLEANUP_IDLE_PROBS = [0.5] + [0.5 / 6] * 6

        # --- 目标颜色 ---
        self.apple_color = DEFAULT_COLOURS[b'A']
        self.waste_color = DEFAULT_COLOURS[b'H']
        self.river_color = DEFAULT_COLOURS[b'R']

        # --- 地图字符 ---
        self.river_char = RIVER # b'R'
        self.waste_char = WASTE # b'H'

        # --- 相反动作映射 ---
        self.opposites = {
            self.ACTION_MOVE_UP: self.ACTION_MOVE_DOWN,
            self.ACTION_MOVE_DOWN: self.ACTION_MOVE_UP,
            self.ACTION_MOVE_LEFT: self.ACTION_MOVE_RIGHT,
            self.ACTION_MOVE_RIGHT: self.ACTION_MOVE_LEFT,
        }

    def compute_action(self, obs: np.ndarray, llm_command: str, agent_pos: np.ndarray, world_map: np.ndarray, last_action: int | None, is_stuck: bool) -> int:
        # 最高优先级：如果上一轮移动被卡住，强制转身
        if is_stuck:
            return np.random.choice([self.ACTION_TURN_CLOCKWISE, self.ACTION_TURN_COUNTERCLOCKWISE])

        # 1. 针对 "clean up" 命令，优先使用全局信息判断是否在目标区域
        if llm_command == "clean up":
            r, c = agent_pos
            tile = world_map[r, c]
            if tile == self.river_char or tile == self.waste_char:
                # 如果在清理区域上，按指定概率选择动作，但要排除相反动作
                valid_actions = list(self.CLEANUP_IDLE_ACTIONS)
                valid_probs = list(self.CLEANUP_IDLE_PROBS)

                if last_action is not None:
                    opposite = self.opposites.get(last_action)
                    if opposite in valid_actions:
                        idx = valid_actions.index(opposite)
                        prob_to_remove = valid_probs.pop(idx)
                        valid_actions.pop(idx)
                        # 重新归一化概率
                        if sum(valid_probs) > 0:
                            valid_probs = np.array(valid_probs) / (1 - prob_to_remove)
                
                if valid_actions:
                    return np.random.choice(valid_actions, p=valid_probs)
                else: # 如果所有动作都被排除了（不太可能），就随机探索
                    pass # 会进入下面的随机探索逻辑

        # --- 如果不在清理区，或任务是其他（如收集苹果），则使用局部视野进行导航 ---

        # 2. 根据高级指令确定视野内的目标颜色
        target_colors = []
        if llm_command == "clean up":
            target_colors = [self.waste_color, self.river_color]
        elif llm_command == "collect apples":
            target_colors = [self.apple_color]
        else: # 默认收集苹果
            target_colors = [self.apple_color]

        # 3. 在视野中寻找所有目标像素
        target_pixels_r, target_pixels_c = [], []
        for color in target_colors:
            r, c = np.where(np.all(obs == color, axis=-1))
            target_pixels_r.extend(r)
            target_pixels_c.extend(c)

        # 4. 如果视野中没有目标，随机探索（排除相反动作）
        if not target_pixels_r:
            valid_explore_actions = list(self.EXPLORE_ACTIONS)
            if last_action is not None:
                opposite = self.opposites.get(last_action)
                if opposite in valid_explore_actions:
                    valid_explore_actions.remove(opposite)
            return np.random.choice(valid_explore_actions)

        # 5. 找到视野内最近的目标
        agent_pos_in_view = (obs.shape[0] // 2, obs.shape[1] // 2)
        min_dist = float('inf')
        closest_target_in_view = None

        for r, c in zip(target_pixels_r, target_pixels_c):
            dist = abs(r - agent_pos_in_view[0]) + abs(c - agent_pos_in_view[1])
            if dist < min_dist:
                # 对于收集苹果任务，如果已经在苹果上(dist=0)，则忽略该苹果，寻找下一个最近的
                if llm_command == "collect apples" and dist == 0:
                    continue
                min_dist = dist
                closest_target_in_view = (r, c)

        if closest_target_in_view is None: # 所有苹果都在脚下，随机移动
             return np.random.choice(self.EXPLORE_ACTIONS)

        # 6. 导航至最近的目标 (通过转身打破僵局)
        target_r, target_c = closest_target_in_view
        center_r, center_c = agent_pos_in_view
        row_diff = target_r - center_r
        col_diff = target_c - center_c

        best_move = None
        # 确定唯一的最佳移动方向
        if abs(row_diff) > abs(col_diff):
            best_move = self.ACTION_MOVE_UP if row_diff < 0 else self.ACTION_MOVE_DOWN
        elif abs(col_diff) > abs(row_diff):
            best_move = self.ACTION_MOVE_LEFT if col_diff < 0 else self.ACTION_MOVE_RIGHT
        else: # 对角线情况，任选一个
            best_move = self.ACTION_MOVE_UP if row_diff < 0 else self.ACTION_MOVE_DOWN
        
        # 检查最佳移动是否与上一步相反
        if last_action is not None and best_move == self.opposites.get(last_action):
            # 如果是，说明陷入僵局，通过随机转身来打破
            return np.random.choice([self.ACTION_TURN_CLOCKWISE, self.ACTION_TURN_COUNTERCLOCKWISE])
        else:
            # 否则，正常执行最佳移动
            return best_move