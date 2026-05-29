# controller_social_hub.py
import json
import time
import random
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import os
from envs.cleanup.controller_module import ControllerModule

class ControllerSocialHub:
    """
    控制器社交中心，管理所有控制器模块并协调决策过程
    """
    
    def __init__(self, agent_ids: List[str], controller_type: str = "rule-based", controller_provider: str = "vllm"):
        """
        初始化社交中心
        
        Args:
            agent_ids: 所有智能体的ID列表
            controller_type: 控制器类型 ("rule-based" 或 "real-llm")
            controller_provider: 控制器来源 ("vllm" 或 "API")，默认为vllm(本地模型)
        """
        self.agent_ids = agent_ids
        self.controller_type = controller_type
        self.controller_provider = controller_provider
        
        # 创建每个智能体的控制器模块，传递controller_provider参数
        self.controller_modules: Dict[str, ControllerModule] = {}
        for agent_id in agent_ids:
            self.controller_modules[agent_id] = ControllerModule(agent_id, controller_provider=self.controller_provider)
        
        self.current_round = 0
        
        # 仅在使用真实LLM时初始化API客户端
        if controller_type == "real-llm":
            
            # 游戏提示词
            self.game_prompt = \
"""You are in a 5-player game. The goal is to collect as many apples as possible.

Game Description:
- All players are in an orchard to collect apples or clean the river.
- Collecting apples earns rewards; the more you collect, the higher the reward.
- The growth rate of apples depends on the river's pollution level. The river gets polluted slowly over time.
- When the river is fully polluted, apples will temporarily stop growing.
- It is recommended that players who have collected more apples clean the pollution, while players who have collected fewer apples focus on collecting apples.
- Each round has two phases:
- Phase 1: You will receive environmental information (pollution level, number of apples, and all players' apple collection status), and then discuss strategies with other players.
- Phase 2: You must make a personal choice between "collect apples" and "clean up" based on the discussion and your own thoughts.
"""
            
            # 合并的游戏历史记录 - 存储最近4轮的讨论和决策
            self.game_rounds = []  # 格式: [{"round": 1, ...]
            self.game_history_info = "No game history yet.\n" # 用于prompt的游戏历史字符串
            
            # 发言顺序管理
            self.speaking_order = list(range(len(agent_ids)))  # [0, 1, 2, 3, 4]
            self.current_first_speaker = 0  # 当前轮次第一个发言者的索引
    
    def reset_game(self):
        """
        重置游戏状态，在新游戏开始时调用
        """
        self.current_round = 0
        if hasattr(self, 'game_history_info'):
            self.game_rounds = []
            self.game_history_info = "No game history yet.\n"
            # 重新随机化发言顺序
            self.speaking_order = list(range(len(self.agent_ids)))
            self.current_first_speaker = random.randint(0, len(self.agent_ids) - 1)
        # print("Social hub reset, starting new game")
    
    def process_game_info(self, density_info: Dict[str, float], all_cumulative_apples: Dict[str, int]) -> Dict[str, Dict[str, str]]:
        """
        处理游戏信息并为所有智能体生成决策
        
        Args:
            density_info: 环境密度信息
            all_cumulative_apples: 所有智能体累计收集的苹果数量
            
        Returns:
            所有智能体的决策结果，格式: {agent_id: {"action": "collect apples", "state": "peaceful"}}
        """
        self.current_round += 1
        
        if self.controller_type == "rule-based":
            # 规则基础模式：每个控制器模块独立使用规则决策
            return self._process_rule_based(density_info, all_cumulative_apples)
        elif self.controller_type == "real-llm":
            # 真实LLM模式：进行社交讨论
            return self._process_real_llm(density_info, all_cumulative_apples)
            
        else:
            # 未实现的控制器类型
            print(f"Warning: Unimplemented controller type '{self.controller_type}', using default decision")
            return self._get_default_decisions()
    
    def _process_rule_based(self, density_info: Dict[str, float], all_cumulative_apples: Dict[str, int]) -> Dict[str, Dict[str, str]]:
        """
        使用规则基础方法处理所有智能体的决策
        """
        decisions = {}
        
        for agent_id in self.agent_ids:
            if agent_id in self.controller_modules:
                # 调用每个模块的规则基础决策方法
                module = self.controller_modules[agent_id]
                decision = module._get_rule_based_command(density_info, all_cumulative_apples)
                
                decisions[agent_id] = decision
                
            else:
                decisions[agent_id] = {"action": None, "state": "peaceful"}
        
        # 规则基础模式不需要保存到游戏历史
        
        return decisions
    
    def _process_real_llm(self, density_info: Dict[str, float], all_cumulative_apples: Dict[str, int]) -> Dict[str, Dict[str, str]]:
        """
        使用真实LLM进行社交讨论并生成决策
        """
        # print(f"=== Social hub round {self.current_round} discussion begins ===")
        
        # 构建环境信息
        player_apples = {}
        for agent_id, apples in all_cumulative_apples.items():
            player_num = int(agent_id.split('_')[1]) + 1
            player_apples[f"Player {player_num}"] = apples
        
        # 将玩家按苹果数量排序，生成语言描述
        sorted_players = sorted(player_apples.items(), key=lambda x: x[1], reverse=True)
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
        
        # 按玩家编号顺序构建描述
        player_descriptions = []
        for i in range(1, 6):
            player_name = f"Player {i}"
            if player_name in player_desc_map:
                player_descriptions.append(f"{player_name}: {player_desc_map[player_name]}")
        
        player_apples_desc = ", ".join(player_descriptions)
        
        # 将数值转换为语言表述
        pollution_percentage = density_info.get('pollution_density', 0.0) * 100
        apple_percentage = density_info.get('apple_density', 0.0) * 100
        
        # 污染程度描述
        if pollution_percentage > 40:
            pollution_desc = "fully polluted"
        elif pollution_percentage > 35:
            pollution_desc = "heavily polluted"
        elif pollution_percentage > 31:
            pollution_desc = "moderately polluted"
        elif pollution_percentage > 23:
            pollution_desc = "slightly polluted"
        else:
            pollution_desc = "not polluted"
        
        # 苹果数量描述
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
        
        env_info = \
f"""
Current environment status:
- Pollution level: {pollution_desc}
- Mature apples: {apple_desc}
- Player collection status: {player_apples_desc}
"""
        print(env_info)
        
        # 生成当前轮次的发言顺序
        speaking_order = self._generate_speaking_order()
        # print(f"Speaking order for this round: {[f'Player {i+1}' for i in speaking_order]}")
        
        # 1. 进行讨论，按顺序发言
        current_round_utterances = []  # 存储当前轮次的发言
        # print(f"--- Discussion ---")
        
        for speaker_idx in speaking_order:
            agent_id = self.agent_ids[speaker_idx]
            player_num = speaker_idx + 1
            
            # 构建讨论prompt，包含当前轮次已有的发言
            discussion_prompt = self._build_discussion_prompt(
                player_num, env_info, self.game_history_info, current_round_utterances, self.current_round
            )
            
            # 调用LLM获取发言
            LLMmodule = self.controller_modules[agent_id]
            response = LLMmodule.call_openai_api(
                messages=[{"role": "user", "content": discussion_prompt}],
                max_tokens=100,
                temperature=0.7
            )
            
            utterance = f"Player {player_num}: {response}"
            current_round_utterances.append(utterance)
            print(utterance)
        
        # 构建当前轮次讨论字符串
        current_round_discussion_str = f"\nDiscussion for round {self.current_round}:\n" + "\n".join(current_round_utterances)
        
        # 2. 最终决策阶段 (并行)
        # print("--- Final Decision Phase ---")
        
        def get_decision(agent_id: str) -> tuple[str, Dict[str, str]]:
            """为单个智能体获取决策"""
            player_num = int(agent_id.split('_')[1]) + 1
            decision_prompt = self._build_decision_prompt(
                player_num, env_info, current_round_discussion_str
            )
            
            # 调用LLM获取第一个token的概率分布
            LLMmodule = self.controller_modules[agent_id]
            token_probs = LLMmodule.call_openai_api_with_logprobs(
                messages=[{"role": "user", "content": decision_prompt}],
                target_tokens=["clean", "collect"],
                max_tokens=1,
                temperature=0.5,
                top_logprobs=10
            )
            
            # 根据概率选择决策
            clean_prob = token_probs.get("clean", 0.0)
            collect_prob = token_probs.get("collect", 0.0)
            
            # 归一化概率
            total_prob = clean_prob + collect_prob
            if total_prob > 0:
                clean_prob_normalized = clean_prob / total_prob
                collect_prob_normalized = collect_prob / total_prob
            else:
                # 如果两个概率都是0，默认各50%
                clean_prob_normalized = 0.5
                collect_prob_normalized = 0.5
            
            # 根据概率采样决策
            import random
            if random.random() < clean_prob_normalized:
                action = "clean up"
            else:
                action = "collect apples"
            
            print(f"Player {player_num}'s decision: {action} (clean: {clean_prob_normalized:.2%}, collect: {collect_prob_normalized:.2%})")

            # -------- Function calling: 让大模型选择mask工具（暂不影响return）--------
            # 这里仅演示“tool选择”，不改变当前函数的返回结构与内容。
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "_map_to_colors_mask_apple",
                        "description": 'If the agent decision is "clean up", choose _map_to_colors_mask_apple.',
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "_map_to_colors_mask_waste",
                        "description": 'If the agent decision is "collect apples", choose _map_to_colors_mask_waste.',
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False
                        }
                    }
                }
            ]

            chosen_tool_name = None

            tool_select_prompt = (
                "Select exactly one tool based on the agent's decision, and respond ONLY with a tool call.\n"
                f'agent_decision: "{action}"\n'
                "Rules:\n"
                '- If agent_decision is "clean up", call tool _map_to_colors_mask_apple.\n'
                '- If agent_decision is "collect apples", call tool _map_to_colors_mask_waste.\n'
                "Do not output any other text."
            )
            tool_resp = LLMmodule.client.chat.completions.create(
                model=LLMmodule.model_name,
                messages=[{"role": "user", "content": tool_select_prompt}],
                tools=tools,
                tool_choice="auto",
                temperature=0.0,
                max_tokens=64,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=30
            )

            msg = tool_resp.choices[0].message
            if getattr(msg, "tool_calls", None):
                chosen_tool_name = msg.tool_calls[0].function.name

            if chosen_tool_name:
                print(f"Player {player_num}'s selected tool: {chosen_tool_name}")

            return agent_id, {"action": action, "state": "peaceful", 
                            "clean_prob": clean_prob_normalized, 
                            "collect_prob": collect_prob_normalized,
                            "chosen_tool": chosen_tool_name}

        with ThreadPoolExecutor(max_workers=len(self.agent_ids)) as executor:
            # 并行执行决策
            results = executor.map(get_decision, self.agent_ids)
            decisions = dict(results)
        
        # 3. 保存完整的游戏回合到历史
        self._save_game_round_to_history(env_info, current_round_utterances, decisions)
        
        return decisions
    
    def _generate_speaking_order(self) -> List[int]:
        """
        生成当前轮次的发言顺序
        
        Returns:
            发言顺序列表（包含智能体索引）
        """
        # 从当前第一发言者开始，按顺序排列
        order = []
        for i in range(len(self.agent_ids)):
            speaker_idx = (self.current_first_speaker + i) % len(self.agent_ids)
            order.append(speaker_idx)
        
        # 更新下一轮的第一发言者（循环前进）
        self.current_first_speaker = (self.current_first_speaker + 1) % len(self.agent_ids)
        
        return order
    
    def _save_game_round_to_history(self, env_info: str, current_round_utterances: List[str], 
                                   decisions: Dict[str, Dict[str, str]]):
        """
        保存完整的游戏回合到历史记录
        
        Args:
            env_info: 环境信息
            current_round_utterances: 当前轮次的发言列表
            decisions: 所有智能体的决策
        """
        # 提取动作决策
        action_decisions = {
            agent_id: decision["action"] 
            for agent_id, decision in decisions.items()
        }
        
        # 构建动作字符串
        action_str = "Decision results:"
        for agent_id, action in sorted(action_decisions.items()):
            player_num = int(agent_id.split('_')[1]) + 1
            action_str += f" Player {player_num}: {action};"
        
        # 保存到游戏历史
        round_data = {
            "round": self.current_round,

            "content": f"\nRound {self.current_round}:\n" + f"\n{env_info}\n" + "\n".join(current_round_utterances) + f"\n{action_str}\n"
        }
        
        self.game_rounds.append(round_data)
        
        # 仅保留最近4轮的记录
        if len(self.game_rounds) > 4:
            self.game_rounds = self.game_rounds[-4:]
        
        # 重新构建用于prompt的游戏历史字符串
        if len(self.game_rounds) == 0:
            self.game_history_info = "No game history yet.\n"
        else:
            self.game_history_info = "".join([round_data["content"] for round_data in self.game_rounds]) + "\n"
    
    def _get_default_decisions(self) -> Dict[str, Dict[str, str]]:
        """
        获取默认决策（所有智能体都收集苹果）
        
        Returns:
            默认决策字典
        """
        return {
            agent_id: {"action": None, "state": "peaceful"}
            for agent_id in self.agent_ids
        }
    
    def _build_discussion_prompt(self, player_num: int, env_info: str, game_history_info: str, 
                               current_round_utterances: List[str], discussion_round: int) -> str:
        """
        构建讨论阶段的prompt
        
        Args:
            player_num: 玩家编号 (1-5)
            env_info: 环境信息
            game_history_info: 游戏历史信息
            current_round_utterances: 当前轮次已有的发言列表
            discussion_round: 讨论轮次
            
        Returns:
            讨论prompt
        """
        # 构建当前轮次其他玩家的发言信息
        current_round_discussion = ""
        if current_round_utterances:
            current_round_discussion = f"\nOther players' statements in this round:\n" + "\n".join(current_round_utterances) + "\n"
        
        prompt = f"""{self.game_prompt}

You are Player {player_num}. It is now Phase 1, group discussion. Below is the game history, the current environment status, and the statements from other players in this round. Be aware that other players' statements may not be accurate.

{game_history_info}

Here is the current environment status:
{env_info}

{current_round_discussion}

Hello Player {player_num}, it is now your turn to speak. Your statement should be fluent and concise, without any line breaks.

"""
        
        return prompt
    
    def _build_decision_prompt(self, player_num: int, env_info: str, current_round_discussion: str) -> str:
        """
        构建决策阶段的prompt
        
        Args:
            player_num: 玩家编号 (1-5)
            env_info: 环境信息
            current_round_discussion: 当前轮次的讨论内容
            
        Returns:
            决策prompt
        """
        # 从讨论内容中提取当前玩家的发言
        player_utterance = ""
        lines = current_round_discussion.split('\n')
        for line in lines:
            if line.strip().startswith(f"Player {player_num}:"):
                player_utterance = line.split(":", 1)[1].strip()
                break
        
        prompt = f"""\
{self.game_prompt}

{env_info}

You are Player {player_num}, and this is the discussion from the current round:
{current_round_discussion}

Your statement was: "{player_utterance}"

It is now Phase 2, individual decision. As Player {player_num}, make your decision based on the current environment status, your cumulative apple count, and the discussion in this round.

Please choose from the following two options and provide no other text:
- clean up 
- collect apples
"""
        
        return prompt
    
    def _parse_decision(self, response: str) -> str:
        """
        解析LLM的决策响应
        
        Args:
            response: LLM的响应内容
            
        Returns:
            解析后的动作 ("collect apples" 或 "clean up")
        """
        response_lower = response.lower().strip()
        
        if "collect apples" in response_lower or "collect" in response_lower:
            return "collect apples"
        elif "clean up" in response_lower or "clean" in response_lower:
            return "clean up"
        else:
            # 默认选择收集苹果
            # print(f"Could not parse decision response: {response}, defaulting to no decision")
            return None
        


