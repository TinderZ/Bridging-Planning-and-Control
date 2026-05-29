# controller_module.py
from typing import Dict, Optional, List
from openai import OpenAI
import os
import time
import random

# 为通义千问模型定义常量。多个 key 可用逗号分隔：
# export QWEN_API_KEYS="key_1,key_2"
QWEN_API_KEYS = [
    key.strip()
    for key in os.getenv("QWEN_API_KEYS", os.getenv("QWEN_API_KEY", "")).split(",")
    if key.strip()
]

QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen2.5-14b-instruct")

# 为vLLM定义常量
VLLM_BASE_URLS = [
    url.strip()
    for url in os.getenv("VLLM_BASE_URLS", "http://localhost:6006/v1,http://localhost:6011/v1").split(",")
    if url.strip()
]
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen/Qwen3-4B")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY") # vLLM通常不需要API密钥，或者使用任意非空字符串

class ControllerModule:
    """
    Controller module for an agent in the SSD environment.
    """
    def __init__(self, agent_id: str, controller_provider: str = "vllm"):
        """
        Initializes the Controller module for a specific agent.
        Args:
            agent_id: The ID of the agent this module belongs to.
            controller_provider: The controller provider to use, "API" or "vllm". Defaults to "vllm".
        """
        self.agent_id = agent_id
        # 初始化两种命令类型
        self.task_command = None  # "collect apples" 或 "clean up"
        self.state_command = "peaceful"  # "peaceful" 或 "armed"，默认为peaceful
        self.controller_provider = controller_provider
        self.model_name = None
        self.client = None

        api_key = None
        base_url = None

        if self.controller_provider == "vllm":
            api_key = VLLM_API_KEY
            base_url = random.choice(VLLM_BASE_URLS)
            self.model_name = VLLM_MODEL_NAME
            print(f"Controller Module for {self.agent_id} initialized with vLLM client. Model: {self.model_name}, URL: {base_url}")

        elif self.controller_provider == "API":
            if not QWEN_API_KEYS:
                raise ValueError("Set QWEN_API_KEY or QWEN_API_KEYS before using controller_provider='API'.")
            api_key = random.choice(QWEN_API_KEYS)
            base_url = QWEN_BASE_URL
            self.model_name = QWEN_MODEL_NAME
            print(f"Controller Module for {self.agent_id} initialized with API client. Model: {self.model_name}, URL: {base_url}")
        else:
            raise ValueError(f"Unsupported controller_provider: {self.controller_provider}. Choose 'API' or 'vllm'.")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )


    def get_rule_param(self) -> tuple:
        """
        获取规则的三个参数阈值。


        Returns:
            包含三个阈值的元组 (threshold1, threshold2, threshold3)
        """

        return (0.60, 0.3, 0.15)

    def _get_rule_based_command(self, density_info: Dict[str, float], all_cumulative_apples: Dict[str, int]) -> Optional[Dict[str, str]]:
        """
        Determines the task command based on pollution/apple densities and the agent's rank in cumulative apples.

        Args:
            density_info: A dictionary containing pollution_density and apple_density.
            all_cumulative_apples: A dictionary mapping agent IDs to their cumulative apples collected.

        Returns:
            A dictionary with "action" and "state" commands (e.g., {"action": "collect apples", "state": "peaceful"})
            if a command can be determined, otherwise None.
        """
        state_command = "peaceful" # Default state for rule-based
        action_command: Optional[str] = None

        # 验证输入参数
        if not all_cumulative_apples or self.agent_id not in all_cumulative_apples:
            print(f"Warning: Could not determine rank for {self.agent_id}. Apples: {all_cumulative_apples}")
            return None

        # 验证密度信息
        if "apple_density" not in density_info:
            print(f"Warning: apple_density not found in density_info for {self.agent_id}")
            return None

        # Get density values
        apple_density = density_info.get("apple_density", 0.0)  # 保持0-1范围

        # 获取规则参数
        threshold1, threshold2, threshold3 = self.get_rule_param()

        # Sort agents by cumulative apples collected (ascending order)
        # 先将字典转换为列表并打乱顺序，确保相同苹果数量的agent随机排序
        agents_list = list(all_cumulative_apples.items())

        # 为每个agent的苹果数随机增加0、1或2，增加随机性
        agents_list_with_random = []
        for agent_id, apple_count in agents_list:
            random_bonus = random.choice([0, 1, 2])  # 随机选择0、1或2
            adjusted_apple_count = apple_count + random_bonus
            agents_list_with_random.append((agent_id, adjusted_apple_count))

        random.shuffle(agents_list_with_random)
        # 然后按调整后的苹果数量排序（稳定排序会保持相同值的相对顺序）
        sorted_agents = sorted(agents_list_with_random, key=lambda item: item[1])

        # 排名计算
        my_rank = None
        for i, (agent_id, _) in enumerate(sorted_agents):
            if agent_id == self.agent_id:
                my_rank = i + 1
                break

        if my_rank is None:
            print(f"Warning: Agent {self.agent_id} not found in sorted apples during rank calculation.")
            return None

        num_agents = len(sorted_agents)

        # 根据规则确定清理和收集的人数分配，考虑智能体总数限制
        # 苹果密度大于80%：1个人clean（剩余collect）
        # 苹果密度小于80大于50：2个人clean
        # 苹果密度小于50大于15：3个人clean
        # 苹果密度小于15：4个人clean
        if apple_density > 0.80:  # > 80%
            # 苹果密度很高，优先收集苹果：0个人clean，剩余collect
            num_clean = 0
        elif apple_density > threshold1:  # > 80%
            # 苹果密度很高，优先收集苹果：1个人clean，剩余collect
            num_clean = min(1, num_agents)
        elif apple_density > threshold2:  # 40% < 苹果密度 <= 66%
            # 苹果密度较高：2个人clean，剩余collect
            num_clean = min(2, num_agents)
        elif apple_density > threshold3:  # 15% < 苹果密度 <= 40%
            # 苹果密度中等：3个人clean，剩余collect
            num_clean = min(3, num_agents)
        else: #苹果密度 <= 15%
            # 苹果密度中等：3个人clean，剩余collect
            num_clean = min(4, num_agents)


        num_collect = max(0, num_agents - num_clean)

        # Assign commands based on rank and the determined distribution
        # Lower ranks (fewer apples) get priority for collecting
        if my_rank <= num_collect:
            action_command = "collect apples"
        else:
            action_command = "clean up"

        return {"action": action_command, "state": state_command}

    def call_openai_api(self, messages: List[Dict], max_tokens: int = 100, max_retries: int = 3, temperature: float = 0.7) -> str:
        """
        调用配置好的 Controller API

        Args:
            messages: 消息列表
            max_tokens: 最大token数量
            max_retries: 最大重试次数

        Returns:
            API响应内容
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,  # 使用初始化时确定的模型名称
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    #enable_thinking=False,  # 关闭思考模式
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=30
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}) for {self.agent_id} using {self.controller_provider}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    return f"[{self.agent_id}] API调用失败，使用默认响应"

    def call_openai_api_with_logprobs(self, messages: List[Dict], target_tokens: List[str] = None, 
                                       max_tokens: int = 1, max_retries: int = 3, 
                                       temperature: float = 0.5, top_logprobs: int = 10) -> Dict[str, float]:
        """
        调用配置好的 Controller API并返回指定token的概率分布

        Args:
            messages: 消息列表
            target_tokens: 需要获取概率的目标token列表，如["clean", "collect"]
            max_tokens: 最大生成token数量，默认1（只生成第一个token）
            max_retries: 最大重试次数
            temperature: 温度参数
            top_logprobs: 返回top多少个候选token的概率

        Returns:
            包含各目标token概率的字典，如 {"clean": 0.7, "collect": 0.3}
        """
        import math
        
        if target_tokens is None:
            target_tokens = ["clean", "collect"]
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=30
                )
                
                # 解析logprobs
                token_probs = {}
                
                # 初始化目标token的概率为0
                for token in target_tokens:
                    token_probs[token] = 0.0
                
                # 获取第一个token的logprobs
                if (response.choices[0].logprobs and 
                    response.choices[0].logprobs.content and 
                    len(response.choices[0].logprobs.content) > 0):
                    
                    first_token_logprobs = response.choices[0].logprobs.content[0]
                    
                    # 遍历top_logprobs，匹配目标token
                    if first_token_logprobs.top_logprobs:
                        for logprob_info in first_token_logprobs.top_logprobs:
                            token_text = logprob_info.token.lower().strip()
                            prob = math.exp(logprob_info.logprob)  # 将对数概率转换为实际概率
                            
                            # 检查是否匹配目标token（支持部分匹配，如"Clean"匹配"clean"）
                            for target in target_tokens:
                                if target.lower() in token_text or token_text in target.lower():
                                    token_probs[target] = max(token_probs[target], prob)
                
                return token_probs
                
            except Exception as e:
                print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}) for {self.agent_id} using {self.controller_provider}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    # 返回默认概率分布
                    return {token: 1.0 / len(target_tokens) for token in target_tokens}

