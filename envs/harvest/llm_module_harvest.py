# harvest_v2/llm_module_harvest.py
from typing import Dict, Optional, List
from openai import OpenAI
import time
import random

class LLMModuleHarvest:
    def __init__(self, agent_id: str, client: OpenAI, model_name: str, llm_provider: str = "vllm"):
        self.agent_id = agent_id
        self.llm_provider = llm_provider
        self.client = client
        self.model_name = model_name

    def get_rule_based_command(self, env_state: Dict, num_cycles: int) -> Dict[str, str]:
        """
        根据环境状态和智能体的社会排名，决定是采集还是保育。
        包含一个基于时间的两阶段策略。
        """
        # Phase 1: Initial Random Exploration
        if num_cycles < 20:
            action = random.choice(["FARM", "CONSERVE"])
            return {"action": action, "state": "initial_exploration"}

        # Phase 2: Rule-Based Cooperative Strategy
        apple_density = env_state.get("apple_density", 0.0)
        agent_status = env_state.get("agent_status", {})
        
        # 从 agent_status 中提取每个智能体的苹果收集数
        agent_rewards = {
            agent_id: status.get("apples_collected", 0)
            for agent_id, status in agent_status.items()
        }

        # --- 1. Input Validation ---
        if not agent_rewards or self.agent_id not in agent_rewards:
            return {"action": "FARM", "state": "cooperative"}

        num_agents = len(agent_rewards)
        if num_agents == 0:
            return {"action": "FARM", "state": "cooperative"}

        # --- 2. Social Ranking ---
        sorted_agents = sorted(agent_rewards.items(), key=lambda item: item[1])
        my_rank = -1
        for i, (agent_id, reward) in enumerate(sorted_agents):
            if agent_id == self.agent_id:
                my_rank = i + 1
                break

        if my_rank == -1: # Safety check
            return {"action": "FARM", "state": "cooperative"}

        # --- 3. Determine Number of Agents to Conserve ---
        if apple_density > 0.6:
            num_agents_to_conserve = 0
        elif apple_density > 0.55:
            num_agents_to_conserve = 1
        elif apple_density > 0.5:
            num_agents_to_conserve = 2
        elif apple_density > 0.45:
            num_agents_to_conserve = 3
        elif apple_density > 0.4:
            num_agents_to_conserve = 4
        else:
            num_agents_to_conserve = 5

        # --- 4. Assign Task Based on Rank ---
        if my_rank > (num_agents - num_agents_to_conserve):
            action_command = "CONSERVE"
        else:
            action_command = "FARM"

        return {"action": action_command, "state": "cooperative"}

    def call_llm_api(self, prompt: str, max_tokens: int = 100, max_retries: int = 3, temperature: float = 0.7) -> str:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}, # Added for Qwen compatibility
                    timeout=30
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"API call failed for {self.agent_id} (attempt {attempt + 1}/{max_retries}) using {self.llm_provider}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"[{self.agent_id}] API call failed, using default response."
