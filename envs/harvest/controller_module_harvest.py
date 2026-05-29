# envs/harvest/controller_module_harvest.py
import json
from typing import Dict
from openai import OpenAI

class ControllerModuleHarvest:
    """
    Controller module for a single agent in the Harvest environment.
    This class is responsible for creating agent-specific prompts and calling the LLM.
    """
    def __init__(self, agent_id: str, client: OpenAI, model_name: str):
        """
        Initializes the Controller module for a specific agent.
        Args:
            agent_id: The ID of the agent this module belongs to.
            client: The pre-configured OpenAI client (for vLLM or API).
            model_name: The name of the LLM to use.
        """
        self.agent_id = agent_id
        self.client = client
        self.model_name = model_name

    def get_decision(self, prompt: str) -> Dict[str, str]:
        """
        Calls the LLM with a given prompt to get a single decision for this agent.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=50, # Smaller max tokens for a single decision
                timeout=30
            )
            content = response.choices[0].message.content.strip()
            
            # Basic parsing for "FARM" or "CONSERVE"
            if "CONSERVE" in content.upper():
                action = "CONSERVE"
            else:
                action = "FARM" # Default to FARM
            
            return {"action": action, "state": "default"}

        except Exception as e:
            print(f"LLM Module ({self.agent_id}): API call failed: {e}")
            # Fallback to a default safe action
            return {"action": "FARM", "state": "default"}
