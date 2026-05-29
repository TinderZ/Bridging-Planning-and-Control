
import time
import numpy as np
from envs.harvest.harvest_env import HarvestEnv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_controller", action="store_true", help="Enable the LLM controller.")
    parser.add_argument("--llm_type", type=str, default="rule-based", help="Type of LLM logic ('rule-based' or 'real-llm').")
    parser.add_argument("--llm_provider", type=str, default="vllm", help="LLM provider ('vllm' for local or 'api' for external).")
    args = parser.parse_args()

    num_agents = 5
    env = HarvestEnv(
        num_agents=num_agents,
        render_mode='human',
        use_llm=args.use_controller,
        llm_f_step=50,
        llm_type=args.llm_type,
        llm_provider=args.llm_provider
    )
    env.reset()

    for _ in range(1000):
        actions = {agent_id: np.random.randint(0, env.action_space(agent_id).n) for agent_id in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        env.render()
        time.sleep(0.01) # Commented out for max speed

        if not env.agents:
            print("All agents done.")
            break

    if env.use_llm and env.llm_social_hub:
        decision_stats = env.llm_social_hub.get_decision_stats()
        print(f"LLM Decision Statistics: {decision_stats}")

if __name__ == '__main__':
    main()
