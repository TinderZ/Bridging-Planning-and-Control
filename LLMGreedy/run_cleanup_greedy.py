import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import argparse
import time
import numpy as np
from envs.cleanup.cleanup_env import CleanupEnv
from envs.cleanup.cleanup_constants import ACTION_MEANING
from cleanup_greedy_policy import CleanupGreedyPolicy

def gini_coefficient(x):
    """Compute Gini coefficient of array of values."""
    x = np.asarray(x, dtype=np.float64)
    # A Gini score of 0 for all zero values is commonly used.
    if np.sum(x) == 0:
        return 0.0
    n = len(x)
    x.sort()
    index = np.arange(1, n + 1)
    # Gini coefficient formula
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def main(args):
    env = CleanupEnv(
        num_agents=args.num_agents,
        render_mode='human',
        max_cycles=args.max_steps,
        use_controller=True,
        controller_f_step=args.llm_f_step,
        controller_type=args.llm_mode
    )

    policies = {f"agent_{i}": CleanupGreedyPolicy(f"agent_{i}") for i in range(args.num_agents)}
    obs, infos = env.reset()
    total_apples = {f"agent_{i}": 0 for i in range(args.num_agents)}
    total_wastes = {f"agent_{i}": 0 for i in range(args.num_agents)}
    last_actions = {agent_id: None for agent_id in env.possible_agents}
    last_positions = {agent_id: None for agent_id in env.possible_agents}
    is_stuck_map = {agent_id: False for agent_id in env.possible_agents}

    for step in range(args.max_steps):
        # 1. 检测上一轮是否有智能体卡住
        for agent_id in env.agents:
            prev_pos = last_positions.get(agent_id)
            prev_action = last_actions.get(agent_id)
            curr_pos = env._agents[agent_id].get_pos()
            
            is_stuck = (
                prev_pos is not None and
                prev_action is not None and
                prev_action in [0, 1, 2, 3] and # 是移动动作
                np.array_equal(prev_pos, curr_pos)
            )
            is_stuck_map[agent_id] = is_stuck

        llm_commands = env.controller_task_commands
        actions = {}
        # 记录行动前的位置
        for agent_id in env.agents:
            last_positions[agent_id] = env._agents[agent_id].get_pos()

        for agent_id in env.agents:
            agent_obs = obs[agent_id]
            command = llm_commands.get(agent_id)
            # 获取智能体全局位置和地图信息
            agent_pos = env._agents[agent_id].get_pos()
            world_map = env.world_map
            current_action = policies[agent_id].compute_action(
                agent_obs, command, agent_pos, world_map, last_actions[agent_id], is_stuck_map[agent_id]
            )
            actions[agent_id] = current_action
            last_actions[agent_id] = current_action

        observations, rewards, terminations, truncations, infos = env.step(actions)
        obs = observations

        # 从infos中提取苹果收集数量
        apples_collected = {agent_id: info.get("apples_collected_step", 0) for agent_id, info in infos.items()}
        wastes_collected = {agent_id: info.get("pollution_cleaned_step", 0) for agent_id, info in infos.items()}

        print(f"--- Step {env.num_cycles}/{args.max_steps} ---")
        print(f"LLM Commands: {llm_commands}")
        action_names = {agent_id: ACTION_MEANING.get(act, f"UNKNOWN({act})") for agent_id, act in actions.items()}
        print(f"Greedy Actions: {action_names}")
        print(f"Apples Collected: {apples_collected}")

        for agent_id, apple_count in apples_collected.items():
            if agent_id in total_apples:
                total_apples[agent_id] += apple_count
        for agent_id, waste_count in wastes_collected.items():
            if agent_id in total_wastes:
                total_wastes[agent_id] += waste_count

        if not env.agents:
            print("Episode finished: All agents are done.")
            break
        # time.sleep(0.2)
    print("\n--- Simulation Finished ---")
    print(f"Total apples per agent: {total_apples}")
    print(f"Total apples: {sum(total_apples.values())}")
    print(f"Total wastes per agent: {total_wastes}")
    print(f"Total wastes: {sum(total_wastes.values())}")
    # Calculate and print Gini coefficient
    apple_counts = list(total_apples.values())
    gini = gini_coefficient(apple_counts)
    print(f"Gini Coefficient of Apple Collection: {gini:.4f}")

    env.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cleanup environment with Greedy Policy controlled by LLM Hub")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps per episode.")
    parser.add_argument("--llm_f_step", type=int, default=80, help="Frequency of LLM controller intervention (in steps).")
    parser.add_argument("--llm_mode", type=str, default="rule-based", choices=["rule-based", "real-llm"], help="LLM controller mode.")
    parser.add_argument("--model_name", type=str, default="qwen2.5-14b-instruct", help="Model name for 'real-llm' mode.")
    args = parser.parse_args()
    main(args)