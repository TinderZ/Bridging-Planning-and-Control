import sys
import os
import argparse
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from envs.harvest.harvest_env import HarvestEnv
from envs.harvest.constants import ACTION_MEANING
from harvest_greedy_policy import HarvestGreedyPolicy

def gini_coefficient(x):
    """Compute Gini coefficient of array of values."""
    x = np.asarray(x, dtype=np.float64)
    if np.sum(x) == 0:
        return 0.0
    n = len(x)
    x.sort()
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))

def main(args):
    env = HarvestEnv(
        num_agents=args.num_agents,
        render_mode='human',
        max_cycles=args.max_steps,
        use_llm=args.use_llm,
        llm_f_step=args.llm_f_step,
        llm_type=args.llm_type
    )

    policies = {f"agent_{i}": HarvestGreedyPolicy(f"agent_{i}") for i in range(args.num_agents)}
    obs, infos = env.reset()
    total_apples = {f"agent_{i}": 0 for i in range(args.num_agents)}
    last_actions = {agent_id: None for agent_id in env.possible_agents}
    last_positions = {agent_id: None for agent_id in env.possible_agents}
    is_stuck_map = {agent_id: False for agent_id in env.possible_agents}

    for step in range(args.max_steps):
        # 1. Detect if any agent was stuck in the last step
        for agent_id in env.agents:
            prev_pos = last_positions.get(agent_id)
            prev_action = last_actions.get(agent_id)
            curr_pos = env._agents[agent_id].get_pos()
            
            is_stuck = (
                prev_pos is not None and
                prev_action is not None and
                prev_action in [0, 1, 2, 3] and # It was a move action
                np.array_equal(prev_pos, curr_pos)
            )
            is_stuck_map[agent_id] = is_stuck

        # Get LLM commands for this step
        llm_commands = env.llm_task_commands

        actions = {}
        # Record positions before taking action
        for agent_id in env.agents:
            last_positions[agent_id] = env._agents[agent_id].get_pos()

        for agent_id in env.agents:
            agent_obs = obs[agent_id]
            command = llm_commands.get(agent_id)
            current_action = policies[agent_id].compute_action(
                agent_obs, command, last_actions[agent_id], is_stuck_map[agent_id]
            )
            actions[agent_id] = current_action
            last_actions[agent_id] = current_action

        observations, rewards, terminations, truncations, infos = env.step(actions)
        obs = observations

        apples_collected = {agent_id: info.get("apples_collected_step", 0) for agent_id, info in infos.items()}

        print(f"--- Step {env.num_cycles}/{args.max_steps} ---")
        if args.use_llm:
            print(f"LLM Commands: {llm_commands}")
        action_names = {agent_id: ACTION_MEANING.get(act, f"UNKNOWN({act})") for agent_id, act in actions.items()}
        print(f"Greedy Actions: {action_names}")
        print(f"Apples Collected: {apples_collected}")

        for agent_id, apple_count in apples_collected.items():
            if agent_id in total_apples:
                total_apples[agent_id] += apple_count
        if not env.agents:
            print("Episode finished: All agents are done.")
            break
    
    print("\n--- Simulation Finished ---")
    print(f"Total apples per agent: {total_apples}")
    print(f"Total apples: {sum(total_apples.values())}")
    
    apple_counts = list(total_apples.values())
    gini = gini_coefficient(apple_counts)
    print(f"Gini Coefficient of Apple Collection: {gini:.4f}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Harvest environment with Greedy Policy")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum simulation steps per episode.")
    parser.add_argument("--use_llm", action='store_true', help="Enable LLM controller.")
    parser.add_argument("--llm_f_step", type=int, default=80, help="Frequency of LLM controller intervention.")
    parser.add_argument("--llm_type", type=str, default="rule-based", choices=["rule-based", "real-llm"], help="LLM controller type.")
    args = parser.parse_args()
    main(args) 