# run_scripts/train_rllib_centralized.py
import argparse
import os
import sys
from datetime import datetime
import pytz
# import pandas as pd # 新增导入 - 不再需要，因为已移除PlottingCallback
# import functools # Ensure functools is not used for this

import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig # Using PPO directly
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks # 新增导入

# Add near the top with other imports
# import matplotlib.pyplot as plt  # 不再需要，因为已移除PlottingCallback
import numpy as np  # 仍然需要，用于TensorBoard指标计算
from collections import defaultdict


# 新增TensorBoard相关导入
from torch.utils.tensorboard import SummaryWriter
import json

# --- Import your refactored environment and model ---
# Assuming PettingZoo AEC interface for the environment creator
from envs.cleanup.cleanup_env import env as cleanup_env_creator
from envs.harvest.harvest_env import env as harvest_env_creator
# Import other env creators if needed (e.g., harvest)
# Import your refactored model(s)
from models.baseline_model import BaselineModel


from ray.tune.progress_reporter import CLIReporter
from ray.tune.experimental.output import get_air_verbosity, AirVerbosity




class TensorBoardMetricsCallback(DefaultCallbacks):
    """
    Enhanced RLlib Callback that logs custom metrics to TensorBoard for the Harvest environment.
    支持多环境运行器，记录episode级别的指标。
    """

    def __init__(self, num_agents: int = None, tensorboard_logdir: str = None):
        super().__init__()
        self.num_agents = num_agents
        self.agent_ids = [f"agent_{i}" for i in range(num_agents)] if num_agents else []
        self.tb_writer = None
        self.logdir = None
        self.tensorboard_logdir = tensorboard_logdir  # 新增：自定义TensorBoard日志目录
        self.global_step = 0

        # 用于聚合多环境数据
        self.episode_buffers = defaultdict(list)  # 存储多个environment的episode数据

        print(f"TensorBoardMetricsCallback initialized with {num_agents} agents for Harvest.")
        if self.tensorboard_logdir:
            print(f"Custom TensorBoard log directory: {self.tensorboard_logdir}")

    def _ensure_tensorboard_writer(self, logdir):
        """确保TensorBoard writer已初始化"""
        if self.tb_writer is None:
            # 使用自定义的TensorBoard日志目录，如果提供的话
            if self.tensorboard_logdir:
                tb_logdir = self.tensorboard_logdir
                # 创建实验特定的子目录，但不再添加额外的tensorboard_metrics子目录
                if logdir:
                    experiment_name = os.path.basename(os.path.normpath(logdir))
                    tb_logdir = os.path.join(tb_logdir, experiment_name)  # 移除 "tensorboard_metrics"
                # else: 保持原有的tb_logdir
            elif logdir:
                # 直接使用Ray结果目录，不创建子目录
                self.logdir = logdir
                tb_logdir = logdir  # 改为直接使用logdir，而不是子目录
            else:
                # 后备方案：使用默认路径
                tb_logdir = os.path.expanduser("~/ray_results")

            os.makedirs(tb_logdir, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_logdir)
            print(f"TensorBoard writer initialized at: {tb_logdir}")

    def on_algorithm_init(self, *, algorithm, **kwargs):
        """Algorithm初始化时设置TensorBoard"""
        if hasattr(algorithm, 'logdir'):
            self._ensure_tensorboard_writer(algorithm.logdir)

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Episode开始时初始化数据结构"""
        if not self.agent_ids and hasattr(episode, "agent_ids") and episode.agent_ids:
            self.agent_ids = list(episode.agent_ids)
            self.num_agents = len(self.agent_ids)

        # 初始化episode级别的累计数据
        episode.user_data["ep_agent_apples"] = {agent_id: 0 for agent_id in self.agent_ids}
        episode.user_data["ep_agent_collect_cmd"] = {agent_id: 0 for agent_id in self.agent_ids}
        episode.user_data["ep_agent_conserve_cmd"] = {agent_id: 0 for agent_id in self.agent_ids}
        episode.user_data["ep_agent_cumulative_reward"] = {agent_id: 0 for agent_id in self.agent_ids}
        episode.user_data["ep_agent_sustainability"] = {agent_id: 0 for agent_id in self.agent_ids}


    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """每步记录数据"""
        # 获取当前步的信息
        for agent_id in self.agent_ids:
            agent_info = episode.last_info_for(agent_id)
            if agent_info:
                # 累计episode数据
                episode.user_data["ep_agent_apples"][agent_id] += agent_info.get("apples_collected_step", 0)
                
                step_sustainability = agent_info.get("step_sustainability", 0)
                episode.user_data["ep_agent_sustainability"][agent_id] += step_sustainability

                # 根据编码的命令更新特定行为的计数器
                llm_command_encoded = agent_info.get("llm_command_encoded", 0)
                if llm_command_encoded == 1:  # FARM
                    episode.user_data["ep_agent_collect_cmd"][agent_id] += 1
                elif llm_command_encoded == 2:  # CONSERVE
                    episode.user_data["ep_agent_conserve_cmd"][agent_id] += 1


    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        """Episode结束时记录累计指标"""
        ep_agent_apples = episode.user_data.get("ep_agent_apples", {})
        ep_agent_collect_cmd = episode.user_data.get("ep_agent_collect_cmd", {})
        ep_agent_conserve_cmd = episode.user_data.get("ep_agent_conserve_cmd", {}) # Get conserve commands

        ep_agent_sustainability = episode.user_data.get("ep_agent_sustainability", {}) 

        ep_agent_cumulative_reward = {agent_id: episode.last_info_for(agent_id).get("cumulative_reward", 0) for agent_id in self.agent_ids}
        

        # Get environment-wide metrics from any agent's info (assuming they are consistent)
        # Check if agent_ids is not empty before accessing index 0
        apple_density = episode.last_info_for(self.agent_ids[0]).get("apple_density", 0.0) if self.agent_ids else 0.0

        # 计算总和
        total_apples = sum(ep_agent_apples.values())

        # 存储到episode缓冲区，用于后续聚合
        episode_data = {
            'total_apples': total_apples,
            'agent_apples': dict(ep_agent_apples),
            'agent_collect_cmd': dict(ep_agent_collect_cmd),
            'agent_conserve_cmd': dict(ep_agent_conserve_cmd), # Add conserve commands to episode_data
            'agent_sustainability': dict(ep_agent_sustainability),
            'agent_cumulative_reward': dict(ep_agent_cumulative_reward),
            'apple_density': apple_density, # Add apple_density to episode_data
            'episode_length': episode.length,
            'env_index': env_index,
            'worker_index': getattr(worker, 'worker_index', 0),
            'episode_id': episode.episode_id
        }

        # 添加到缓冲区
        self.episode_buffers[worker.worker_index].append(episode_data)

        # 设置RLlib自定义指标（保持兼容性）
        episode.custom_metrics["total_apples_collected"] = total_apples
        episode.custom_metrics["var_apples_fairness"] = np.var(list(ep_agent_apples.values()))
        episode.custom_metrics["std_apples_fairness"] = np.std(list(ep_agent_apples.values()))
        episode.custom_metrics["env/apple_density"] = apple_density # Log apple_density

        # Calculate Gini coefficient
        def calculate_gini(values_list):
            if not values_list:
                return 0.0

            values = np.array(values_list, dtype=np.float64)
            mean_val = np.mean(values)

            if mean_val == 0:
                return 0.0

            n = len(values)
            sum_abs_diff = np.sum(np.abs(values - values[:, np.newaxis]))
            denominator = 2 * n**2 * mean_val
            return sum_abs_diff / denominator

        episode.custom_metrics["gini_apples_fairness"] = calculate_gini(list(ep_agent_apples.values()))
        episode.custom_metrics["sustain_5agents_mean"] = np.mean(list(ep_agent_sustainability.values())) / 1000

        for agent_id in self.agent_ids:
            episode.custom_metrics[f"{agent_id}_apples_collected"] = ep_agent_apples.get(agent_id, 0)
            episode.custom_metrics[f"{agent_id}_sustainability"] = ep_agent_sustainability.get(agent_id, 0) / 1000

            episode.custom_metrics[f"{agent_id}_collect_cmd_count"] = ep_agent_collect_cmd.get(agent_id, 0)
            episode.custom_metrics[f"{agent_id}_conserve_cmd_count"] = ep_agent_conserve_cmd.get(agent_id, 0) # Log conserve commands
            episode.custom_metrics[f"{agent_id}_cumulative_reward"] = ep_agent_cumulative_reward.get(agent_id, 0)
            




    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        """训练结果更新时，聚合并记录所有指标到TensorBoard"""
        self.global_step = result.get("timesteps_total", self.global_step + 1)

        # 确保TensorBoard writer已初始化
        if not self.tb_writer and hasattr(algorithm, 'logdir'):
            self._ensure_tensorboard_writer(algorithm.logdir)

        if not self.tb_writer:
            return

        # 聚合所有worker的episode数据
        all_episode_data = []
        for worker_idx, episodes in self.episode_buffers.items():
            all_episode_data.extend(episodes)

        if all_episode_data:
            #self._aggregate_and_log_episode_metrics(all_episode_data)
            # 清空缓冲区
            self.episode_buffers.clear()

        # 记录训练指标
        self._log_training_metrics(result)

        # 保存TensorBoard数据
        self.tb_writer.flush()



    def _log_training_metrics(self, result):
        """记录训练相关指标"""
        # 基础训练指标
        if "episode_reward_mean" in result:
            self.tb_writer.add_scalar("training/episode_reward_mean", result["episode_reward_mean"], self.global_step)

        if "episode_len_mean" in result:
            self.tb_writer.add_scalar("training/episode_len_mean", result["episode_len_mean"], self.global_step)

        # 学习率
        if "info" in result and "learner" in result["info"]:
            for policy_id, policy_info in result["info"]["learner"].items():
                if "learner_stats" in policy_info:
                    stats = policy_info["learner_stats"]
                    if "curr_lr" in stats:
                        self.tb_writer.add_scalar(f"training/{policy_id}_learning_rate",
                                                stats["curr_lr"], self.global_step)
                    if "policy_loss" in stats:
                        self.tb_writer.add_scalar(f"training/{policy_id}_policy_loss",
                                                stats["policy_loss"], self.global_step)
                    if "vf_loss" in stats:
                        self.tb_writer.add_scalar(f"training/{policy_id}_value_loss",
                                                stats["vf_loss"], self.global_step)
                    if "entropy" in stats:
                        self.tb_writer.add_scalar(f"training/{policy_id}_entropy",
                                                stats["entropy"], self.global_step)

    def close(self):
        """关闭TensorBoard writer"""
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None



# --- Environment Registration ---
# It's common to register environments here before Tune runs.

# +++ START NEW CODE +++
class CombinedRLlibCallbacks(DefaultCallbacks):
    def __init__(self): # Constructor no longer takes num_agents and interval directly
        super().__init__()
        self.tensorboard_cb = None
        self._initialized = False # Guard against multiple initializations
        # print("CombinedRLlibCallbacks: __init__ called") # Debug print

    def _lazy_init(self, source_config): # Renamed algorithm_config to source_config
        if self._initialized:
            return

        # cb_config = source_config.get("callbacks_config") # Use .get() for safety
        # Directly access callbacks_config attribute from AlgorithmConfig
        if not hasattr(source_config, 'callbacks_config') or source_config.callbacks_config is None:
            print("ERROR CombinedRLlibCallbacks: 'callbacks_config' not found or is None in source_config. Cannot initialize.")
            self._initialized = False # Explicitly set to false
            return
        cb_config_dict = source_config.callbacks_config


        if cb_config_dict is None: # Should be caught by hasattr check, but as a safeguard
            print("ERROR CombinedRLlibCallbacks: 'callbacks_config' is None in source_config. Cannot initialize.")
            self._initialized = False # Explicitly set to false
            return

        num_agents = cb_config_dict.get("num_agents")
        tensorboard_logdir = cb_config_dict.get("tensorboard_logdir")  # 新增：获取TensorBoard日志目录

        if num_agents is None:
            print(
                "ERROR CombinedRLlibCallbacks: 'num_agents' not found in "
                "source_config.callbacks_config. Please ensure it is set in PPOConfig().callbacks_config. Cannot initialize."
            )
            self._initialized = False # Explicitly set to false
            return

        print(f"CombinedRLlibCallbacks: Initializing TensorBoard callback with num_agents={num_agents} from {type(source_config)}")
        try:
            # 使用新的TensorBoard回调替换旧的回调，传递tensorboard_logdir参数
            self.tensorboard_cb = TensorBoardMetricsCallback(
                num_agents=num_agents,
                tensorboard_logdir=tensorboard_logdir
            )
            self._initialized = True
            print("CombinedRLlibCallbacks: TensorBoard callback initialized successfully.")
        except Exception as e:
            print(f"ERROR CombinedRLlibCallbacks: Exception during initialization of TensorBoard callback: {e}")
            self._initialized = False # Explicitly set to false

    def on_algorithm_init(self, *, algorithm, **kwargs):
        # print("CombinedRLlibCallbacks: on_algorithm_init called (driver).") # Debug print
        self._lazy_init(algorithm.config)

        if self._initialized and self.tensorboard_cb: # Ensure initialized before forwarding
            if hasattr(self.tensorboard_cb, "on_algorithm_init"):
                self.tensorboard_cb.on_algorithm_init(algorithm=algorithm, **kwargs)

    def on_worker_init(self, *, worker, **kwargs):
        # print(f"CombinedRLlibCallbacks: on_worker_init called (worker PID: {os.getpid()}).") # Debug print
        self._lazy_init(worker.config) # worker.config is an AlgorithmConfig instance

        if self._initialized and self.tensorboard_cb: # Ensure initialized before forwarding
            if hasattr(self.tensorboard_cb, "on_worker_init"):
                self.tensorboard_cb.on_worker_init(worker=worker, **kwargs)

    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        if not self._initialized:
            print(f"ERROR: CombinedRLlibCallbacks.on_episode_start called before initialization (worker PID: {os.getpid()}). Attempting to init.")
            # Fallback: try to initialize if somehow missed. This is a less ideal path.
            # On a worker, worker.config should be available. On driver, this callback might not see `worker`.
            if worker:
                self._lazy_init(worker.config)
            else:
                # If on driver and not initialized, this is more problematic. Relies on on_algorithm_init or on_train_result.
                print(f"ERROR: CombinedRLlibCallbacks.on_episode_start on DRIVER without worker and not initialized.")
                return # Cannot proceed safely
            if not self._initialized: # If still not initialized after attempt
                 print(f"ERROR: CombinedRLlibCallbacks.on_episode_start failed to initialize (worker PID: {os.getpid()}).")
                 return

        if self.tensorboard_cb:
            self.tensorboard_cb.on_episode_start(
                worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs
            )


    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        if not self._initialized:
            print(f"ERROR: CombinedRLlibCallbacks.on_episode_step called before initialization (worker PID: {os.getpid()}). Attempting to init.")
            if worker:
                self._lazy_init(worker.config)
            else:
                print(f"ERROR: CombinedRLlibCallbacks.on_episode_step on DRIVER without worker and not initialized.")
                return
            if not self._initialized: # If still not initialized after attempt
                 print(f"ERROR: CombinedRLlibCallbacks.on_episode_step failed to initialize (worker PID: {os.getpid()}).")
                 return
        if self.tensorboard_cb:
            self.tensorboard_cb.on_episode_step(
                worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs
            )

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        if not self._initialized:
            print(f"ERROR: CombinedRLlibCallbacks.on_episode_end called before initialization (worker PID: {os.getpid()}). Attempting to init.")
            if worker:
                self._lazy_init(worker.config)
            else:
                print(f"ERROR: CombinedRLlibCallbacks.on_episode_end on DRIVER without worker and not initialized.")
                return
            if not self._initialized: # If still not initialized after attempt
                 print(f"ERROR: CombinedRLlibCallbacks.on_episode_end failed to initialize (worker PID: {os.getpid()}).")
                 return
        if self.tensorboard_cb:
            self.tensorboard_cb.on_episode_end(
                worker=worker, base_env=base_env, policies=policies, episode=episode, env_index=env_index, **kwargs
            )

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # print("CombinedRLlibCallbacks: on_train_result called.") # Debug print
        if not self._initialized:
            # This primarily applies to the driver-side callback instance.
            # print("CombinedRLlibCallbacks: Attempting lazy init from on_train_result (driver).") # Debug print
            self._lazy_init(algorithm.config)

        if not self._initialized: # If still not initialized (e.g. config missing crucial keys)
            print("ERROR: CombinedRLlibCallbacks.on_train_result called but still not initialized (driver).")
            return

        # 使用TensorBoard回调的on_train_result方法
        if self.tensorboard_cb:
             self.tensorboard_cb.on_train_result(algorithm=algorithm, result=result, **kwargs)
# +++ END NEW CODE +++


def env_creator(env_config):
    env_name = env_config.get("env_name", "cleanup")
    num_agents = env_config.get("num_agents", 5)
    max_cycles = env_config.get("max_cycles", 1000)

    if env_name == "cleanup":
        aec_env = cleanup_env_creator(
            num_agents=num_agents,
            use_controller=env_config.get("use_controller", False),
            controller_f_step=env_config.get("controller_f_step", 50),
            controller_type=env_config.get("controller_type", "rule-based"),
            max_cycles=max_cycles,
            use_collective_reward=env_config.get("use_collective_reward", False),
            inequity_averse_reward=env_config.get("inequity_averse_reward", False),
            alpha=env_config.get("alpha", 0.0),
            beta=env_config.get("beta", 0.0)
        )
    elif env_name == "harvest":
        aec_env = harvest_env_creator(
            num_agents=num_agents,
            use_llm=env_config.get("use_controller", False),  # Map from controller to llm
            llm_f_step=env_config.get("controller_f_step", 50),
            llm_type=env_config.get("controller_type", "rule-based"),
            max_cycles=max_cycles
        )
    else:
        raise ValueError(f"Unknown environment name: {env_name}")

    return PettingZooEnv(aec_env)

# Environment registration is now handled inside the main() function
# to allow for dynamic environment selection based on command-line arguments.

# --- 
# Register custom models with RLlib
# You can choose unique names or use the class directly in config
ModelCatalog.register_custom_model("baseline_model", BaselineModel)


# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment Identification
    parser.add_argument(
        "--exp_name", type=str, default=None, help="Experiment name prefix."
    )
    parser.add_argument(
        "--env", type=str, default="harvest", choices=["cleanup", "harvest"], # Add others if refactored
        help="Environment name."
    )
    parser.add_argument(
        "--algorithm", type=str, default="PPO", choices=["PPO"], # Extend if needed
        help="RLlib algorithm."
    )
    parser.add_argument(
        "--model", type=str, default="baseline", choices=["baseline"], # Extend if needed
        help="Model architecture."
    )
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="centralized",
        choices=["centralized", "decentralized", "two_policies"],
        help="Defines the multi-agent policy configuration. "
             "'centralized': Single policy for all agents. "
             "'decentralized': One policy per agent. "
             "'two_policies': Agents 0,1 use policy_A, rest policy_B. "
    )
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of trials to run.")
    parser.add_argument( "--seed", type=int, default=None, help="Set seed for reproducibility.")

    # Ray and Tune Control
    parser.add_argument("--local_mode", action="store_true", help="Run Ray in local mode for debugging.")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint if found.")
    parser.add_argument("--restore", type=str, default=None, help="Explicit path to checkpoint to restore from.")
    parser.add_argument("--use_s3", action="store_true", help="Upload results to S3.")
    parser.add_argument("--s3_bucket_prefix", type=str, default="s3://your-bucket-name/ssd-results", help="S3 bucket prefix for uploads.") # CHANGE BUCKET NAME  TODO:default
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="Save checkpoint every N iterations.")
    parser.add_argument("--stop_timesteps", type=int, default=int(500e6), help="Stop after N total env steps.")
    parser.add_argument("--stop_reward", type=float, default=None, help="Stop if avg reward reaches this value.")
    parser.add_argument("--stop_iters", type=int, default=None, help="Stop after N training iterations.")

    # Environment Specific Args (missing max_cycles parameter)
    parser.add_argument("--max_cycles", type=int, default=1000, help="Maximum cycles per episode.")

    # Environment Specific Args (if needed)


    # Resource Allocation (match run script)
    parser.add_argument("--num_env_runners", type=int, default=6, help="Number of environment runners (formerly num_workers).")
    parser.add_argument("--num_envs_per_env_runner", type=int, default=16, help="Number of envs per environment runner (formerly num_envs_per_worker).")
    parser.add_argument("--num_cpus_per_env_runner", type=float, default=1, help="CPUs per environment runner (formerly cpus_per_worker).")
    parser.add_argument("--num_gpus_per_env_runner", type=float, default=0, help="GPUs per environment runner (formerly gpus_per_worker).")
    parser.add_argument("--num_cpus_for_main_process", type=int, default=1, help="CPUs for the main process (driver/trainer, formerly cpus_for_driver).")
    parser.add_argument("--num_gpus", type=float, default=1, help="GPUs for the main process (driver/trainer, formerly gpus_for_driver).")
    parser.add_argument("--episode_timeout_s", type=int, default=600, help="Episode timeout in seconds (RLlib default is ~60s).")

    # Core PPO Hyperparameters (match run script)
    parser.add_argument("--rollout_fragment_length", type=int, default=1000, help="RLlib rollout fragment length.")

    parser.add_argument("--horizon", type=int, default=None, help="Episode horizon (max steps per episode). If None, PPO's default or env max steps will be used.")
    parser.add_argument("--soft_horizon", action="store_true", help="Enable soft horizon. Episodes are truncated at horizon but env is not reset if True.")
    parser.add_argument("--no_done_at_end", action="store_true", help="Set no_done_at_end. If True, the done=True signal will not be set when an episode ends solely due to reaching the horizon. Useful for RNNs.")

    parser.add_argument("--train_batch_size", type=int, default=None, help="RLlib train batch size (if None, calculated).")
    parser.add_argument("--minibatch_size", type=int, default=None, help="RLlib SGD minibatch size (formerly sgd_minibatch_size, if None, calculated).")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of SGD epochs (formerly num_sgd_iter).") # Default PPO is 10-30
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides schedule if set).")
    parser.add_argument("--lr_schedule_steps", nargs="+", type=int, default=[0, 20000000], help="Timesteps for LR schedule points.")
    parser.add_argument("--lr_schedule_weights", nargs="+", type=float, default=[0.00136, 0.000028], help="LR values for schedule points.")
    parser.add_argument("--entropy_coeff", type=float, default=0.000687, help="Entropy coefficient.")
    parser.add_argument("--vf_loss_coeff", type=float, default=0.5, help="Value function loss coefficient (PPO default).") # Common PPO default
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clip parameter (PPO default).") # Common PPO default
    parser.add_argument("--grad_clip", type=float, default=40.0, help="Gradient clipping.")

    # Model Hyperparameters (Specific to your refactored models)
    parser.add_argument("--lstm_hidden_size", type=int, default=128, help="LSTM hidden state size.")
    # Add other model-specific args if they differ from defaults in BaselineModel etc.
    # e.g., parser.add_argument("--fcnet_hiddens", nargs='+', type=int, default=[32, 32])

    # Controller Args (if applicable)
    parser.add_argument("--use_controller", action="store_true", help="Enable controller features in the environment.")
    parser.add_argument("--controller_f_step", type=int, default=50, help="Controller update frequency in steps.")
    parser.add_argument(
        "--controller_type",
        type=str,
        default="rule-based",
        choices=["rule-based", "real-llm", "random"],
        help="Type of controller to use in the environment."
    )

    # TensorBoard Args
    parser.add_argument("--enable_tensorboard", action="store_true", help="Enable TensorBoard logging foopenai-social metrics.")
    parser.add_argument("--tensorboard_port", type=int, default=6006, help="Port for TensorBoard server.")
    parser.add_argument("--tensorboard_logdir", type=str, default=None, help="Custom directory for TensorBoard event files. If not set, will use Ray results directory.")
    parser.add_argument("--storage_path", type=str, default=None, help="Custom directory for Ray training results. If not set, will use ~/ray_results.")

    args = parser.parse_args()

    # Calculate default batch sizes if not provided
    if args.train_batch_size is None:
        args.train_batch_size = args.num_env_runners * args.num_envs_per_env_runner * args.rollout_fragment_length
        print(f"Calculated train_batch_size: {args.train_batch_size}")
    if args.minibatch_size is None:
        # PPO often uses smaller minibatches than the full train batch
        # A common default is 128 or 256, or derived from train_batch_size
        args.minibatch_size = max(128, args.train_batch_size // 16) # Example derivation
        print(f"Calculated minibatch_size: {args.minibatch_size}")

    # Add reasonableness checks for batch sizes
    if args.train_batch_size > 100000:
        print(f"Warning: train_batch_size ({args.train_batch_size}) is very large and may cause memory issues")
    if args.minibatch_size > args.train_batch_size:
        print(f"Warning: minibatch_size ({args.minibatch_size}) is larger than train_batch_size ({args.train_batch_size})")
        args.minibatch_size = min(args.minibatch_size, args.train_batch_size)

    return args


# --- Main Execution ---
def main(args):
    # --- Environment Registration ---
    ENV_NAME_REGISTERED = f"ssd_{args.env}"
    register_env(ENV_NAME_REGISTERED, env_creator)

    # Initialize Ray
    if args.local_mode:
        ray.init(num_cpus=args.num_cpus_for_main_process + args.num_env_runners * args.num_cpus_per_env_runner,
                 local_mode=True)
    else:
        # Connect to existing cluster or start new one
        ray.init(address=os.environ.get("RAY_ADDRESS", None)) # Assumes RAY_ADDRESS is set for clusters

    # 启动TensorBoard服务器（如果启用）
    tensorboard_process = None
    if args.enable_tensorboard:
        import subprocess
        import atexit

        # 确定TensorBoard日志目录
        if args.tensorboard_logdir:
            tb_logdir = args.tensorboard_logdir
        else:
            # 使用训练结果目录作为TensorBoard日志目录
            tb_logdir = args.storage_path if args.storage_path else os.path.expanduser("~/ray_results")

        # 创建TensorBoard日志目录
        os.makedirs(tb_logdir, exist_ok=True)

        print(f"启动TensorBoard服务器，端口: {args.tensorboard_port}, 日志目录: {tb_logdir}")
        try:
            tensorboard_process = subprocess.Popen([
                "tensorboard", "--logdir", tb_logdir,
                "--port", str(args.tensorboard_port),
                "--reload_interval", "30"  # 每30秒刷新一次
            ])
            print(f"TensorBoard已启动，访问: http://localhost:{args.tensorboard_port}")

            # 注册退出时清理TensorBoard进程
            def cleanup_tensorboard():
                if tensorboard_process and tensorboard_process.poll() is None:
                    print("关闭TensorBoard服务器...")
                    tensorboard_process.terminate()
                    tensorboard_process.wait()

            atexit.register(cleanup_tensorboard)

        except FileNotFoundError:
            print("警告: 未找到tensorboard命令，请安装tensorboard: pip install tensorboard")
            args.enable_tensorboard = False
        except Exception as e:
            print(f"启动TensorBoard失败: {e}")
            args.enable_tensorboard = False


    temp_env_config = {
        "env_name": args.env,
        "num_agents": args.num_agents,
        "use_controller": args.use_controller,
        "controller_f_step": args.controller_f_step,
        "controller_type": args.controller_type,  # Add controller_type here
    }
    print(f"[DEBUG main] Creating temp env with config: {temp_env_config}")
    temp_env = env_creator(temp_env_config)
    print(f"[DEBUG main] Temp env created, possible_agents: {temp_env.possible_agents}")
    print(f"[DEBUG main] Temp env observation_space keys: {list(temp_env.observation_space.keys())}")
    obs_space = temp_env.observation_space["agent_0"]
    act_space = temp_env.action_space["agent_0"]
    print(f"[DEBUG main] obs_space type: {type(obs_space)}, obs_space: {obs_space}")
    print(f"[DEBUG main] act_space type: {type(act_space)}, act_space: {act_space}")
    temp_env.close()

    # Dynamically define policies and mapping function ---
    policies_dict = {}
    actual_policy_mapping_fn = None
    print(f"Using policy mode: {args.policy_mode}")
    print(f"[DEBUG main] Model: {args.model}, Algorithm: {args.algorithm}")

    if args.policy_mode == "centralized":
        policies_dict = {
            "shared_policy": PolicySpec(
                # policy_class is None to use the default for the algorithm (e.g., PPO TorchPolicy)
                observation_space=obs_space,
                action_space=act_space,
                # config can be added here to override model or other policy-specific settings
            )
        }
        actual_policy_mapping_fn = lambda agent_id, *a, **kw: "shared_policy"
        print("Policy setup: All agents use 'shared_policy'.")

    elif args.policy_mode == "decentralized":
        policies_dict = {
            f"agent_{i}": PolicySpec(
                observation_space=obs_space,
                action_space=act_space,
            )
            for i in range(args.num_agents)
        }
        # Each agent_id (e.g., "agent_0") maps to a policy_id with the same name.
        actual_policy_mapping_fn = lambda agent_id, *a, **kw: agent_id
        print(f"Policy setup: Each of {args.num_agents} agents uses its own policy (agent_0, agent_1, ...).")

    elif args.policy_mode == "two_policies":
        policies_dict = {
            "policy_A": PolicySpec(observation_space=obs_space, action_space=act_space),
            "policy_B": PolicySpec(observation_space=obs_space, action_space=act_space),
        }
        def mapping_fn_two_policies(agent_id, episode, worker, **kwargs):
            agent_num = int(agent_id.split('_')[-1])
            # Example: First 2 agents use policy_A, the rest use policy_B
            # Adjust this condition based on how you want to split them
            if agent_num < 2: # Agents "agent_0" and "agent_1"
                return "policy_A"
            else:
                return "policy_B"
        actual_policy_mapping_fn = mapping_fn_two_policies
        print("Policy setup: Agents 0 & 1 use 'policy_A', others use 'policy_B'.")

    else:
        raise ValueError(f"Unknown --policy_mode: {args.policy_mode}")


    # --- Configure Algorithm ---
    # Select the correct algorithm class based on the model
    if args.algorithm == "PPO":
        # 对于其他模型如baseline，使用标准PPO
        print(f"Using standard PPO algorithm for {args.model} model.")
        algorithm_class = "PPO"
        config = PPOConfig()
    else:

        raise ValueError(f"Unsupported algorithm configuration: {args.algorithm}")

    # --- Select Model and Prepare Model Config ---
    # 根据模型选择，准备相应的custom_model_config
    model_config = {}

    if args.model == "baseline":
        model_name_registered = "baseline_model"
        model_config = {
             "lstm_hidden_size": args.lstm_hidden_size,
             "conv_filters": [[6, [3, 3], 1]],
             "fcnet_hiddens": [32, 32],
        }

    else:
        raise ValueError(f"Unsupported model: {args.model}")


    # Learning Rate Schedule
    initial_lr = None
    # This variable will hold the schedule if provided, otherwise None
    actual_lr_schedule = None

    if args.lr is not None:
        initial_lr = args.lr
        # actual_lr_schedule remains None
    elif args.lr_schedule_steps and args.lr_schedule_weights:
        # 根据文档，lr_schedule应该是 [[timestep, lr-value], [timestep, lr-value], ...]
        actual_lr_schedule = [[int(step), float(weight)] for step, weight in zip(args.lr_schedule_steps, args.lr_schedule_weights)]

        if actual_lr_schedule:
            initial_lr = actual_lr_schedule[0][1]
        else:
            initial_lr = 0.0001
            actual_lr_schedule = None
    else:
        initial_lr = 0.0001
        # actual_lr_schedule remains None

    # Environment Config (passed to env_creator)
    env_config = {
        "env_name": args.env,
        "num_agents": args.num_agents,
        "use_controller": args.use_controller,
        "controller_f_step": args.controller_f_step,
        "controller_type": args.controller_type, # Pass controller_type to env_config
        "max_cycles": args.max_cycles,
    }

    config = (
        config
        .environment(
            env=ENV_NAME_REGISTERED,
            env_config=env_config,
            disable_env_checking=True # Recommended for multi-agent/complex envs
        )
        .framework("torch") # Or "tf2"
        .env_runners(
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_env_runner,
            num_cpus_per_env_runner=args.num_cpus_per_env_runner,
            num_gpus_per_env_runner=args.num_gpus_per_env_runner,
            rollout_fragment_length=args.rollout_fragment_length,
            sample_timeout_s=args.episode_timeout_s,  # Episode超时时间（秒）
        )
        .training(
            gamma=0.99,
            lr=initial_lr,  # 设置初始学习率
            lambda_=0.95, # GAE lambda (PPO default)
            kl_coeff=0.2, # PPO default
            minibatch_size=args.minibatch_size,
            num_epochs=args.num_epochs,
            train_batch_size=args.train_batch_size,
            vf_loss_coeff=args.vf_loss_coeff,
            entropy_coeff=args.entropy_coeff,
            clip_param=args.clip_param,
            grad_clip=args.grad_clip,
            model={
                "custom_model": model_name_registered,
                "custom_model_config": model_config, # 使用统一构建的model_config
                "use_lstm": False
            },
        )

        .multi_agent(
            policies=policies_dict,
            policy_mapping_fn=actual_policy_mapping_fn,
            # Optional: If you only want to train a subset of policies explicitly
            # policies_to_train=["list_of_policy_ids_to_train_if_needed"]
        )

        .resources(
            num_gpus=args.num_gpus,
            num_cpus_for_main_process=args.num_cpus_for_main_process,
            # num_cpus_per_env_runner and num_gpus_per_env_runner moved to .env_runners()
        )
    )

    # # Apply the generated model config to the main config object
    # config.training(model={"custom_model_config": model_config})


    # 重要：学习率调度应该在所有其他training配置之后设置
    if actual_lr_schedule is not None:
        # 根据文档，lr_schedule参数应该直接传递给training()方法
        config = config.training(lr_schedule=actual_lr_schedule)


    # # Set callback class and config using the builder pattern and direct attribute assignment
    # config = config.callbacks(CombinedRLlibCallbacks) # Use builder pattern
    # config.callbacks_config = { # Assign config dict to the attribute
    #     "num_agents": args.num_agents,
    #     "tensorboard_logdir": args.tensorboard_logdir  # 传递TensorBoard日志目录
    # }

    config = config.callbacks(CombinedRLlibCallbacks) # Use builder pattern
    config.callbacks_config = { # Assign config dict to the attribute
        "num_agents": args.num_agents,
        "tensorboard_logdir": args.tensorboard_logdir  # 传递TensorBoard日志目录
    }


    # Disable new API stack to use ModelV2 custom models   TODO
    config = config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    # --- Stopping Criteria ---
    stop_criteria = {}
    if args.stop_timesteps:
        stop_criteria["timesteps_total"] = args.stop_timesteps
    if args.stop_reward:
        stop_criteria["episode_reward_mean"] = args.stop_reward
    if args.stop_iters:
        stop_criteria["training_iteration"] = args.stop_iters
    if not stop_criteria:
        stop_criteria["training_iteration"] = 100 # Default stop after 100 iters if nothing else set


    # --- Experiment Naming and Storage ---
    experiment_base_name = args.exp_name if args.exp_name else f"{args.env}_{args.model}_{args.algorithm}"
    # Add date/time for uniqueness?
    # timestamp = datetime.now(pytz.timezone("US/Pacific")).strftime("%Y-%m-%d_%H-%M-%S")
    # experiment_full_name = f"{experiment_base_name}_{timestamp}"
    experiment_full_name = experiment_base_name # Keep it simple for now

    # 使用自定义存储路径或默认路径
    if args.storage_path:
        storage_path = args.storage_path
    else:
        storage_path = os.path.expanduser("~/ray_results")

    # 确保存储目录存在
    if not storage_path.startswith("s3://"):  # 只对本地路径创建目录
        os.makedirs(storage_path, exist_ok=True)
        print(f"训练结果将保存到: {storage_path}")

    if args.use_s3:
        # Ensure path ends with / for S3 uploads
        s3_prefix = args.s3_bucket_prefix
        if not s3_prefix.endswith('/'):
            s3_prefix += '/'
        storage_path = s3_prefix



    # --- Setup Tune ---
    tuner = tune.Tuner(
        algorithm_class, # Use the dynamically selected algorithm class
        param_space=config.to_dict(),
        run_config=ray.air.RunConfig(
            name=experiment_full_name,
            stop=stop_criteria,
            storage_path=storage_path,
            checkpoint_config=ray.air.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=True,
                num_to_keep=10
            ),

            callbacks=[],
            # progress_reporter=custom_reporter,  # 使用自定义reporter即可，详细指标在TensorBoard中查看
            verbose=get_air_verbosity(AirVerbosity.VERBOSE),  # 设置详细程度
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            metric="env_runners/episode_reward_mean",
            mode="max",
        ),
    )

    # --- Restore and Run ---
    # if args.resume:
    #      print(f"Attempting to resume experiment: {experiment_full_name} from {storage_path}")
    #      # Note: Tuner automatically handles resuming if the experiment name/path exists
    #      # tuner = tune.Tuner.restore(os.path.join(storage_path, experiment_full_name), trainable=args.algorithm)
    #      tuner = tune.Tuner.restore(os.path.join(storage_path, experiment_full_name), trainable=algorithm_class)
    #      # The above might be needed for specific resume cases, but Tuner(..., run_config=...) often handles it.
    # elif args.restore:
    #      print(f"Restoring experiment from checkpoint: {args.restore}")
    #      # Restore requires the specific trainable and path to checkpoint *directory*
    #      tuner = tune.Tuner.restore(path=args.restore, trainable=algorithm_class)
         # Need to potentially re-apply some config/stop criteria if not in checkpoint?
         # tuner.update_config(...) # Less common, usually restore loads most things

    # Run the experiment(s)
    results = tuner.fit()

    print("Training finished.")
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    # --- 增加检查 ---
    if best_result:
        print("Best trial config: {}".format(best_result.config))
        if best_result.metrics and "episode_reward_mean" in best_result.metrics:
            print("Best trial final reward: {}".format(best_result.metrics["episode_reward_mean"]))
        else:
            print("Best trial found, but 'episode_reward_mean' metric is missing.")
            print(f"All metrics for best trial: {best_result.metrics}")
    else:
        print("No best trial found (likely due to errors or no completed trials).")


    ray.shutdown()



if __name__ == "__main__":
    args = parse_args()
    # Handle potential debug mode setting local_mode
    if sys.gettrace() is not None:
         print("Debug mode detected, forcing local_mode=True")
         args.local_mode = True
         if args.exp_name is None:
             args.exp_name = "debug_experiment" # Override name for debug runs

    main(args)