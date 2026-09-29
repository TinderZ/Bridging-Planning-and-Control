# Bridging Planning and Control

Code for our [ECML PKDD 2026 paper](https://doi.org/10.1007/978-3-032-37667-1_2) on "Bridging Planning and Control: A Hybrid LLM-RL Framework for Multi-Agent Systems".

This repository contains RLlib/PettingZoo implementations for Cleanup and Harvest, rule-based and LLM-based controllers, greedy baselines, and scripts used to run PPO experiments.

## Repository Layout

```text
.
|-- envs/
|   |-- cleanup/           # Cleanup environment and controller modules
|   `-- harvest/           # Harvest environment and controller modules
|-- models/                # RLlib model definitions
|-- LLMGreedy/             # Greedy baseline runners and policies
|-- tests/                 # Environment checks and LLM/vLLM demos
|-- train_rllib.py         # PPO training entry for Cleanup by default
|-- train_rllib_harvest.py # PPO training entry for Harvest by default
|-- train_rllib*.sh        # Example experiment launch scripts
|-- start_vllm.sh          # Local vLLM OpenAI-compatible server script
`-- requirements.txt
```

## Installation

The code was developed with Python 3.11.

```bash
conda create -n planning-control python=3.11
conda activate planning-control
sudo apt-get update && sudo apt-get install -y swig
pip install -r requirements.txt
```

If you use a local vLLM controller, download or point to a compatible chat model first. For example:

```bash
export VLLM_MODEL_NAME=Qwen/Qwen3-4B
export VLLM_PORT=6006
bash start_vllm.sh
```

For external Qwen/DashScope-compatible API usage, set:

```bash
export QWEN_API_KEY=your_api_key
# Optional:
export QWEN_MODEL_NAME=qwen2.5-14b-instruct
export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Multiple API keys can be supplied with `QWEN_API_KEYS=key1,key2,key3`.

## Quick Start

Run a small Cleanup PPO experiment:

```bash
python train_rllib.py \
  --env cleanup \
  --algorithm PPO \
  --model baseline \
  --policy_mode decentralized \
  --num_agents 5 \
  --num_env_runners 0 \
  --num_envs_per_env_runner 2 \
  --train_batch_size 2000 \
  --minibatch_size 500 \
  --stop_iters 2 \
  --use_controller \
  --controller_type rule-based
```

Run a Harvest PPO experiment:

```bash
python train_rllib_harvest.py \
  --env harvest \
  --algorithm PPO \
  --model baseline \
  --policy_mode decentralized \
  --num_agents 5 \
  --num_env_runners 0 \
  --num_envs_per_env_runner 2 \
  --train_batch_size 2000 \
  --minibatch_size 500 \
  --stop_iters 2 \
  --use_controller \
  --controller_type random
```

The shell scripts `train_rllib.sh`, `train_rllib_harvest.sh`, and `train_rllib_autodl.sh` contain larger experiment configurations. Adjust CPU/GPU counts, batch sizes, and `--storage_path` for your machine.

## Controllers

Controller options:

- `rule-based`: deterministic heuristic controller.
- `random`: random controller baseline.
- `real-llm`: calls an OpenAI-compatible chat API through either local vLLM or an external API.

Useful environment variables:

- `VLLM_MODEL_NAME`: model name/path sent to vLLM, default `Qwen/Qwen3-4B`.
- `VLLM_BASE_URL` or `VLLM_BASE_URLS`: OpenAI-compatible vLLM endpoint(s).
- `VLLM_API_KEY`: API key placeholder for vLLM, default `EMPTY`.
- `QWEN_API_KEY` or `QWEN_API_KEYS`: external API key(s).
- `QWEN_MODEL_NAME`: external model name.
- `QWEN_BASE_URL`: external OpenAI-compatible base URL.

## Baselines and Tests

Greedy baselines:

```bash
python LLMGreedy/run_cleanup_greedy.py --episodes 1 --max_cycles 100
python LLMGreedy/run_harvest_greedy.py --episodes 1 --max_cycles 100
```

Environment tests:

```bash
python -m unittest tests/test_cleanup_env.py
```

vLLM load-test demo:

```bash
export VLLM_CHAT_ENDPOINTS=http://127.0.0.1:6006/v1/chat/completions
export VLLM_MODEL_NAME=Qwen/Qwen3-4B
python tests/test_vllm.py
```

## Notes

Generated training outputs, checkpoints, TensorBoard events, local model weights, and private data are intentionally ignored by Git. Keep API keys in environment variables rather than source files.

## Citation

If you use this code, please cite the [published paper](https://doi.org/10.1007/978-3-032-37667-1_2):

```bibtex
@inproceedings{zhang2027bridging,
  title     = {Bridging Planning and Control: A Hybrid LLM-RL Framework for Multi-agent Systems},
  author    = {Zhang, Zhurun and Ma, Xu and Yuan, Haokuan and Wei, Guni and Zhang, Bolei},
  booktitle = {Machine Learning and Knowledge Discovery in Databases. Research Track},
  series    = {Lecture Notes in Computer Science},
  volume    = {16944},
  pages     = {23--39},
  year      = {2027},
  publisher = {Springer},
  address   = {Cham},
  doi       = {10.1007/978-3-032-37667-1_2}
}
```
