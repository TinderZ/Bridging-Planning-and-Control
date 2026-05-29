#!/bin/bash

# 设置可见GPU为1
export CUDA_VISIBLE_DEVICES=0

# 脚本：启动 vLLM OpenAI兼容的API服务器

# 要加载的 Hugging Face 模型标识符或本地模型路径
MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen3-4B}"

# vLLM 服务监听的端口
PORT="${VLLM_PORT:-6006}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

echo "正在启动 vLLM 服务..."
echo "模型: $MODEL_NAME"
echo "端口: $PORT"
echo "GPU数量 (张量并行): $TENSOR_PARALLEL_SIZE"
echo "使用本地模型"

# 启动 vLLM API 服务器的命令
# 你可能需要根据你的环境和需求调整其他参数
# 例如 --dtype auto (或 float16, bfloat16), --max-model-len 等
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-num-seqs 24 \
    --trust-remote-code \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-10000}"
    # 你可以在下面添加更多 vLLM 参数，例如:
    # --gpu-memory-utilization 0.9 # 控制GPU显存使用率


echo "vLLM 服务已尝试启动。"
echo "你可以通过 http://localhost:$PORT/v1/completions (或其他端点) 访问API。"
echo "请检查终端输出以获取详细的启动日志和任何潜在错误。"
