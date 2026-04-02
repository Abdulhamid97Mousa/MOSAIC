#!/bin/bash
# Start LLM inference servers for benchmarking.
#
# Usage:
#   ./start_servers.sh vllm    [model]   # Start vLLM on :8000
#   ./start_servers.sh sglang  [model]   # Start SGLang on :30000
#   ./start_servers.sh ollama  [model]   # Start Ollama on :11434
#   ./start_servers.sh stop              # Stop all servers
#
# Default model: Qwen/Qwen2.5-3B-Instruct

MODEL="${2:-Qwen/Qwen2.5-3B-Instruct}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1:-help}" in
    vllm)
        echo "Starting vLLM server with model: $MODEL on port 8000..."
        echo "Install first: pip install vllm"
        python -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --port 8000 \
            --max-model-len 2048 \
            --gpu-memory-utilization 0.5 \
            --dtype auto \
            --enforce-eager
        ;;
    sglang)
        echo "Starting SGLang server with model: $MODEL on port 30000..."
        echo "Install first: pip install 'sglang[all]'"
        python -m sglang.launch_server \
            --model-path "$MODEL" \
            --port 30000 \
            --context-length 2048 \
            --mem-fraction-static 0.5
        ;;
    ollama)
        echo "Starting Ollama with model: $MODEL on port 11434..."
        echo "Install first: curl -fsSL https://ollama.ai/install.sh | sh"
        # Map HuggingFace model names to Ollama names
        OLLAMA_MODEL="$MODEL"
        case "$MODEL" in
            *Qwen2.5-3B*) OLLAMA_MODEL="qwen2.5:3b" ;;
            *Qwen2.5-7B*) OLLAMA_MODEL="qwen2.5:7b" ;;
            *Llama-3.1-8B*) OLLAMA_MODEL="llama3.1:8b" ;;
        esac
        echo "Ollama model name: $OLLAMA_MODEL"
        ollama pull "$OLLAMA_MODEL"
        OLLAMA_NUM_PARALLEL=1 ollama serve &
        sleep 3
        echo "Ollama ready. Model: $OLLAMA_MODEL"
        echo "Run benchmark with: --model $OLLAMA_MODEL"
        ;;
    stop)
        echo "Stopping all inference servers..."
        pkill -f "vllm.entrypoints" 2>/dev/null && echo "  Stopped vLLM"
        pkill -f "sglang.launch_server" 2>/dev/null && echo "  Stopped SGLang"
        pkill -f "ollama serve" 2>/dev/null && echo "  Stopped Ollama"
        echo "Done."
        ;;
    *)
        echo "Usage: $0 {vllm|sglang|ollama|stop} [model]"
        echo ""
        echo "Default model: Qwen/Qwen2.5-3B-Instruct"
        echo ""
        echo "Example workflow:"
        echo "  1. Start server:  ./start_servers.sh vllm"
        echo "  2. Run benchmark: python benchmark.py --backend vllm --url http://localhost:8000/v1 --scenarios realtime"
        echo "  3. Stop server:   ./start_servers.sh stop"
        echo "  4. Repeat for sglang, ollama"
        ;;
esac
