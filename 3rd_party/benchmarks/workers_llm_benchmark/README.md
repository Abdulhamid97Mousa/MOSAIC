# MOSAIC LLM Inference Backend Benchmark

Measures actual latency of local LLM inference backends for RL action selection on an RTX 4090.

## Quick Start

```bash
cd 3rd_party/benchmarks/workers_llm_benchmark
source .venv/bin/activate

# 1. Start a backend (in a separate terminal)
./start_servers.sh vllm

# 2. Run the benchmark
python benchmark.py --backend vllm --url http://localhost:8000/v1 --scenarios realtime

# 3. Stop the server
./start_servers.sh stop

# 4. Repeat for sglang, ollama
```

## Scenarios

| Scenario | Concurrency | Output Tokens | Purpose |
|---|---|---|---|
| `realtime_1tok` | 1 | 1 | Can LLMs do 60Hz Atari? (TTFT only) |
| `realtime_medium` | 1 | 1 | Prefill cost with soccer-length prompt |
| `realtime_long` | 1 | 1 | Worst-case prefill with D&D prompt |
| `single_agent` | 1 | 50 | Standard MOSAIC grid-world agent |
| `soccer_2v2` | 4 | 50 | 4 concurrent agents, tests batching |
| `batch_eval` | 16 | 50 | Script mode evaluation |
| `stress` | 32 | 50 | Maximum throughput |

## Feasibility Thresholds

| Environment | Required TTFT | Frequency |
|---|---|---|
| Atari | < 16ms | 60 Hz |
| ViZDoom | < 28ms | 35 Hz |
| Grid World | < 500ms total | Turn-based |
