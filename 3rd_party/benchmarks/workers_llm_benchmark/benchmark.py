#!/usr/bin/env python3
"""LLM Inference Backend Benchmark for MOSAIC.

Measures the actual latency of local LLM inference backends (vLLM, SGLang, Ollama)
for RL action selection. The critical metric is TTFT (time to first token) with
minimal output, since RL action selection only needs 1 token (the action index).

This benchmark answers a question the prior literature (Atari-GPT, TextAtari)
never tested: can a locally-hosted small LLM on a consumer GPU generate actions
fast enough for real-time RL environments?

Usage:
    # Benchmark a single backend (server must be running)
    python benchmark.py --backend vllm --url http://localhost:8000/v1

    # Benchmark with specific model
    python benchmark.py --backend sglang --url http://localhost:30000/v1 --model Qwen/Qwen2.5-3B-Instruct

    # Run all scenarios
    python benchmark.py --backend vllm --url http://localhost:8000/v1 --scenarios all

    # Real-time action selection test (1-token output, the key experiment)
    python benchmark.py --backend vllm --url http://localhost:8000/v1 --scenarios realtime
"""

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Single benchmark run result."""
    backend: str
    model: str
    scenario: str
    concurrency: int
    num_requests: int
    max_tokens: int
    prompt_tokens: int

    # Latency (milliseconds)
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    ttft_min_ms: float = 0.0
    ttft_max_ms: float = 0.0
    ttft_mean_ms: float = 0.0

    total_latency_p50_ms: float = 0.0
    total_latency_p95_ms: float = 0.0
    total_latency_p99_ms: float = 0.0
    total_latency_mean_ms: float = 0.0

    # Throughput
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # GPU
    vram_mb: float = 0.0

    # Feasibility
    atari_viable: str = ""      # need < 16ms TTFT
    vizdoom_viable: str = ""    # need < 28ms TTFT
    grid_world_viable: str = "" # need < 500ms total

    timestamp: str = ""

    def save(self, results_dir: Path):
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"{self.backend}_{self.scenario}_{self.concurrency}c.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ---------------------------------------------------------------------------
# Prompts that simulate MOSAIC LLM worker action selection
# ---------------------------------------------------------------------------

# Short prompt: MiniGrid-style observation (action index output)
PROMPT_SHORT = (
    "You are an agent in a grid world. "
    "You see: wall ahead, key to the right, door behind you. "
    "Valid actions: 0=left, 1=right, 2=forward, 3=pickup, 4=drop, 5=toggle, 6=done. "
    "Output ONLY the action number."
)

# Medium prompt: MOSAIC MultiGrid Soccer observation
PROMPT_MEDIUM = (
    "You are green_0 in a 2v2 soccer game on a 15x15 grid. "
    "Your team: green_0 at (3,7), green_1 at (5,9). "
    "Opponents: blue_0 at (11,6), blue_1 at (10,8). "
    "Ball is at (7,7). Your goal is on the right edge. "
    "Opponent goal is on the left edge. "
    "The ball can be kicked by moving into its cell. "
    "Valid actions: 0=still, 1=left, 2=right, 3=forward, 4=backward, 5=pickup, 6=drop. "
    "You want to score. Output ONLY the action number."
)

# Long prompt: D&D-style detailed state (like the Dayo et al. paper)
PROMPT_LONG = (
    "We are playing a tactical grid game. It is your turn. "
    "You play as P (a level 2 fighter). Enemy E (a level 2 wizard) is nearby. "
    "Your health: 15/18 (83%). Enemy health: 14/14 (100%). "
    "Map (7x7, P=you, E=enemy, .=open, #=wall, ~=water):\n"
    ".......\n"
    ".......\n"
    "...P...\n"
    "..E.#..\n"
    "..~~#..\n"
    "..~~...\n"
    ".......\n"
    "Available actions:\n"
    "0: end turn\n"
    "1: attack with sword\n"
    "2: attack with bow\n"
    "3: move up\n"
    "4: move down\n"
    "5: move left\n"
    "6: move right\n"
    "7: dodge\n"
    "8: use healing potion\n"
    "Output ONLY the action number."
)

PROMPTS = {
    "short": PROMPT_SHORT,
    "medium": PROMPT_MEDIUM,
    "long": PROMPT_LONG,
}

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    concurrency: int
    max_tokens: int
    prompt_key: str
    num_requests: int
    description: str


SCENARIOS = {
    # The key experiment: can LLMs do real-time RL?
    "realtime_1tok": Scenario(
        name="realtime_1tok",
        concurrency=1,
        max_tokens=1,
        prompt_key="short",
        num_requests=50,
        description="1-token output, single request. Measures pure TTFT for real-time viability.",
    ),
    "realtime_medium": Scenario(
        name="realtime_medium",
        concurrency=1,
        max_tokens=1,
        prompt_key="medium",
        num_requests=50,
        description="1-token output with longer prompt. Measures prefill cost.",
    ),
    "realtime_long": Scenario(
        name="realtime_long",
        concurrency=1,
        max_tokens=1,
        prompt_key="long",
        num_requests=50,
        description="1-token output with D&D-length prompt. Worst-case prefill.",
    ),
    # Standard MOSAIC usage: grid-world agent
    "single_agent": Scenario(
        name="single_agent",
        concurrency=1,
        max_tokens=50,
        prompt_key="medium",
        num_requests=30,
        description="1 LLM agent in MiniGrid. Standard usage.",
    ),
    "soccer_2v2": Scenario(
        name="soccer_2v2",
        concurrency=4,
        max_tokens=50,
        prompt_key="medium",
        num_requests=40,
        description="4 LLM agents in 2v2 Soccer. Tests batching.",
    ),
    # Stress tests
    "batch_eval": Scenario(
        name="batch_eval",
        concurrency=16,
        max_tokens=50,
        prompt_key="short",
        num_requests=64,
        description="Script mode bulk evaluation. 16 concurrent.",
    ),
    "stress": Scenario(
        name="stress",
        concurrency=32,
        max_tokens=50,
        prompt_key="short",
        num_requests=128,
        description="Maximum throughput stress test.",
    ),
}


# ---------------------------------------------------------------------------
# Benchmark core: async HTTP requests with streaming for TTFT measurement
# ---------------------------------------------------------------------------

async def make_streaming_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str,
) -> dict:
    """Make a single streaming request and measure TTFT + total latency."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t_start = time.perf_counter()
    ttft = None
    tokens_generated = 0
    completion = ""

    try:
        async with session.post(
            f"{url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return {
                    "error": f"HTTP {resp.status}: {error_text[:200]}",
                    "ttft_ms": -1,
                    "total_ms": -1,
                    "tokens": 0,
                }

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content and ttft is None:
                        ttft = (time.perf_counter() - t_start) * 1000  # ms
                    if content:
                        tokens_generated += 1
                        completion += content
                except json.JSONDecodeError:
                    continue

    except asyncio.TimeoutError:
        return {"error": "timeout", "ttft_ms": -1, "total_ms": -1, "tokens": 0}
    except Exception as e:
        return {"error": str(e), "ttft_ms": -1, "total_ms": -1, "tokens": 0}

    total_ms = (time.perf_counter() - t_start) * 1000

    return {
        "ttft_ms": ttft if ttft is not None else total_ms,
        "total_ms": total_ms,
        "tokens": tokens_generated,
        "completion": completion.strip(),
        "error": None,
    }


async def run_concurrent_batch(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    num_requests: int,
    api_key: str,
) -> list:
    """Run num_requests with given concurrency, return list of result dicts."""
    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session):
        async with semaphore:
            return await make_streaming_request(
                session, url, model, prompt, max_tokens, api_key
            )

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup: 2 requests to prime the KV cache
        console.print("  [dim]Warming up (2 requests)...[/dim]")
        for _ in range(2):
            await make_streaming_request(
                session, url, model, prompt, max_tokens, api_key
            )

        console.print(f"  [dim]Running {num_requests} requests, concurrency={concurrency}...[/dim]")
        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

    return results


# ---------------------------------------------------------------------------
# GPU memory measurement
# ---------------------------------------------------------------------------

def get_gpu_vram_mb() -> float:
    """Read current GPU memory usage via nvidia-smi."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            text=True,
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Run a single scenario
# ---------------------------------------------------------------------------

async def run_scenario(
    backend: str,
    url: str,
    model: str,
    scenario: Scenario,
    api_key: str,
) -> BenchmarkResult:
    """Run a benchmark scenario and return results."""
    prompt = PROMPTS[scenario.prompt_key]

    console.print(f"\n  [bold cyan]{scenario.name}[/bold cyan]: {scenario.description}")

    raw_results = await run_concurrent_batch(
        url=url,
        model=model,
        prompt=prompt,
        max_tokens=scenario.max_tokens,
        concurrency=scenario.concurrency,
        num_requests=scenario.num_requests,
        api_key=api_key,
    )

    # Filter successful results
    good = [r for r in raw_results if r.get("error") is None]
    errors = [r for r in raw_results if r.get("error") is not None]

    if errors:
        console.print(f"  [red]Errors: {len(errors)}/{len(raw_results)}[/red]")
        for e in errors[:3]:
            console.print(f"    [dim red]{e['error'][:100]}[/dim red]")

    if not good:
        console.print("  [red]All requests failed![/red]")
        return BenchmarkResult(
            backend=backend, model=model, scenario=scenario.name,
            concurrency=scenario.concurrency, num_requests=scenario.num_requests,
            max_tokens=scenario.max_tokens, prompt_tokens=0,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    ttfts = [r["ttft_ms"] for r in good]
    totals = [r["total_ms"] for r in good]
    tokens = sum(r["tokens"] for r in good)
    total_wall = max(totals) if scenario.concurrency > 1 else sum(totals)

    vram = get_gpu_vram_mb()

    ttft_p50 = float(np.percentile(ttfts, 50))
    ttft_p95 = float(np.percentile(ttfts, 95))
    ttft_p99 = float(np.percentile(ttfts, 99))

    result = BenchmarkResult(
        backend=backend,
        model=model,
        scenario=scenario.name,
        concurrency=scenario.concurrency,
        num_requests=scenario.num_requests,
        max_tokens=scenario.max_tokens,
        prompt_tokens=len(prompt.split()) * 2,  # rough estimate

        ttft_p50_ms=ttft_p50,
        ttft_p95_ms=ttft_p95,
        ttft_p99_ms=ttft_p99,
        ttft_min_ms=min(ttfts),
        ttft_max_ms=max(ttfts),
        ttft_mean_ms=statistics.mean(ttfts),

        total_latency_p50_ms=float(np.percentile(totals, 50)),
        total_latency_p95_ms=float(np.percentile(totals, 95)),
        total_latency_p99_ms=float(np.percentile(totals, 99)),
        total_latency_mean_ms=statistics.mean(totals),

        requests_per_second=len(good) / (total_wall / 1000) if total_wall > 0 else 0,
        tokens_per_second=tokens / (total_wall / 1000) if total_wall > 0 else 0,

        vram_mb=vram,

        atari_viable="YES" if ttft_p50 < 16 else "NO" if ttft_p50 > 30 else "BORDERLINE",
        vizdoom_viable="YES" if ttft_p50 < 28 else "NO" if ttft_p50 > 50 else "BORDERLINE",
        grid_world_viable="YES" if statistics.mean(totals) < 500 else "NO",

        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Print inline summary
    console.print(
        f"  TTFT p50={ttft_p50:.1f}ms  p99={ttft_p99:.1f}ms  "
        f"Total p50={result.total_latency_p50_ms:.1f}ms  "
        f"RPS={result.requests_per_second:.1f}  "
        f"VRAM={vram:.0f}MB  "
        f"Errors={len(errors)}/{len(raw_results)}"
    )

    return result


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_summary(results: list[BenchmarkResult]):
    """Print a rich table summarizing all results."""
    table = Table(title="LLM Inference Backend Benchmark Results")
    table.add_column("Scenario", style="cyan")
    table.add_column("Backend", style="green")
    table.add_column("Max Tok", justify="right")
    table.add_column("Conc.", justify="right")
    table.add_column("TTFT p50", justify="right", style="bold")
    table.add_column("TTFT p99", justify="right")
    table.add_column("Total p50", justify="right")
    table.add_column("RPS", justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("Atari?", justify="center")
    table.add_column("VizDoom?", justify="center")

    for r in results:
        atari_style = "green" if r.atari_viable == "YES" else "red" if r.atari_viable == "NO" else "yellow"
        viz_style = "green" if r.vizdoom_viable == "YES" else "red" if r.vizdoom_viable == "NO" else "yellow"

        table.add_row(
            r.scenario,
            r.backend,
            str(r.max_tokens),
            str(r.concurrency),
            f"{r.ttft_p50_ms:.1f}ms",
            f"{r.ttft_p99_ms:.1f}ms",
            f"{r.total_latency_p50_ms:.1f}ms",
            f"{r.requests_per_second:.1f}",
            f"{r.vram_mb:.0f}MB",
            f"[{atari_style}]{r.atari_viable}[/{atari_style}]",
            f"[{viz_style}]{r.vizdoom_viable}[/{viz_style}]",
        )

    console.print()
    console.print(table)
    console.print()

    # Feasibility summary
    console.print("[bold]Feasibility Thresholds:[/bold]")
    console.print("  Atari (60 Hz):    TTFT < 16ms")
    console.print("  ViZDoom (35 Hz):  TTFT < 28ms")
    console.print("  Grid World:       Total < 500ms")


def save_csv(results: list[BenchmarkResult], output_dir: Path):
    """Save all results to a single CSV."""
    import csv
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "llm_benchmark_results.csv"

    fields = [
        "backend", "model", "scenario", "concurrency", "num_requests", "max_tokens",
        "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms", "ttft_min_ms", "ttft_max_ms", "ttft_mean_ms",
        "total_latency_p50_ms", "total_latency_p95_ms", "total_latency_p99_ms", "total_latency_mean_ms",
        "requests_per_second", "tokens_per_second", "vram_mb",
        "atari_viable", "vizdoom_viable", "grid_world_viable", "timestamp",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            d = asdict(r)
            writer.writerow({k: d[k] for k in fields})

    console.print(f"\n[green]CSV saved: {csv_path}[/green]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    # Determine API key
    api_key = "EMPTY"
    if args.backend == "ollama":
        api_key = "ollama"
    elif args.backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")

    # Determine scenarios
    if args.scenarios == "all":
        scenario_names = list(SCENARIOS.keys())
    elif args.scenarios == "realtime":
        scenario_names = ["realtime_1tok", "realtime_medium", "realtime_long"]
    elif args.scenarios == "mosaic":
        scenario_names = ["single_agent", "soccer_2v2", "batch_eval"]
    else:
        scenario_names = args.scenarios.split(",")

    console.print(f"\n[bold]LLM Inference Backend Benchmark[/bold]")
    console.print(f"  Backend: [cyan]{args.backend}[/cyan]")
    console.print(f"  URL: {args.url}")
    console.print(f"  Model: {args.model}")
    console.print(f"  Scenarios: {', '.join(scenario_names)}")
    console.print(f"  GPU VRAM: {get_gpu_vram_mb():.0f} MB currently used")

    results = []
    for name in scenario_names:
        scenario = SCENARIOS.get(name)
        if scenario is None:
            console.print(f"  [red]Unknown scenario: {name}[/red]")
            continue
        result = await run_scenario(args.backend, args.url, args.model, scenario, api_key)
        results.append(result)
        result.save(Path(__file__).parent / "results")

    print_summary(results)
    save_csv(results, Path(__file__).parent / "results")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LLM inference backends for MOSAIC RL action selection",
    )
    parser.add_argument(
        "--backend", required=True,
        choices=["vllm", "sglang", "ollama", "openai"],
        help="Which backend to benchmark",
    )
    parser.add_argument(
        "--url", required=True,
        help="Base URL of the OpenAI-compatible API (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name as the server knows it",
    )
    parser.add_argument(
        "--scenarios", default="realtime",
        help="Comma-separated scenario names, or 'all', 'realtime', 'mosaic'",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
