#!/usr/bin/env python3
"""
vLLM vs TGI Benchmark Load Generator

Measures key inference metrics:
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- E2E Latency
- Throughput (tokens/sec)

Usage:
    python load_generator.py --help
    python load_generator.py run --concurrency 10 --duration 60
    python load_generator.py sweep --concurrency-levels 1,10,50,100
"""

import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import click
import pandas as pd
from dotenv import load_dotenv
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Load environment
load_dotenv()

console = Console()

# ============================================================================
# Configuration
# ============================================================================

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
TGI_URL = os.getenv("TGI_URL", "http://localhost:8080")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://localhost:9091")
PROMPTS_FILE = os.getenv("PROMPTS_FILE", "datasets/prompts.jsonl")
MODEL_ID = os.getenv("MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.3")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")

def get_vllm_headers():
    """Get headers for vLLM requests, including API key if configured."""
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    return headers


@dataclass
class RequestResult:
    """Result of a single inference request."""
    engine: str
    prompt_tokens: int
    output_tokens: int
    ttft_ms: float  # Time to first token in milliseconds
    total_time_ms: float  # Total end-to-end time
    tpot_ms: float  # Time per output token (excluding first)
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass 
class BenchmarkResult:
    """Aggregated results for a benchmark run."""
    engine: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens_generated: int
    duration_seconds: float
    
    # Throughput
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float
    
    # TTFT stats (ms)
    ttft_mean: float
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    ttft_min: float
    ttft_max: float
    
    # TPOT stats (ms)
    tpot_mean: float
    tpot_p50: float
    tpot_p90: float
    tpot_p99: float
    
    # E2E latency stats (ms)
    latency_mean: float
    latency_p50: float
    latency_p90: float
    latency_p99: float


# ============================================================================
# Prompt Loading
# ============================================================================

def load_prompts(file_path: str) -> list[dict]:
    """Load prompts from JSONL file."""
    prompts = []
    path = Path(__file__).parent / file_path
    
    if not path.exists():
        console.print(f"[yellow]Warning: {file_path} not found, using default prompts[/yellow]")
        return [
            {"prompt": "Explain the concept of machine learning in simple terms.", "max_tokens": 256},
            {"prompt": "Write a Python function to calculate fibonacci numbers.", "max_tokens": 512},
            {"prompt": "What are the key differences between TCP and UDP?", "max_tokens": 384},
            {"prompt": "Describe the architecture of a modern web application.", "max_tokens": 512},
            {"prompt": "How does photosynthesis work?", "max_tokens": 256},
        ]
    
    with open(path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    
    return prompts


# ============================================================================
# HTTP Clients
# ============================================================================

async def request_vllm(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int = 256,
) -> RequestResult:
    """Send streaming request to vLLM and measure TTFT/TPOT."""
    url = f"{VLLM_URL}/v1/completions"
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    headers = get_vllm_headers()
    
    start_time = time.perf_counter()
    first_token_time = None
    output_tokens = 0
    
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return RequestResult(
                    engine="vllm",
                    prompt_tokens=len(prompt.split()),  # Approximate
                    output_tokens=0,
                    ttft_ms=0,
                    total_time_ms=0,
                    tpot_ms=0,
                    success=False,
                    error=f"HTTP {resp.status}: {error_text[:100]}",
                )
            
            async for line in resp.content:
                line = line.decode().strip()
                if not line or not line.startswith("data:"):
                    continue
                
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data)
                    if chunk.get("choices", [{}])[0].get("text"):
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        output_tokens += 1
                except json.JSONDecodeError:
                    continue
            
            end_time = time.perf_counter()
            
            if first_token_time is None:
                first_token_time = end_time
            
            total_time_ms = (end_time - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000
            
            # TPOT: time for tokens after the first
            if output_tokens > 1:
                tpot_ms = (end_time - first_token_time) * 1000 / (output_tokens - 1)
            else:
                tpot_ms = 0
            
            return RequestResult(
                engine="vllm",
                prompt_tokens=len(prompt.split()),
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                total_time_ms=total_time_ms,
                tpot_ms=tpot_ms,
                success=True,
            )
    
    except Exception as e:
        return RequestResult(
            engine="vllm",
            prompt_tokens=len(prompt.split()),
            output_tokens=0,
            ttft_ms=0,
            total_time_ms=0,
            tpot_ms=0,
            success=False,
            error=str(e),
        )


async def request_tgi(
    session: aiohttp.ClientSession,
    prompt: str,
    max_tokens: int = 256,
) -> RequestResult:
    """Send streaming request to TGI and measure TTFT/TPOT."""
    url = f"{TGI_URL}/generate_stream"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "do_sample": True,
        },
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    output_tokens = 0
    
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return RequestResult(
                    engine="tgi",
                    prompt_tokens=len(prompt.split()),
                    output_tokens=0,
                    ttft_ms=0,
                    total_time_ms=0,
                    tpot_ms=0,
                    success=False,
                    error=f"HTTP {resp.status}: {error_text[:100]}",
                )
            
            async for line in resp.content:
                line = line.decode().strip()
                if not line or not line.startswith("data:"):
                    continue
                
                data = line[5:].strip()
                
                try:
                    chunk = json.loads(data)
                    if chunk.get("token", {}).get("text"):
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        output_tokens += 1
                except json.JSONDecodeError:
                    continue
            
            end_time = time.perf_counter()
            
            if first_token_time is None:
                first_token_time = end_time
            
            total_time_ms = (end_time - start_time) * 1000
            ttft_ms = (first_token_time - start_time) * 1000
            
            if output_tokens > 1:
                tpot_ms = (end_time - first_token_time) * 1000 / (output_tokens - 1)
            else:
                tpot_ms = 0
            
            return RequestResult(
                engine="tgi",
                prompt_tokens=len(prompt.split()),
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                total_time_ms=total_time_ms,
                tpot_ms=tpot_ms,
                success=True,
            )
    
    except Exception as e:
        return RequestResult(
            engine="tgi",
            prompt_tokens=len(prompt.split()),
            output_tokens=0,
            ttft_ms=0,
            total_time_ms=0,
            tpot_ms=0,
            success=False,
            error=str(e),
        )


# ============================================================================
# Benchmark Runner
# ============================================================================

async def run_benchmark(
    engine: str,
    concurrency: int,
    duration_seconds: int,
    prompts: list[dict],
) -> BenchmarkResult:
    """Run a benchmark for a single engine at a specific concurrency level."""
    
    results: list[RequestResult] = []
    semaphore = asyncio.Semaphore(concurrency)
    stop_event = asyncio.Event()
    prompt_index = 0
    
    async def worker():
        nonlocal prompt_index
        
        connector = aiohttp.TCPConnector(limit=concurrency * 2)
        timeout = aiohttp.ClientTimeout(total=300)
        headers = get_vllm_headers() if engine == "vllm" else {}
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
            while not stop_event.is_set():
                async with semaphore:
                    if stop_event.is_set():
                        break
                    
                    # Round-robin through prompts
                    prompt_data = prompts[prompt_index % len(prompts)]
                    prompt_index += 1
                    
                    prompt = prompt_data.get("prompt", prompt_data.get("text", "Hello"))
                    max_tokens = prompt_data.get("max_tokens", 256)
                    
                    if engine == "vllm":
                        result = await request_vllm(session, prompt, max_tokens)
                    else:
                        result = await request_tgi(session, prompt, max_tokens)
                    
                    results.append(result)
    
    # Start workers
    start_time = time.perf_counter()
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    
    # Run for specified duration
    await asyncio.sleep(duration_seconds)
    stop_event.set()
    
    # Wait for workers to finish
    await asyncio.gather(*workers, return_exceptions=True)
    end_time = time.perf_counter()
    
    # Calculate statistics
    actual_duration = end_time - start_time
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    if not successful:
        console.print(f"[red]No successful requests for {engine}![/red]")
        return BenchmarkResult(
            engine=engine,
            concurrency=concurrency,
            total_requests=len(results),
            successful_requests=0,
            failed_requests=len(failed),
            total_tokens_generated=0,
            duration_seconds=actual_duration,
            throughput_tokens_per_sec=0,
            throughput_requests_per_sec=0,
            ttft_mean=0, ttft_p50=0, ttft_p90=0, ttft_p99=0, ttft_min=0, ttft_max=0,
            tpot_mean=0, tpot_p50=0, tpot_p90=0, tpot_p99=0,
            latency_mean=0, latency_p50=0, latency_p90=0, latency_p99=0,
        )
    
    # Extract metrics
    ttfts = [r.ttft_ms for r in successful]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms > 0]
    latencies = [r.total_time_ms for r in successful]
    total_tokens = sum(r.output_tokens for r in successful)
    
    def percentile(data: list, p: float) -> float:
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)
    
    return BenchmarkResult(
        engine=engine,
        concurrency=concurrency,
        total_requests=len(results),
        successful_requests=len(successful),
        failed_requests=len(failed),
        total_tokens_generated=total_tokens,
        duration_seconds=actual_duration,
        throughput_tokens_per_sec=total_tokens / actual_duration,
        throughput_requests_per_sec=len(successful) / actual_duration,
        ttft_mean=statistics.mean(ttfts),
        ttft_p50=percentile(ttfts, 0.50),
        ttft_p90=percentile(ttfts, 0.90),
        ttft_p99=percentile(ttfts, 0.99),
        ttft_min=min(ttfts),
        ttft_max=max(ttfts),
        tpot_mean=statistics.mean(tpots) if tpots else 0,
        tpot_p50=percentile(tpots, 0.50),
        tpot_p90=percentile(tpots, 0.90),
        tpot_p99=percentile(tpots, 0.99),
        latency_mean=statistics.mean(latencies),
        latency_p50=percentile(latencies, 0.50),
        latency_p90=percentile(latencies, 0.90),
        latency_p99=percentile(latencies, 0.99),
    )


# ============================================================================
# Output & Reporting
# ============================================================================

def display_results(vllm_result: BenchmarkResult, tgi_result: BenchmarkResult):
    """Display side-by-side comparison in a rich table."""
    table = Table(title=f"Benchmark Results (Concurrency: {vllm_result.concurrency})")
    
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("vLLM", style="green", justify="right")
    table.add_column("TGI", style="red", justify="right")
    table.add_column("Advantage", style="yellow", justify="right")
    
    def fmt_advantage(vllm_val: float, tgi_val: float, lower_is_better: bool = False) -> str:
        if tgi_val == 0:
            return "N/A"
        if lower_is_better:
            ratio = tgi_val / vllm_val if vllm_val > 0 else 0
        else:
            ratio = vllm_val / tgi_val
        
        if ratio > 1:
            return f"[green]{ratio:.2f}x[/green]"
        elif ratio < 1:
            return f"[red]{1/ratio:.2f}x worse[/red]"
        return "1.00x"
    
    # Request stats
    table.add_row(
        "Successful Requests",
        str(vllm_result.successful_requests),
        str(tgi_result.successful_requests),
        "",
    )
    table.add_row(
        "Failed Requests",
        str(vllm_result.failed_requests),
        str(tgi_result.failed_requests),
        "",
    )
    table.add_row("", "", "", "")  # Separator
    
    # Throughput
    table.add_row(
        "Throughput (tok/s)",
        f"{vllm_result.throughput_tokens_per_sec:.1f}",
        f"{tgi_result.throughput_tokens_per_sec:.1f}",
        fmt_advantage(vllm_result.throughput_tokens_per_sec, tgi_result.throughput_tokens_per_sec),
    )
    table.add_row(
        "Throughput (req/s)",
        f"{vllm_result.throughput_requests_per_sec:.2f}",
        f"{tgi_result.throughput_requests_per_sec:.2f}",
        fmt_advantage(vllm_result.throughput_requests_per_sec, tgi_result.throughput_requests_per_sec),
    )
    table.add_row("", "", "", "")
    
    # TTFT
    table.add_row(
        "TTFT P50 (ms)",
        f"{vllm_result.ttft_p50:.1f}",
        f"{tgi_result.ttft_p50:.1f}",
        fmt_advantage(vllm_result.ttft_p50, tgi_result.ttft_p50, lower_is_better=True),
    )
    table.add_row(
        "TTFT P99 (ms)",
        f"{vllm_result.ttft_p99:.1f}",
        f"{tgi_result.ttft_p99:.1f}",
        fmt_advantage(vllm_result.ttft_p99, tgi_result.ttft_p99, lower_is_better=True),
    )
    table.add_row("", "", "", "")
    
    # TPOT
    table.add_row(
        "TPOT P50 (ms/tok)",
        f"{vllm_result.tpot_p50:.2f}",
        f"{tgi_result.tpot_p50:.2f}",
        fmt_advantage(vllm_result.tpot_p50, tgi_result.tpot_p50, lower_is_better=True),
    )
    table.add_row("", "", "", "")
    
    # E2E Latency
    table.add_row(
        "E2E Latency P50 (ms)",
        f"{vllm_result.latency_p50:.1f}",
        f"{tgi_result.latency_p50:.1f}",
        fmt_advantage(vllm_result.latency_p50, tgi_result.latency_p50, lower_is_better=True),
    )
    table.add_row(
        "E2E Latency P99 (ms)",
        f"{vllm_result.latency_p99:.1f}",
        f"{tgi_result.latency_p99:.1f}",
        fmt_advantage(vllm_result.latency_p99, tgi_result.latency_p99, lower_is_better=True),
    )
    
    console.print(table)


def push_results_to_prometheus(result: BenchmarkResult):
    """Push benchmark results to Prometheus Pushgateway."""
    registry = CollectorRegistry()
    
    labels = {"engine": result.engine, "concurrency": str(result.concurrency)}
    
    # Define gauges
    throughput = Gauge(
        "benchmark_throughput_tokens_per_sec",
        "Tokens generated per second",
        labelnames=list(labels.keys()),
        registry=registry,
    )
    throughput.labels(**labels).set(result.throughput_tokens_per_sec)
    
    ttft_p50 = Gauge(
        "benchmark_ttft_p50_ms",
        "Time to first token P50 in milliseconds",
        labelnames=list(labels.keys()),
        registry=registry,
    )
    ttft_p50.labels(**labels).set(result.ttft_p50)
    
    ttft_p99 = Gauge(
        "benchmark_ttft_p99_ms",
        "Time to first token P99 in milliseconds",
        labelnames=list(labels.keys()),
        registry=registry,
    )
    ttft_p99.labels(**labels).set(result.ttft_p99)
    
    tpot_p50 = Gauge(
        "benchmark_tpot_p50_ms",
        "Time per output token P50 in milliseconds",
        labelnames=list(labels.keys()),
        registry=registry,
    )
    tpot_p50.labels(**labels).set(result.tpot_p50)
    
    latency_p50 = Gauge(
        "benchmark_latency_p50_ms",
        "End-to-end latency P50 in milliseconds",
        labelnames=list(labels.keys()),
        registry=registry,
    )
    latency_p50.labels(**labels).set(result.latency_p50)
    
    try:
        push_to_gateway(PUSHGATEWAY_URL, job="inference_benchmark", registry=registry)
        console.print(f"[dim]Pushed metrics to Pushgateway for {result.engine}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Warning: Could not push to Pushgateway: {e}[/yellow]")


def save_results_to_csv(results: list[BenchmarkResult], output_path: str):
    """Save results to CSV file."""
    records = []
    for r in results:
        records.append({
            "engine": r.engine,
            "concurrency": r.concurrency,
            "total_requests": r.total_requests,
            "successful_requests": r.successful_requests,
            "failed_requests": r.failed_requests,
            "total_tokens": r.total_tokens_generated,
            "duration_s": r.duration_seconds,
            "throughput_tok_s": r.throughput_tokens_per_sec,
            "throughput_req_s": r.throughput_requests_per_sec,
            "ttft_mean_ms": r.ttft_mean,
            "ttft_p50_ms": r.ttft_p50,
            "ttft_p90_ms": r.ttft_p90,
            "ttft_p99_ms": r.ttft_p99,
            "tpot_mean_ms": r.tpot_mean,
            "tpot_p50_ms": r.tpot_p50,
            "tpot_p90_ms": r.tpot_p90,
            "tpot_p99_ms": r.tpot_p99,
            "latency_mean_ms": r.latency_mean,
            "latency_p50_ms": r.latency_p50,
            "latency_p90_ms": r.latency_p90,
            "latency_p99_ms": r.latency_p99,
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    console.print(f"[green]Results saved to {output_path}[/green]")


# ============================================================================
# CLI Commands
# ============================================================================

@click.group()
def cli():
    """vLLM vs TGI Benchmark Suite"""
    pass


@cli.command()
@click.option("--concurrency", "-c", default=10, help="Number of concurrent requests")
@click.option("--duration", "-d", default=60, help="Duration in seconds")
@click.option("--warmup", "-w", default=10, help="Warmup requests per engine")
@click.option("--output", "-o", default="results.csv", help="Output CSV file")
@click.option("--vllm-only", is_flag=True, help="Only benchmark vLLM")
@click.option("--tgi-only", is_flag=True, help="Only benchmark TGI")
def run(concurrency: int, duration: int, warmup: int, output: str, vllm_only: bool, tgi_only: bool):
    """Run a single benchmark at specified concurrency."""
    prompts = load_prompts(PROMPTS_FILE)
    console.print(f"[cyan]Loaded {len(prompts)} prompts[/cyan]")
    
    results = []
    
    async def benchmark():
        # Warmup
        if warmup > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Warming up...", total=None)
                
                connector = aiohttp.TCPConnector(limit=10)
                timeout = aiohttp.ClientTimeout(total=300)
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    warmup_tasks = []
                    for i in range(warmup):
                        prompt = prompts[i % len(prompts)]
                        if not tgi_only:
                            warmup_tasks.append(
                                request_vllm(session, prompt.get("prompt", ""), prompt.get("max_tokens", 64))
                            )
                        if not vllm_only:
                            warmup_tasks.append(
                                request_tgi(session, prompt.get("prompt", ""), prompt.get("max_tokens", 64))
                            )
                    await asyncio.gather(*warmup_tasks, return_exceptions=True)
                
                progress.remove_task(task)
            console.print("[green]Warmup complete[/green]")
        
        # Run benchmarks
        if not tgi_only:
            console.print(f"\n[cyan]Benchmarking vLLM at concurrency {concurrency}...[/cyan]")
            vllm_result = await run_benchmark("vllm", concurrency, duration, prompts)
            results.append(vllm_result)
            push_results_to_prometheus(vllm_result)
        
        if not vllm_only:
            console.print(f"\n[cyan]Benchmarking TGI at concurrency {concurrency}...[/cyan]")
            tgi_result = await run_benchmark("tgi", concurrency, duration, prompts)
            results.append(tgi_result)
            push_results_to_prometheus(tgi_result)
        
        # Display comparison
        if len(results) == 2:
            console.print("\n")
            display_results(results[0], results[1])
    
    asyncio.run(benchmark())
    
    # Save results
    save_results_to_csv(results, output)


@cli.command()
@click.option("--concurrency-levels", "-c", default="1,10,50,100,200", help="Comma-separated concurrency levels")
@click.option("--duration", "-d", default=60, help="Duration per level in seconds")
@click.option("--warmup", "-w", default=20, help="Warmup requests per engine")
@click.option("--output", "-o", default="sweep_results.csv", help="Output CSV file")
def sweep(concurrency_levels: str, duration: int, warmup: int, output: str):
    """Run benchmarks across multiple concurrency levels."""
    levels = [int(x.strip()) for x in concurrency_levels.split(",")]
    prompts = load_prompts(PROMPTS_FILE)
    console.print(f"[cyan]Loaded {len(prompts)} prompts[/cyan]")
    console.print(f"[cyan]Running sweep across concurrency levels: {levels}[/cyan]")
    
    all_results = []
    
    async def run_sweep():
        # Warmup
        if warmup > 0:
            console.print("[cyan]Running warmup...[/cyan]")
            connector = aiohttp.TCPConnector(limit=20)
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                warmup_tasks = []
                for i in range(warmup):
                    prompt = prompts[i % len(prompts)]
                    warmup_tasks.append(
                        request_vllm(session, prompt.get("prompt", ""), prompt.get("max_tokens", 64))
                    )
                    warmup_tasks.append(
                        request_tgi(session, prompt.get("prompt", ""), prompt.get("max_tokens", 64))
                    )
                await asyncio.gather(*warmup_tasks, return_exceptions=True)
            console.print("[green]Warmup complete[/green]")
        
        # Run sweep
        for level in levels:
            console.print(f"\n{'='*60}")
            console.print(f"[bold cyan]Concurrency Level: {level}[/bold cyan]")
            console.print(f"{'='*60}")
            
            # vLLM
            console.print(f"\n[cyan]Benchmarking vLLM...[/cyan]")
            vllm_result = await run_benchmark("vllm", level, duration, prompts)
            all_results.append(vllm_result)
            push_results_to_prometheus(vllm_result)
            
            # TGI
            console.print(f"\n[cyan]Benchmarking TGI...[/cyan]")
            tgi_result = await run_benchmark("tgi", level, duration, prompts)
            all_results.append(tgi_result)
            push_results_to_prometheus(tgi_result)
            
            # Display comparison
            console.print("\n")
            display_results(vllm_result, tgi_result)
    
    asyncio.run(run_sweep())
    
    # Save all results
    save_results_to_csv(all_results, output)
    
    # Print summary
    console.print("\n" + "="*60)
    console.print("[bold green]SWEEP COMPLETE[/bold green]")
    console.print("="*60)


@cli.command()
def health():
    """Check health of inference endpoints."""
    async def check_health():
        async with aiohttp.ClientSession() as session:
            # vLLM
            try:
                async with session.get(f"{VLLM_URL}/health", timeout=aiohttp.ClientTimeout(total=5), headers=get_vllm_headers()) as resp:
                    if resp.status == 200:
                        console.print(f"[green]✓ vLLM ({VLLM_URL}): Healthy[/green]")
                    else:
                        console.print(f"[red]✗ vLLM ({VLLM_URL}): HTTP {resp.status}[/red]")
            except Exception as e:
                console.print(f"[red]✗ vLLM ({VLLM_URL}): {e}[/red]")
            
            # TGI
            try:
                async with session.get(f"{TGI_URL}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        console.print(f"[green]✓ TGI ({TGI_URL}): Healthy[/green]")
                    else:
                        console.print(f"[red]✗ TGI ({TGI_URL}): HTTP {resp.status}[/red]")
            except Exception as e:
                console.print(f"[red]✗ TGI ({TGI_URL}): {e}[/red]")
    
    asyncio.run(check_health())


if __name__ == "__main__":
    cli()

