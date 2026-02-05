#!/usr/bin/env python3
"""
ROI Calculator for vLLM vs TGI Benchmarks

Calculates CTO-relevant metrics:
- Cost per 1M tokens
- Hardware efficiency ratio
- Break-even analysis
- Annual cost savings projections

Usage:
    python roi_calculator.py --input sweep_results.csv
    python roi_calculator.py --input results.csv --gpu-cost 1.50 --annual-tokens 1000000000
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()


@dataclass
class CostAnalysis:
    """Cost analysis results for a single engine."""
    engine: str
    concurrency: int
    throughput_tokens_per_sec: float
    cost_per_million_tokens: float
    cost_per_hour_at_capacity: float
    tokens_per_dollar: float
    requests_per_dollar: float


@dataclass
class ROIComparison:
    """ROI comparison between vLLM and TGI."""
    concurrency: int
    
    # vLLM metrics
    vllm_throughput: float
    vllm_cost_per_million: float
    
    # TGI metrics  
    tgi_throughput: float
    tgi_cost_per_million: float
    
    # Comparison
    throughput_advantage: float  # vLLM / TGI
    cost_savings_percent: float  # % cheaper with vLLM
    
    # Business projections
    annual_savings_usd: float  # Based on token volume


def calculate_cost_per_million(
    throughput_tokens_per_sec: float,
    gpu_cost_per_hour: float,
) -> float:
    """Calculate cost per 1 million tokens."""
    if throughput_tokens_per_sec <= 0:
        return float('inf')
    
    tokens_per_hour = throughput_tokens_per_sec * 3600
    cost_per_token = gpu_cost_per_hour / tokens_per_hour
    return cost_per_token * 1_000_000


def analyze_results(
    df: pd.DataFrame,
    gpu_cost_per_hour: float,
) -> dict[int, tuple[CostAnalysis, CostAnalysis]]:
    """Analyze results and return cost analysis for each concurrency level."""
    results = {}
    
    for concurrency in df['concurrency'].unique():
        level_df = df[df['concurrency'] == concurrency]
        
        vllm_row = level_df[level_df['engine'] == 'vllm']
        tgi_row = level_df[level_df['engine'] == 'tgi']
        
        if vllm_row.empty or tgi_row.empty:
            continue
        
        vllm_throughput = vllm_row['throughput_tok_s'].values[0]
        tgi_throughput = tgi_row['throughput_tok_s'].values[0]
        
        vllm_analysis = CostAnalysis(
            engine='vllm',
            concurrency=concurrency,
            throughput_tokens_per_sec=vllm_throughput,
            cost_per_million_tokens=calculate_cost_per_million(vllm_throughput, gpu_cost_per_hour),
            cost_per_hour_at_capacity=gpu_cost_per_hour,
            tokens_per_dollar=(vllm_throughput * 3600) / gpu_cost_per_hour if gpu_cost_per_hour > 0 else 0,
            requests_per_dollar=(vllm_row['throughput_req_s'].values[0] * 3600) / gpu_cost_per_hour if gpu_cost_per_hour > 0 else 0,
        )
        
        tgi_analysis = CostAnalysis(
            engine='tgi',
            concurrency=concurrency,
            throughput_tokens_per_sec=tgi_throughput,
            cost_per_million_tokens=calculate_cost_per_million(tgi_throughput, gpu_cost_per_hour),
            cost_per_hour_at_capacity=gpu_cost_per_hour,
            tokens_per_dollar=(tgi_throughput * 3600) / gpu_cost_per_hour if gpu_cost_per_hour > 0 else 0,
            requests_per_dollar=(tgi_row['throughput_req_s'].values[0] * 3600) / gpu_cost_per_hour if gpu_cost_per_hour > 0 else 0,
        )
        
        results[concurrency] = (vllm_analysis, tgi_analysis)
    
    return results


def calculate_roi_comparison(
    vllm: CostAnalysis,
    tgi: CostAnalysis,
    annual_token_volume: int,
) -> ROIComparison:
    """Calculate ROI comparison between vLLM and TGI."""
    throughput_advantage = vllm.throughput_tokens_per_sec / tgi.throughput_tokens_per_sec if tgi.throughput_tokens_per_sec > 0 else 0
    
    cost_savings_percent = (
        (tgi.cost_per_million_tokens - vllm.cost_per_million_tokens) / tgi.cost_per_million_tokens * 100
        if tgi.cost_per_million_tokens > 0 else 0
    )
    
    # Annual savings calculation
    annual_cost_vllm = (annual_token_volume / 1_000_000) * vllm.cost_per_million_tokens
    annual_cost_tgi = (annual_token_volume / 1_000_000) * tgi.cost_per_million_tokens
    annual_savings = annual_cost_tgi - annual_cost_vllm
    
    return ROIComparison(
        concurrency=vllm.concurrency,
        vllm_throughput=vllm.throughput_tokens_per_sec,
        vllm_cost_per_million=vllm.cost_per_million_tokens,
        tgi_throughput=tgi.throughput_tokens_per_sec,
        tgi_cost_per_million=tgi.cost_per_million_tokens,
        throughput_advantage=throughput_advantage,
        cost_savings_percent=cost_savings_percent,
        annual_savings_usd=annual_savings,
    )


def display_cost_table(analyses: dict[int, tuple[CostAnalysis, CostAnalysis]], gpu_name: str):
    """Display cost analysis table."""
    table = Table(title=f"Cost Analysis ({gpu_name} @ ${analyses[list(analyses.keys())[0]][0].cost_per_hour_at_capacity:.2f}/hr)")
    
    table.add_column("Concurrency", style="cyan", justify="center")
    table.add_column("Engine", style="bold")
    table.add_column("Throughput\n(tok/s)", justify="right")
    table.add_column("$/1M tokens", justify="right")
    table.add_column("Tokens/$", justify="right")
    
    for concurrency in sorted(analyses.keys()):
        vllm, tgi = analyses[concurrency]
        
        # vLLM row
        table.add_row(
            str(concurrency),
            "[green]vLLM[/green]",
            f"{vllm.throughput_tokens_per_sec:,.1f}",
            f"[green]${vllm.cost_per_million_tokens:.4f}[/green]",
            f"{vllm.tokens_per_dollar:,.0f}",
        )
        
        # TGI row
        table.add_row(
            "",
            "[red]TGI[/red]",
            f"{tgi.throughput_tokens_per_sec:,.1f}",
            f"[red]${tgi.cost_per_million_tokens:.4f}[/red]",
            f"{tgi.tokens_per_dollar:,.0f}",
        )
        
        table.add_row("", "", "", "", "")
    
    console.print(table)


def display_roi_table(comparisons: list[ROIComparison]):
    """Display ROI comparison table."""
    table = Table(title="ROI Analysis: vLLM vs TGI")
    
    table.add_column("Concurrency", style="cyan", justify="center")
    table.add_column("Throughput\nAdvantage", justify="center")
    table.add_column("Cost Savings\n(%)", justify="center")
    table.add_column("Annual Savings\n(USD)", justify="right")
    
    for comp in comparisons:
        advantage_style = "green" if comp.throughput_advantage > 1 else "red"
        savings_style = "green" if comp.cost_savings_percent > 0 else "red"
        
        table.add_row(
            str(comp.concurrency),
            f"[{advantage_style}]{comp.throughput_advantage:.2f}x[/{advantage_style}]",
            f"[{savings_style}]{comp.cost_savings_percent:.1f}%[/{savings_style}]",
            f"[green]${comp.annual_savings_usd:,.2f}[/green]" if comp.annual_savings_usd > 0 else f"[red]-${abs(comp.annual_savings_usd):,.2f}[/red]",
        )
    
    console.print(table)


def display_executive_summary(comparisons: list[ROIComparison], gpu_name: str, annual_tokens: int):
    """Display executive summary panel."""
    # Find best performing concurrency level
    best = max(comparisons, key=lambda x: x.cost_savings_percent)
    avg_advantage = sum(c.throughput_advantage for c in comparisons) / len(comparisons)
    avg_savings = sum(c.cost_savings_percent for c in comparisons) / len(comparisons)
    total_annual_savings = sum(c.annual_savings_usd for c in comparisons) / len(comparisons)
    
    summary = f"""
[bold cyan]Hardware:[/bold cyan] {gpu_name}
[bold cyan]Token Volume:[/bold cyan] {annual_tokens:,} tokens/year

[bold green]KEY FINDINGS[/bold green]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[bold]Average Throughput Advantage:[/bold] [green]{avg_advantage:.2f}x[/green]
  └─ vLLM processes {avg_advantage:.1f}x more tokens per second

[bold]Average Cost Savings:[/bold] [green]{avg_savings:.1f}%[/green]
  └─ vLLM is {avg_savings:.1f}% cheaper per token

[bold]Best Performance (Concurrency {best.concurrency}):[/bold]
  └─ Throughput: [green]{best.throughput_advantage:.2f}x[/green] faster
  └─ Cost Savings: [green]{best.cost_savings_percent:.1f}%[/green]

[bold]Projected Annual Savings:[/bold] [green bold]${total_annual_savings:,.2f}[/green bold]

[bold green]RECOMMENDATION[/bold green]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{"✅ vLLM demonstrates significant ROI improvements over TGI." if avg_savings > 0 else "⚠️ Results vary. Consider workload-specific testing."}
{"Optimal concurrency level: " + str(best.concurrency) + " concurrent requests."}
{"At scale ({:,} tokens/year), switching to vLLM saves ${:,.2f}/year.".format(annual_tokens, total_annual_savings)}
"""
    
    console.print(Panel(summary, title="[bold]Executive Summary[/bold]", border_style="green"))


def generate_report(
    df: pd.DataFrame,
    gpu_cost_per_hour: float,
    gpu_name: str,
    annual_token_volume: int,
    output_file: Optional[str] = None,
):
    """Generate comprehensive ROI report."""
    console.print("\n" + "="*70)
    console.print("[bold cyan]vLLM vs TGI: ROI Analysis Report[/bold cyan]")
    console.print("="*70 + "\n")
    
    # Analyze results
    analyses = analyze_results(df, gpu_cost_per_hour)
    
    if not analyses:
        console.print("[red]No valid results found for analysis.[/red]")
        return
    
    # Calculate ROI comparisons
    comparisons = []
    for concurrency, (vllm, tgi) in sorted(analyses.items()):
        comp = calculate_roi_comparison(vllm, tgi, annual_token_volume)
        comparisons.append(comp)
    
    # Display tables
    display_cost_table(analyses, gpu_name)
    console.print()
    display_roi_table(comparisons)
    console.print()
    display_executive_summary(comparisons, gpu_name, annual_token_volume)
    
    # Export to CSV if requested
    if output_file:
        records = []
        for comp in comparisons:
            records.append({
                'concurrency': comp.concurrency,
                'vllm_throughput_tok_s': comp.vllm_throughput,
                'tgi_throughput_tok_s': comp.tgi_throughput,
                'vllm_cost_per_million': comp.vllm_cost_per_million,
                'tgi_cost_per_million': comp.tgi_cost_per_million,
                'throughput_advantage_x': comp.throughput_advantage,
                'cost_savings_percent': comp.cost_savings_percent,
                'annual_savings_usd': comp.annual_savings_usd,
            })
        
        roi_df = pd.DataFrame(records)
        roi_df.to_csv(output_file, index=False)
        console.print(f"\n[green]ROI report saved to {output_file}[/green]")


@click.command()
@click.option("--input", "-i", "input_file", required=True, help="Input CSV file from benchmark")
@click.option("--gpu-cost", "-g", default=float(os.getenv("GPU_COST_PER_HOUR", "1.50")), help="GPU cost per hour in USD")
@click.option("--gpu-name", "-n", default=os.getenv("GPU_NAME", "L40S"), help="GPU model name")
@click.option("--annual-tokens", "-t", default=1_000_000_000, help="Projected annual token volume")
@click.option("--output", "-o", default=None, help="Output CSV file for ROI report")
def main(input_file: str, gpu_cost: float, gpu_name: str, annual_tokens: int, output: Optional[str]):
    """Calculate ROI metrics from benchmark results."""
    # Load results
    input_path = Path(input_file)
    if not input_path.exists():
        # Try relative to script directory
        input_path = Path(__file__).parent.parent / input_file
    
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        return
    
    df = pd.read_csv(input_path)
    
    # Generate report
    generate_report(
        df=df,
        gpu_cost_per_hour=gpu_cost,
        gpu_name=gpu_name,
        annual_token_volume=annual_tokens,
        output_file=output,
    )


if __name__ == "__main__":
    main()

