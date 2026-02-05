#!/usr/bin/env python3
"""
Compare vLLM vs TGI Results from Separate Benchmark Runs

For single-GPU setups where you can only run one engine at a time,
this script merges separate CSV files and generates comparison reports.

Usage:
    python compare_results.py --vllm vllm_c4.csv vllm_c8.csv --tgi tgi_c4.csv tgi_c8.csv
    python compare_results.py --vllm vllm_*.csv --tgi tgi_*.csv -o combined_results.csv
    python compare_results.py --input-dir ./results -o sweep_results.csv
"""

import click
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
import glob

console = Console()


def load_and_merge_csvs(file_patterns: list[str]) -> pd.DataFrame:
    """Load multiple CSV files and merge them."""
    all_dfs = []
    
    for pattern in file_patterns:
        # Expand glob patterns
        files = glob.glob(pattern)
        if not files:
            # Try as literal filename
            if Path(pattern).exists():
                files = [pattern]
            else:
                console.print(f"[yellow]Warning: No files found matching '{pattern}'[/yellow]")
                continue
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                df['source_file'] = Path(file_path).name
                all_dfs.append(df)
                console.print(f"[green]✓[/green] Loaded {file_path} ({len(df)} rows)")
            except Exception as e:
                console.print(f"[red]✗[/red] Failed to load {file_path}: {e}")
    
    if not all_dfs:
        return pd.DataFrame()
    
    return pd.concat(all_dfs, ignore_index=True)


def find_matching_results(vllm_df: pd.DataFrame, tgi_df: pd.DataFrame) -> list[tuple]:
    """Find matching concurrency levels between vLLM and TGI results."""
    vllm_levels = set(vllm_df['concurrency'].unique())
    tgi_levels = set(tgi_df['concurrency'].unique())
    
    common_levels = sorted(vllm_levels & tgi_levels)
    
    if vllm_levels - tgi_levels:
        console.print(f"[yellow]Note: vLLM has results for concurrency levels not in TGI: {sorted(vllm_levels - tgi_levels)}[/yellow]")
    if tgi_levels - vllm_levels:
        console.print(f"[yellow]Note: TGI has results for concurrency levels not in vLLM: {sorted(tgi_levels - vllm_levels)}[/yellow]")
    
    return common_levels


def display_comparison(vllm_df: pd.DataFrame, tgi_df: pd.DataFrame):
    """Display side-by-side comparison table."""
    common_levels = find_matching_results(vllm_df, tgi_df)
    
    if not common_levels:
        console.print("[red]No matching concurrency levels found between vLLM and TGI results![/red]")
        return
    
    def fmt_advantage(vllm_val: float, tgi_val: float, lower_is_better: bool = False) -> str:
        if tgi_val == 0 or vllm_val == 0:
            return "N/A"
        if lower_is_better:
            ratio = tgi_val / vllm_val
        else:
            ratio = vllm_val / tgi_val
        
        if ratio > 1.05:
            return f"[green]{ratio:.2f}x vLLM[/green]"
        elif ratio < 0.95:
            return f"[red]{1/ratio:.2f}x TGI[/red]"
        return "[dim]~same[/dim]"
    
    for level in common_levels:
        vllm_row = vllm_df[vllm_df['concurrency'] == level].iloc[0]
        tgi_row = tgi_df[tgi_df['concurrency'] == level].iloc[0]
        
        table = Table(title=f"Comparison @ Concurrency {level}")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("vLLM", style="green", justify="right")
        table.add_column("TGI", style="red", justify="right")
        table.add_column("Winner", style="yellow", justify="center")
        
        # Request stats
        table.add_row(
            "Successful Requests",
            str(int(vllm_row['successful_requests'])),
            str(int(tgi_row['successful_requests'])),
            ""
        )
        table.add_row(
            "Failed Requests",
            str(int(vllm_row['failed_requests'])),
            str(int(tgi_row['failed_requests'])),
            ""
        )
        table.add_row("─" * 20, "─" * 10, "─" * 10, "─" * 12)
        
        # Throughput (higher is better)
        table.add_row(
            "Throughput (tok/s)",
            f"{vllm_row['throughput_tok_s']:.1f}",
            f"{tgi_row['throughput_tok_s']:.1f}",
            fmt_advantage(vllm_row['throughput_tok_s'], tgi_row['throughput_tok_s'])
        )
        table.add_row(
            "Throughput (req/s)",
            f"{vllm_row['throughput_req_s']:.2f}",
            f"{tgi_row['throughput_req_s']:.2f}",
            fmt_advantage(vllm_row['throughput_req_s'], tgi_row['throughput_req_s'])
        )
        table.add_row("─" * 20, "─" * 10, "─" * 10, "─" * 12)
        
        # TTFT (lower is better)
        table.add_row(
            "TTFT P50 (ms)",
            f"{vllm_row['ttft_p50_ms']:.1f}",
            f"{tgi_row['ttft_p50_ms']:.1f}",
            fmt_advantage(vllm_row['ttft_p50_ms'], tgi_row['ttft_p50_ms'], lower_is_better=True)
        )
        table.add_row(
            "TTFT P99 (ms)",
            f"{vllm_row['ttft_p99_ms']:.1f}",
            f"{tgi_row['ttft_p99_ms']:.1f}",
            fmt_advantage(vllm_row['ttft_p99_ms'], tgi_row['ttft_p99_ms'], lower_is_better=True)
        )
        table.add_row("─" * 20, "─" * 10, "─" * 10, "─" * 12)
        
        # TPOT (lower is better)
        table.add_row(
            "TPOT P50 (ms/tok)",
            f"{vllm_row['tpot_p50_ms']:.2f}",
            f"{tgi_row['tpot_p50_ms']:.2f}",
            fmt_advantage(vllm_row['tpot_p50_ms'], tgi_row['tpot_p50_ms'], lower_is_better=True)
        )
        table.add_row("─" * 20, "─" * 10, "─" * 10, "─" * 12)
        
        # E2E Latency (lower is better)
        table.add_row(
            "E2E Latency P50 (ms)",
            f"{vllm_row['latency_p50_ms']:.1f}",
            f"{tgi_row['latency_p50_ms']:.1f}",
            fmt_advantage(vllm_row['latency_p50_ms'], tgi_row['latency_p50_ms'], lower_is_better=True)
        )
        table.add_row(
            "E2E Latency P99 (ms)",
            f"{vllm_row['latency_p99_ms']:.1f}",
            f"{tgi_row['latency_p99_ms']:.1f}",
            fmt_advantage(vllm_row['latency_p99_ms'], tgi_row['latency_p99_ms'], lower_is_better=True)
        )
        
        console.print(table)
        console.print()


def generate_summary(vllm_df: pd.DataFrame, tgi_df: pd.DataFrame):
    """Generate executive summary across all concurrency levels."""
    common_levels = find_matching_results(vllm_df, tgi_df)
    
    if not common_levels:
        return
    
    # Calculate average advantages
    throughput_ratios = []
    ttft_ratios = []
    tpot_ratios = []
    
    for level in common_levels:
        vllm_row = vllm_df[vllm_df['concurrency'] == level].iloc[0]
        tgi_row = tgi_df[tgi_df['concurrency'] == level].iloc[0]
        
        if tgi_row['throughput_tok_s'] > 0:
            throughput_ratios.append(vllm_row['throughput_tok_s'] / tgi_row['throughput_tok_s'])
        if vllm_row['ttft_p50_ms'] > 0:
            ttft_ratios.append(tgi_row['ttft_p50_ms'] / vllm_row['ttft_p50_ms'])
        if vllm_row['tpot_p50_ms'] > 0:
            tpot_ratios.append(tgi_row['tpot_p50_ms'] / vllm_row['tpot_p50_ms'])
    
    avg_throughput = sum(throughput_ratios) / len(throughput_ratios) if throughput_ratios else 1
    avg_ttft = sum(ttft_ratios) / len(ttft_ratios) if ttft_ratios else 1
    avg_tpot = sum(tpot_ratios) / len(tpot_ratios) if tpot_ratios else 1
    
    def winner_str(ratio: float, metric: str) -> str:
        if ratio > 1.05:
            return f"[bold green]vLLM is {ratio:.1%} faster[/bold green]"
        elif ratio < 0.95:
            return f"[bold red]TGI is {1/ratio:.1%} faster[/bold red]"
        return "[dim]Roughly equivalent[/dim]"
    
    summary = f"""
[bold cyan]Concurrency Levels Compared:[/bold cyan] {common_levels}

[bold]Average Performance (vLLM vs TGI):[/bold]

  • [cyan]Throughput:[/cyan] {winner_str(avg_throughput, 'throughput')}
    vLLM averages {avg_throughput:.2f}x the tokens/sec of TGI

  • [cyan]Time to First Token (TTFT):[/cyan] {winner_str(avg_ttft, 'latency')}
    vLLM P50 is {avg_ttft:.2f}x faster than TGI

  • [cyan]Time Per Output Token (TPOT):[/cyan] {winner_str(avg_tpot, 'latency')}
    vLLM P50 is {avg_tpot:.2f}x faster than TGI

[bold]Recommendation:[/bold]
"""
    
    if avg_throughput > 1.2 or avg_ttft > 1.2:
        summary += "  [green]✓ vLLM shows clear advantages - recommended for production[/green]"
    elif avg_throughput < 0.8 or avg_ttft < 0.8:
        summary += "  [red]✗ TGI outperforms vLLM in this configuration[/red]"
    else:
        summary += "  [yellow]≈ Performance is similar - choose based on features/ecosystem[/yellow]"
    
    console.print(Panel(summary, title="[bold]Executive Summary[/bold]", border_style="cyan"))


@click.command()
@click.option("--vllm", "-v", default="", help="vLLM result CSV files (comma-separated or glob pattern, e.g. 'vllm_*.csv' or 'vllm_c4.csv,vllm_c8.csv')")
@click.option("--tgi", "-t", default="", help="TGI result CSV files (comma-separated or glob pattern, e.g. 'tgi_*.csv' or 'tgi_c4.csv,tgi_c8.csv')")
@click.option("--input-dir", "-d", type=click.Path(exists=True), help="Directory containing result CSVs (auto-detect by filename)")
@click.option("--output", "-o", help="Output combined CSV file (compatible with roi_calculator.py)")
@click.option("--summary-only", is_flag=True, help="Only show executive summary")
def compare(vllm: str, tgi: str, input_dir: Optional[str], output: Optional[str], summary_only: bool):
    """Compare vLLM vs TGI benchmark results from separate runs.
    
    Examples:
    
        # Specify files with glob pattern
        python compare_results.py --vllm "vllm_*.csv" --tgi "tgi_*.csv"
        
        # Specify files comma-separated
        python compare_results.py --vllm vllm_c4.csv,vllm_c8.csv --tgi tgi_c4.csv,tgi_c8.csv
        
        # Auto-detect from directory
        python compare_results.py --input-dir ./results
        
        # Generate combined CSV for ROI calculator
        python compare_results.py -d ./results -o sweep_results.csv
        
        # Current directory auto-detect
        python compare_results.py -d .
    """
    console.print(Panel("[bold]vLLM vs TGI Benchmark Comparison[/bold]", style="cyan"))
    
    # Parse comma-separated file lists or glob patterns
    vllm_files = []
    tgi_files = []
    
    if vllm:
        # Split by comma if present, otherwise treat as single pattern
        if ',' in vllm:
            vllm_files = [f.strip() for f in vllm.split(',') if f.strip()]
        else:
            vllm_files = [vllm]
    
    if tgi:
        if ',' in tgi:
            tgi_files = [f.strip() for f in tgi.split(',') if f.strip()]
        else:
            tgi_files = [tgi]
    
    # Auto-detect files from directory
    if input_dir:
        dir_path = Path(input_dir)
        vllm_files.extend([str(f) for f in dir_path.glob("*vllm*.csv")])
        tgi_files.extend([str(f) for f in dir_path.glob("*tgi*.csv")])
    
    if not vllm_files and not tgi_files:
        console.print("[red]Error: No input files specified. Use --vllm, --tgi, or --input-dir[/red]")
        raise SystemExit(1)
    
    console.print("\n[bold]Loading vLLM results:[/bold]")
    vllm_df = load_and_merge_csvs(vllm_files)
    
    console.print("\n[bold]Loading TGI results:[/bold]")
    tgi_df = load_and_merge_csvs(tgi_files)
    
    if vllm_df.empty:
        console.print("[red]Error: No vLLM results loaded[/red]")
        raise SystemExit(1)
    if tgi_df.empty:
        console.print("[red]Error: No TGI results loaded[/red]")
        raise SystemExit(1)
    
    # Ensure engine column is set correctly (lowercase to match roi_calculator.py)
    vllm_df['engine'] = 'vllm'
    tgi_df['engine'] = 'tgi'
    
    console.print(f"\n[cyan]vLLM: {len(vllm_df)} result(s) at concurrency levels {sorted(vllm_df['concurrency'].unique())}[/cyan]")
    console.print(f"[cyan]TGI: {len(tgi_df)} result(s) at concurrency levels {sorted(tgi_df['concurrency'].unique())}[/cyan]")
    console.print()
    
    # Display comparison tables
    if not summary_only:
        display_comparison(vllm_df, tgi_df)
    
    # Generate summary
    generate_summary(vllm_df, tgi_df)
    
    # Save combined output
    if output:
        combined_df = pd.concat([vllm_df, tgi_df], ignore_index=True)
        # Remove source_file column for compatibility with roi_calculator
        if 'source_file' in combined_df.columns:
            combined_df = combined_df.drop(columns=['source_file'])
        combined_df.to_csv(output, index=False)
        console.print(f"\n[green]✓ Combined results saved to {output}[/green]")
        console.print(f"  Run ROI analysis with: [cyan]python roi_calculator.py --input {output}[/cyan]")


if __name__ == "__main__":
    compare()

