"""Generate Markdown reports from forensic analysis results"""
from pathlib import Path
from typing import Dict, List
import json


def generate_markdown_report(analysis_results: Dict, output_path: Path) -> None:
    """
    Generate a comprehensive Markdown report from analysis results.
    
    Args:
        analysis_results: Dictionary containing all breakdown results
        output_path: Path where the Markdown file will be written
    """
    lines = []
    
    # Header
    run_name = analysis_results.get('run_name', 'unknown')
    lines.append(f"# Step 2 — Forensic Analysis (Run: {run_name})")
    lines.append("")
    lines.append(f"*Generated from: {analysis_results.get('run_dir', 'unknown')}*")
    lines.append("")
    
    # 1. Overview
    lines.append("## 1. Overview")
    lines.append("")
    metrics = analysis_results.get('metrics', {})
    lines.extend(_format_overview(metrics))
    lines.append("")
    
    # 2. Time Breakdown
    lines.append("## 2. Time Breakdown")
    lines.append("")
    time_breakdown = analysis_results.get('time_breakdown', {})
    lines.extend(_format_time_breakdown(time_breakdown))
    lines.append("")
    
    # 3. Module Breakdown
    lines.append("## 3. Module Breakdown")
    lines.append("")
    module_breakdown = analysis_results.get('module_breakdown', {})
    lines.extend(_format_module_breakdown(module_breakdown))
    lines.append("")
    
    # 4. Symbol & Direction Breakdown
    lines.append("## 4. Symbol & Direction Breakdown")
    lines.append("")
    symbol_breakdown = analysis_results.get('symbol_breakdown', {})
    direction_breakdown = analysis_results.get('direction_breakdown', {})
    lines.extend(_format_symbol_direction_breakdown(symbol_breakdown, direction_breakdown))
    lines.append("")
    
    # 5. Cost Decomposition
    lines.append("## 5. Cost Decomposition")
    lines.append("")
    cost_decomposition = analysis_results.get('cost_decomposition', {})
    lines.extend(_format_cost_decomposition(cost_decomposition))
    lines.append("")
    
    # 6. Findings & Suspected Failure Modes
    lines.append("## 6. Findings & Suspected Failure Modes")
    lines.append("")
    lines.extend(_format_findings(analysis_results))
    lines.append("")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _format_overview(metrics: Dict) -> List[str]:
    """Format overview section from metrics.json"""
    lines = []
    
    # Key metrics
    total_return = metrics.get('total_return', 0.0) * 100
    cagr = metrics.get('cagr', 0.0) * 100
    max_dd_pct = metrics.get('max_drawdown_pct', 0.0) * 100
    sharpe = metrics.get('sharpe', 0.0)
    sortino = metrics.get('sortino', 0.0)
    calmar = metrics.get('calmar', 0.0)
    profit_factor = metrics.get('profit_factor', 0.0)
    win_rate = metrics.get('win_rate', 0.0) * 100
    avg_win = metrics.get('avg_win', 0.0)
    avg_loss = metrics.get('avg_loss', 0.0)
    total_trades = metrics.get('total_trades', 0)
    exposure_pct = metrics.get('exposure_pct', 0.0)
    
    lines.append("### Key Performance Metrics")
    lines.append("")
    lines.append(f"- **Total Return**: {total_return:.2f}%")
    lines.append(f"- **CAGR**: {cagr:.2f}%")
    lines.append(f"- **Max Drawdown**: {max_dd_pct:.2f}%")
    lines.append(f"- **Sharpe Ratio**: {sharpe:.4f}")
    lines.append(f"- **Sortino Ratio**: {sortino:.4f}")
    lines.append(f"- **Calmar Ratio**: {calmar:.4f}")
    lines.append(f"- **Profit Factor**: {profit_factor:.4f}")
    lines.append(f"- **Win Rate**: {win_rate:.2f}%")
    lines.append(f"- **Avg Win**: ${avg_win:.2f}")
    lines.append(f"- **Avg Loss**: ${avg_loss:.2f}")
    lines.append(f"- **Total Trades**: {total_trades}")
    lines.append(f"- **Exposure %**: {exposure_pct:.2f}%")
    lines.append("")
    
    # Cost summary
    total_fees = metrics.get('total_fees', 0.0)
    funding_cost = metrics.get('funding_cost_total', 0.0)
    signal_pnl_before_costs = metrics.get('signal_pnl_before_costs', 0.0)
    total_costs = metrics.get('total_costs', 0.0)
    
    lines.append("### Cost Summary")
    lines.append("")
    lines.append(f"- **Signal PnL (before costs)**: ${signal_pnl_before_costs:,.2f}")
    lines.append(f"- **Total Fees**: ${total_fees:,.2f}")
    lines.append(f"- **Total Funding**: ${funding_cost:,.2f}")
    lines.append(f"- **Total Costs**: ${total_costs:,.2f}")
    if signal_pnl_before_costs != 0:
        cost_pct = (total_costs / abs(signal_pnl_before_costs)) * 100
        lines.append(f"- **Cost % of Gross PnL**: {cost_pct:.2f}%")
    lines.append("")
    
    return lines


def _format_time_breakdown(time_breakdown: Dict) -> List[str]:
    """Format time breakdown section"""
    lines = []
    
    # Year breakdown
    year_data = time_breakdown.get('year', {})
    if year_data:
        lines.append("### By Year")
        lines.append("")
        lines.append("| Year | Net PnL (USD) | Trades | Win Rate | Profit Factor | Max DD |")
        lines.append("|------|---------------|--------|----------|---------------|--------|")
        for year in sorted(year_data.keys()):
            data = year_data[year]
            lines.append(
                f"| {year} | ${data['net_pnl']:,.2f} | {data['trades']} | "
                f"{data['win_rate']*100:.1f}% | {_format_pf(data['profit_factor'])} | "
                f"{data['max_drawdown']*100:.2f}% |"
            )
        lines.append("")
    
    # Quarter breakdown
    quarter_data = time_breakdown.get('quarter', {})
    if quarter_data:
        lines.append("### By Quarter")
        lines.append("")
        lines.append("| Quarter | Net PnL (USD) | Trades | Win Rate | Profit Factor | Max DD |")
        lines.append("|---------|---------------|--------|----------|---------------|--------|")
        for quarter in sorted(quarter_data.keys()):
            data = quarter_data[quarter]
            lines.append(
                f"| {quarter} | ${data['net_pnl']:,.2f} | {data['trades']} | "
                f"{data['win_rate']*100:.1f}% | {_format_pf(data['profit_factor'])} | "
                f"{data['max_drawdown']*100:.2f}% |"
            )
        lines.append("")
    
    # Month breakdown (top 10 worst and best)
    month_data = time_breakdown.get('month', {})
    if month_data:
        # Sort by net_pnl
        sorted_months = sorted(month_data.items(), key=lambda x: x[1]['net_pnl'])
        worst_months = sorted_months[:10]
        best_months = sorted(sorted_months[-10:], key=lambda x: x[1]['net_pnl'], reverse=True)
        
        if worst_months:
            lines.append("### Worst Months (Top 10)")
            lines.append("")
            lines.append("| Month | Net PnL (USD) | Trades | Win Rate | Profit Factor |")
            lines.append("|-------|---------------|--------|----------|---------------|")
            for month, data in worst_months:
                lines.append(
                    f"| {month} | ${data['net_pnl']:,.2f} | {data['trades']} | "
                    f"{data['win_rate']*100:.1f}% | {_format_pf(data['profit_factor'])} |"
                )
            lines.append("")
        
        if best_months:
            lines.append("### Best Months (Top 10)")
            lines.append("")
            lines.append("| Month | Net PnL (USD) | Trades | Win Rate | Profit Factor |")
            lines.append("|-------|---------------|--------|----------|---------------|")
            for month, data in best_months:
                lines.append(
                    f"| {month} | ${data['net_pnl']:,.2f} | {data['trades']} | "
                    f"{data['win_rate']*100:.1f}% | {_format_pf(data['profit_factor'])} |"
                )
            lines.append("")
    
    # Commentary
    lines.append("### Commentary")
    lines.append("")
    if year_data:
        worst_year = min(year_data.items(), key=lambda x: x[1]['net_pnl'])
        best_year = max(year_data.items(), key=lambda x: x[1]['net_pnl'])
        lines.append(
            f"- Worst performing year: {worst_year[0]} (${worst_year[1]['net_pnl']:,.2f}, "
            f"{worst_year[1]['trades']} trades)"
        )
        lines.append(
            f"- Best performing year: {best_year[0]} (${best_year[1]['net_pnl']:,.2f}, "
            f"{best_year[1]['trades']} trades)"
        )
    lines.append("")
    
    return lines


def _format_module_breakdown(module_breakdown: Dict) -> List[str]:
    """Format module breakdown section"""
    lines = []
    
    if not module_breakdown:
        lines.append("*No module data available.*")
        lines.append("")
        return lines
    
    lines.append("### Module Performance Summary")
    lines.append("")
    lines.append(
        "| Module | Trades | Net PnL (USD) | Win Rate | Profit Factor | "
        "Avg Gross PnL/Trade | Avg Net PnL/Trade | Avg Holding (bars) | Cost Ratio |"
    )
    lines.append(
        "|--------|--------|---------------|----------|---------------|"
        "---------------------|-------------------|-------------------|-----------|"
    )
    
    for module in sorted(module_breakdown.keys()):
        data = module_breakdown[module]
        cost_ratio = data.get('cost_ratio', 0.0)
        holding_time = data.get('avg_holding_time_bars', 0.0)
        lines.append(
            f"| {module} | {data['trades']} | ${data['net_pnl']:,.2f} | "
            f"{data['win_rate']*100:.1f}% | {_format_pf(data['profit_factor'])} | "
            f"${data['avg_gross_pnl_per_trade']:,.2f} | ${data['avg_net_pnl_per_trade']:,.2f} | "
            f"{holding_time:.1f} | {cost_ratio:.4f} |"
        )
    lines.append("")
    
    # Commentary
    lines.append("### Commentary")
    lines.append("")
    worst_module = min(module_breakdown.items(), key=lambda x: x[1]['net_pnl'])
    lines.append(
        f"- **{worst_module[0]}** is the largest PnL drain: ${worst_module[1]['net_pnl']:,.2f} "
        f"({worst_module[1]['trades']} trades, {worst_module[1]['win_rate']*100:.1f}% win rate)"
    )
    
    # Cost analysis
    for module, data in module_breakdown.items():
        gross_pnl = data.get('gross_signal_pnl', 0.0)
        total_costs = data.get('total_costs', 0.0)
        if gross_pnl > 0 and total_costs > 0:
            cost_pct = (total_costs / gross_pnl) * 100
            if cost_pct > 50:
                lines.append(
                    f"- **{module}** has positive gross PnL (${gross_pnl:,.2f}) but costs "
                    f"({cost_pct:.1f}% of gross) are eating into profits"
                )
    lines.append("")
    
    return lines


def _format_symbol_direction_breakdown(
    symbol_breakdown: Dict,
    direction_breakdown: Dict
) -> List[str]:
    """Format symbol and direction breakdown section"""
    lines = []
    
    # Symbol breakdown
    by_symbol = symbol_breakdown.get('by_symbol', {})
    if by_symbol:
        lines.append("### By Symbol")
        lines.append("")
        lines.append("| Symbol | Trades | Net PnL (USD) | Win Rate | Profit Factor | Avg Holding (bars) |")
        lines.append("|--------|--------|---------------|----------|---------------|-------------------|")
        
        # Sort by net_pnl
        sorted_symbols = sorted(by_symbol.items(), key=lambda x: x[1]['net_pnl'])
        for symbol, data in sorted_symbols:
            holding_time = data.get('avg_holding_time_bars', 0.0)
            lines.append(
                f"| {symbol} | {data['trades']} | ${data['net_pnl']:,.2f} | "
                f"{data['win_rate']*100:.1f}% | {_format_pf(data['profit_factor'])} | "
                f"{holding_time:.1f} |"
            )
        lines.append("")
    
    # Module×Symbol breakdown (top losers)
    by_module_symbol = symbol_breakdown.get('by_module_symbol', {})
    if by_module_symbol:
        lines.append("### By Module×Symbol (Top 10 Losers)")
        lines.append("")
        lines.append("| Module-Symbol | Trades | Net PnL (USD) | Win Rate | Profit Factor |")
        lines.append("|--------------|--------|---------------|----------|---------------|")
        
        sorted_ms = sorted(by_module_symbol.items(), key=lambda x: x[1]['net_pnl'])[:10]
        for key, data in sorted_ms:
            lines.append(
                f"| {key} | {data['trades']} | ${data['net_pnl']:,.2f} | "
                f"{data['win_rate']*100:.1f}% | {_format_pf(data['profit_factor'])} |"
            )
        lines.append("")
    
    # Direction breakdown
    overall_dir = direction_breakdown.get('overall', {})
    if overall_dir:
        lines.append("### By Direction (Overall)")
        lines.append("")
        lines.append("| Direction | Trades | Net PnL (USD) | Win Rate | Profit Factor |")
        lines.append("|-----------|--------|---------------|----------|---------------|")
        for direction in sorted(overall_dir.keys()):
            data = overall_dir[direction]
            lines.append(
                f"| {direction} | {data['trades']} | ${data['net_pnl']:,.2f} | "
                f"{data['win_rate']*100:.1f}% | {_format_pf(data['profit_factor'])} |"
            )
        lines.append("")
    
    # Module×Direction breakdown
    by_module_dir = direction_breakdown.get('by_module', {})
    if by_module_dir:
        lines.append("### By Module×Direction")
        lines.append("")
        lines.append("| Module-Direction | Trades | Net PnL (USD) | Win Rate | Profit Factor |")
        lines.append("|-----------------|--------|---------------|----------|---------------|")
        for key in sorted(by_module_dir.keys()):
            data = by_module_dir[key]
            lines.append(
                f"| {key} | {data['trades']} | ${data['net_pnl']:,.2f} | "
                f"{data['win_rate']*100:.1f}% | {_format_pf(data['profit_factor'])} |"
            )
        lines.append("")
    
    # Commentary
    lines.append("### Commentary")
    lines.append("")
    if by_symbol:
        worst_symbol = min(by_symbol.items(), key=lambda x: x[1]['net_pnl'])
        best_symbol = max(by_symbol.items(), key=lambda x: x[1]['net_pnl'])
        lines.append(
            f"- Worst symbol: **{worst_symbol[0]}** (${worst_symbol[1]['net_pnl']:,.2f}, "
            f"{worst_symbol[1]['trades']} trades)"
        )
        lines.append(
            f"- Best symbol: **{best_symbol[0]}** (${best_symbol[1]['net_pnl']:,.2f}, "
            f"{best_symbol[1]['trades']} trades)"
        )
    lines.append("")
    
    return lines


def _format_cost_decomposition(cost_decomposition: Dict) -> List[str]:
    """Format cost decomposition section"""
    lines = []
    
    overall = cost_decomposition.get('overall', {})
    if overall:
        lines.append("### Overall Cost Breakdown")
        lines.append("")
        lines.append("| Metric | Value (USD) |")
        lines.append("|--------|-------------|")
        lines.append(f"| Gross PnL (before costs) | ${overall.get('gross_pnl_before_costs', 0.0):,.2f} |")
        lines.append(f"| Total Fees | ${overall.get('total_fees', 0.0):,.2f} |")
        lines.append(f"| Total Slippage | ${overall.get('total_slippage', 0.0):,.2f} |")
        lines.append(f"| Total Funding | ${overall.get('total_funding', 0.0):,.2f} |")
        lines.append(f"| Total Costs | ${overall.get('total_costs', 0.0):,.2f} |")
        lines.append(f"| Net PnL | ${overall.get('net_pnl', 0.0):,.2f} |")
        cost_share = overall.get('cost_share', 0.0)
        lines.append(f"| Cost Share | {cost_share:.4f} |")
        lines.append(f"| Avg Cost per Trade | ${overall.get('avg_cost_per_trade', 0.0):,.2f} |")
        lines.append(f"| Cost per Trade (p50) | ${overall.get('cost_per_trade_p50', 0.0):,.2f} |")
        lines.append(f"| Cost per Trade (p90) | ${overall.get('cost_per_trade_p90', 0.0):,.2f} |")
        lines.append("")
    
    # By module
    by_module = cost_decomposition.get('by_module', {})
    if by_module:
        lines.append("### Cost Breakdown by Module")
        lines.append("")
        lines.append(
            "| Module | Gross PnL | Fees | Slippage | Funding | Total Costs | "
            "Net PnL | Cost Share |"
        )
        lines.append(
            "|--------|-----------|------|----------|---------|------------|"
            "---------|------------|"
        )
        for module in sorted(by_module.keys()):
            data = by_module[module]
            lines.append(
                f"| {module} | ${data.get('gross_pnl_before_costs', 0.0):,.2f} | "
                f"${data.get('total_fees', 0.0):,.2f} | ${data.get('total_slippage', 0.0):,.2f} | "
                f"${data.get('total_funding', 0.0):,.2f} | ${data.get('total_costs', 0.0):,.2f} | "
                f"${data.get('net_pnl', 0.0):,.2f} | {data.get('cost_share', 0.0):.4f} |"
            )
        lines.append("")
    
    # By symbol (top 10 by absolute costs)
    by_symbol = cost_decomposition.get('by_symbol', {})
    if by_symbol:
        lines.append("### Cost Breakdown by Symbol (Top 10 by Total Costs)")
        lines.append("")
        lines.append(
            "| Symbol | Gross PnL | Fees | Slippage | Funding | Total Costs | "
            "Net PnL | Cost Share |"
        )
        lines.append(
            "|--------|-----------|------|----------|---------|------------|"
            "---------|------------|"
        )
        sorted_symbols = sorted(by_symbol.items(), key=lambda x: x[1].get('total_costs', 0.0), reverse=True)[:10]
        for symbol, data in sorted_symbols:
            lines.append(
                f"| {symbol} | ${data.get('gross_pnl_before_costs', 0.0):,.2f} | "
                f"${data.get('total_fees', 0.0):,.2f} | ${data.get('total_slippage', 0.0):,.2f} | "
                f"${data.get('total_funding', 0.0):,.2f} | ${data.get('total_costs', 0.0):,.2f} | "
                f"${data.get('net_pnl', 0.0):,.2f} | {data.get('cost_share', 0.0):.4f} |"
            )
        lines.append("")
    
    # Commentary
    lines.append("### Commentary")
    lines.append("")
    if by_module:
        highest_cost_module = max(by_module.items(), key=lambda x: x[1].get('total_costs', 0.0))
        lines.append(
            f"- **{highest_cost_module[0]}** has the highest total costs: "
            f"${highest_cost_module[1].get('total_costs', 0.0):,.2f}"
        )
    lines.append("")
    
    return lines


def _format_findings(analysis_results: Dict) -> List[str]:
    """Format findings and suspected failure modes"""
    lines = []
    
    metrics = analysis_results.get('metrics', {})
    module_breakdown = analysis_results.get('module_breakdown', {})
    symbol_breakdown = analysis_results.get('symbol_breakdown', {})
    direction_breakdown = analysis_results.get('direction_breakdown', {})
    cost_decomposition = analysis_results.get('cost_decomposition', {})
    time_breakdown = analysis_results.get('time_breakdown', {})
    
    findings = []
    
    # Finding 1: Overall performance
    total_return = metrics.get('total_return', 0.0) * 100
    total_trades = metrics.get('total_trades', 0)
    findings.append({
        'fact': f"Total return over the period: {total_return:.2f}% ({total_trades} trades)",
        'interpretation': "The strategy lost money overall, indicating a structural issue rather than isolated bad periods."
    })
    
    # Finding 2: Module performance
    if module_breakdown:
        worst_module = min(module_breakdown.items(), key=lambda x: x[1]['net_pnl'])
        worst_pnl = worst_module[1]['net_pnl']
        worst_trades = worst_module[1]['trades']
        worst_wr = worst_module[1]['win_rate'] * 100
        findings.append({
            'fact': f"{worst_module[0]} module lost ${worst_pnl:,.2f} across {worst_trades} trades with {worst_wr:.1f}% win rate",
            'interpretation': f"{worst_module[0]} appears to be the primary source of losses, suggesting the strategy logic for this module may be flawed or mis-calibrated."
        })
    
    # Finding 3: Win rate
    win_rate = metrics.get('win_rate', 0.0) * 100
    profit_factor = metrics.get('profit_factor', 0.0)
    findings.append({
        'fact': f"Win rate: {win_rate:.1f}%, Profit factor: {profit_factor:.4f}",
        'interpretation': "Low win rate combined with profit factor < 1.0 indicates that losses are larger than wins on average, or that costs are eroding edge."
    })
    
    # Finding 4: Symbol concentration
    by_symbol = symbol_breakdown.get('by_symbol', {})
    if by_symbol:
        worst_symbol = min(by_symbol.items(), key=lambda x: x[1]['net_pnl'])
        worst_sym_pnl = worst_symbol[1]['net_pnl']
        worst_sym_trades = worst_symbol[1]['trades']
        findings.append({
            'fact': f"{worst_symbol[0]} contributed ${worst_sym_pnl:,.2f} in losses ({worst_sym_trades} trades)",
            'interpretation': f"Concentration of losses in {worst_symbol[0]} suggests either poor signal quality for this symbol or execution issues specific to this market."
        })
    
    # Finding 5: Direction bias
    overall_dir = direction_breakdown.get('overall', {})
    if overall_dir and 'LONG' in overall_dir and 'SHORT' in overall_dir:
        long_pnl = overall_dir['LONG']['net_pnl']
        short_pnl = overall_dir['SHORT']['net_pnl']
        findings.append({
            'fact': f"LONG trades: ${long_pnl:,.2f}, SHORT trades: ${short_pnl:,.2f}",
            'interpretation': "Directional bias analysis reveals whether the strategy has a systematic bias toward one direction, which could indicate regime misclassification or market structure issues."
        })
    
    # Finding 6: Cost impact
    overall_costs = cost_decomposition.get('overall', {})
    if overall_costs:
        gross_pnl = overall_costs.get('gross_pnl_before_costs', 0.0)
        total_costs = overall_costs.get('total_costs', 0.0)
        cost_share = overall_costs.get('cost_share', 0.0)
        if gross_pnl > 0:
            findings.append({
                'fact': f"Gross PnL before costs: ${gross_pnl:,.2f}, Total costs: ${total_costs:,.2f} ({cost_share:.2%} of gross)",
                'interpretation': "Costs are consuming a significant portion of gross profits, suggesting that either execution costs are too high or the strategy needs higher edge per trade to overcome costs."
            })
        elif gross_pnl < 0:
            findings.append({
                'fact': f"Gross PnL before costs: ${gross_pnl:,.2f}, Total costs: ${total_costs:,.2f}",
                'interpretation': "Even before costs, the strategy has negative gross PnL, indicating that the signal itself lacks edge rather than being eroded by execution costs."
            })
    
    # Finding 7: Time concentration
    year_data = time_breakdown.get('year', {})
    if year_data:
        worst_year = min(year_data.items(), key=lambda x: x[1]['net_pnl'])
        findings.append({
            'fact': f"Worst year: {worst_year[0]} with ${worst_year[1]['net_pnl']:,.2f} in losses ({worst_year[1]['trades']} trades)",
            'interpretation': f"Losses concentrated in {worst_year[0]} may indicate regime changes, market structure shifts, or parameter drift that the strategy failed to adapt to."
        })
    
    # Finding 8: Module cost efficiency
    by_module_costs = cost_decomposition.get('by_module', {})
    if by_module_costs and module_breakdown:
        for module in module_breakdown.keys():
            if module in by_module_costs:
                module_costs = by_module_costs[module]
                module_metrics = module_breakdown[module]
                gross_pnl = module_costs.get('gross_pnl_before_costs', 0.0)
                total_costs = module_costs.get('total_costs', 0.0)
                if gross_pnl > 0 and total_costs > gross_pnl:
                    findings.append({
                        'fact': f"{module} has positive gross PnL (${gross_pnl:,.2f}) but costs (${total_costs:,.2f}) exceed it",
                        'interpretation': f"{module} shows signal edge but execution costs are too high, suggesting either excessive trading frequency, poor execution, or insufficient position sizing."
                    })
                    break
    
    # Format findings
    for i, finding in enumerate(findings[:10], 1):  # Limit to 10 findings
        lines.append(f"**Finding {i}:**")
        lines.append("")
        lines.append(f"- **FACT**: {finding['fact']}")
        lines.append(f"- **INTERPRETATION**: {finding['interpretation']}")
        lines.append("")
    
    return lines


def _format_pf(profit_factor: float) -> str:
    """Format profit factor, handling infinity"""
    if profit_factor >= 999999.0:
        return "inf"
    elif profit_factor == 0.0:
        return "0.00"
    else:
        return f"{profit_factor:.2f}"

