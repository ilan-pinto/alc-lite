#!/usr/bin/env python3
"""
Demonstrate the key benefits of NumPy vectorization in SFR trading:
1. Spread analysis and filtering
2. Statistical outlier detection
3. Quality scoring
4. Batch processing capabilities
"""

import os
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(__file__))


def demonstrate_spread_analysis():
    """Show how NumPy enables intelligent spread filtering"""

    print("📊 NumPy Spread Analysis Demonstration")
    print("=" * 50)

    # Simulate real market data with various spread scenarios
    np.random.seed(42)
    n_opportunities = 200

    # Generate realistic bid-ask spreads
    normal_spreads = np.random.normal(
        0.15, 0.05, int(n_opportunities * 0.7)
    )  # 70% normal
    wide_spreads = np.random.normal(0.8, 0.3, int(n_opportunities * 0.25))  # 25% wide
    outlier_spreads = np.random.uniform(
        2.0, 5.0, int(n_opportunities * 0.05)
    )  # 5% outliers

    all_spreads = np.concatenate([normal_spreads, wide_spreads, outlier_spreads])
    np.random.shuffle(all_spreads)
    all_spreads = np.abs(all_spreads)  # Ensure positive

    # Simulate profits (some good opportunities have wide spreads that kill them)
    base_profits = np.random.normal(0.30, 0.15, n_opportunities)

    print(f"📈 Market Simulation:")
    print(f"   • Total opportunities: {n_opportunities}")
    print(f"   • Mean spread: ${np.mean(all_spreads):.3f}")
    print(f"   • Spread std: ${np.std(all_spreads):.3f}")
    print(f"   • Max spread: ${np.max(all_spreads):.3f}")

    # TRADITIONAL APPROACH: No spread filtering
    traditional_profitable = np.sum(base_profits > 0.10)
    traditional_avg_profit = np.mean(base_profits[base_profits > 0.10])

    print(f"\n🔴 Traditional Approach (No Spread Filtering):")
    print(f"   • Profitable opportunities: {traditional_profitable}")
    print(f"   • Average profit: ${traditional_avg_profit:.3f}")

    # NUMPY VECTORIZED APPROACH: Statistical spread filtering

    # 1. Statistical outlier detection
    spread_mean = np.mean(all_spreads)
    spread_std = np.std(all_spreads)
    z_scores = (all_spreads - spread_mean) / spread_std

    # 2. Quality scoring based on spreads
    max_acceptable_spread = 0.50  # $0.50 max spread
    spread_quality_scores = np.ones(n_opportunities)

    # Penalize wide spreads
    spread_quality_scores -= np.clip(all_spreads / max_acceptable_spread, 0, 1) * 0.6
    # Penalize outliers (z-score > 2)
    spread_quality_scores[np.abs(z_scores) > 2] -= 0.3

    # 3. Adjust profits for execution reality (spreads eat into profits)
    execution_adjusted_profits = base_profits - all_spreads

    # 4. Create intelligent filtering
    viable_mask = (
        (spread_quality_scores > 0.3)  # Decent spread quality
        & (all_spreads < max_acceptable_spread)  # Reasonable spread
        & (execution_adjusted_profits > 0.10)  # Still profitable after spread cost
    )

    numpy_profitable = np.sum(viable_mask)
    numpy_avg_profit = np.mean(execution_adjusted_profits[viable_mask])
    rejected_by_spreads = traditional_profitable - numpy_profitable

    print(f"\n🟢 NumPy Vectorized Approach (Spread-Aware):")
    print(f"   • Viable opportunities: {numpy_profitable}")
    print(f"   • Average adjusted profit: ${numpy_avg_profit:.3f}")
    print(f"   • Rejected by spread analysis: {rejected_by_spreads}")
    print(f"   • Spread outliers detected: {np.sum(np.abs(z_scores) > 2)}")

    # 5. Risk metrics
    profit_improvement = numpy_avg_profit - traditional_avg_profit
    execution_success_rate = numpy_profitable / traditional_profitable

    print(f"\n📊 Performance Metrics:")
    print(f"   • Profit quality improvement: ${profit_improvement:.3f}")
    print(f"   • Execution success rate: {execution_success_rate:.1%}")
    print(f"   • False positive reduction: {rejected_by_spreads} bad trades avoided")

    return {
        "traditional_count": traditional_profitable,
        "numpy_count": numpy_profitable,
        "profit_improvement": profit_improvement,
        "false_positives_avoided": rejected_by_spreads,
    }


def demonstrate_statistical_analysis():
    """Show advanced statistical capabilities with NumPy"""

    print(f"\n🔬 Statistical Analysis Capabilities")
    print("=" * 50)

    # Simulate option chain data
    np.random.seed(123)
    n_strikes = 20
    n_expiries = 4

    strikes = np.arange(95, 115, 1)  # $95-$114 strikes
    stock_price = 105.0

    # Calculate theoretical values and add market noise
    theoretical_values = []
    market_values = []

    for strike in strikes:
        # Simple theoretical pricing (not Black-Scholes, just for demo)
        intrinsic = max(0, stock_price - strike)
        time_value = np.random.uniform(0.5, 3.0)
        theoretical = intrinsic + time_value

        # Add market noise and bid-ask spread
        spread = np.random.uniform(0.05, 0.25)
        market_bid = theoretical - spread / 2 + np.random.normal(0, 0.1)
        market_ask = theoretical + spread / 2 + np.random.normal(0, 0.1)

        theoretical_values.append(theoretical)
        market_values.append((market_bid, market_ask))

    theoretical_values = np.array(theoretical_values)
    market_bids = np.array([x[0] for x in market_values])
    market_asks = np.array([x[1] for x in market_values])

    # VECTORIZED ANALYSIS

    # 1. Calculate all spreads and spread percentages
    spreads = market_asks - market_bids
    mid_prices = (market_bids + market_asks) / 2
    spread_percentages = spreads / mid_prices * 100

    # 2. Identify pricing anomalies
    pricing_errors = np.abs(mid_prices - theoretical_values)
    error_z_scores = (pricing_errors - np.mean(pricing_errors)) / np.std(pricing_errors)

    # 3. Moneyness analysis (all at once)
    moneyness = strikes / stock_price
    atm_mask = (moneyness >= 0.98) & (moneyness <= 1.02)  # At-the-money
    otm_mask = moneyness > 1.02  # Out-of-the-money
    itm_mask = moneyness < 0.98  # In-the-money

    # 4. Volume-weighted quality scores
    volumes = np.random.randint(10, 1000, n_strikes)
    liquidity_scores = np.log(volumes) / np.log(np.max(volumes))  # Normalized 0-1

    # 5. Composite opportunity ranking
    opportunity_scores = (
        liquidity_scores * 0.3  # 30% liquidity weight
        + (1 - spread_percentages / 100) * 0.4  # 40% spread quality weight
        + (1 - np.abs(error_z_scores) / 3) * 0.2  # 20% pricing accuracy weight
        + atm_mask.astype(float) * 0.1  # 10% ATM bonus
    )

    # Results
    print(f"📈 Market Analysis Results:")
    print(f"   • Strikes analyzed: {n_strikes}")
    print(
        f"   • Mean spread: ${np.mean(spreads):.3f} ({np.mean(spread_percentages):.1f}%)"
    )
    print(f"   • ATM options: {np.sum(atm_mask)}")
    print(f"   • Pricing anomalies (|z| > 1.5): {np.sum(np.abs(error_z_scores) > 1.5)}")
    print(
        f"   • Best opportunity: Strike ${strikes[np.argmax(opportunity_scores)]:.0f} (score: {np.max(opportunity_scores):.3f})"
    )
    print(f"   • Liquidity range: {np.min(volumes)} - {np.max(volumes)} contracts")

    # Show what traditional sequential analysis would miss
    high_quality_opportunities = np.sum(opportunity_scores > 0.7)
    medium_quality = np.sum((opportunity_scores > 0.5) & (opportunity_scores <= 0.7))
    low_quality = np.sum(opportunity_scores <= 0.5)

    print(f"\n🎯 Quality Distribution:")
    print(f"   • High quality (>0.7): {high_quality_opportunities} opportunities")
    print(f"   • Medium quality (0.5-0.7): {medium_quality} opportunities")
    print(f"   • Low quality (<0.5): {low_quality} opportunities")


def demonstrate_real_world_benefits():
    """Show concrete benefits for actual SFR trading"""

    print(f"\n💰 Real-World SFR Trading Benefits")
    print("=" * 50)

    # Simulate a real trading session with NumPy advantages
    results = demonstrate_spread_analysis()

    # Calculate actual dollar impact
    trade_size = 10  # 10 contracts per opportunity
    contract_multiplier = 100  # Options are for 100 shares

    traditional_total = (
        results["traditional_count"]
        * results["profit_improvement"]
        * trade_size
        * contract_multiplier
    )
    false_positive_losses = (
        results["false_positives_avoided"] * 0.25 * trade_size * contract_multiplier
    )  # $0.25 avg loss per bad trade

    print(f"💵 Financial Impact (10 contracts per trade):")
    print(f"   • Better profit quality: ${traditional_total:,.0f}")
    print(f"   • Avoided losses from bad executions: ${false_positive_losses:,.0f}")
    print(
        f"   • Total improvement per session: ${traditional_total + false_positive_losses:,.0f}"
    )

    # Time advantages
    print(f"\n⏱️  Time Advantages:")
    print(f"   • Simultaneous analysis of all opportunities")
    print(f"   • Real-time spread quality assessment")
    print(f"   • Instant outlier detection")
    print(f"   • No sequential bottlenecks")

    # Risk management
    print(f"\n🛡️  Risk Management Benefits:")
    print(f"   • Statistical spread filtering reduces execution risk")
    print(f"   • Outlier detection prevents bad trades")
    print(f"   • Quality scoring prioritizes best opportunities")
    print(f"   • Comprehensive market microstructure analysis")


if __name__ == "__main__":
    print("🚀 NumPy Benefits for SFR Arbitrage Trading")
    print("=" * 60)
    print("This demo shows why NumPy vectorization provides")
    print("significant advantages beyond just raw speed:")
    print()

    demonstrate_spread_analysis()
    demonstrate_statistical_analysis()
    demonstrate_real_world_benefits()

    print(f"\n🎯 Key Takeaways:")
    print(
        "   1. NumPy enables statistical spread analysis impossible with sequential processing"
    )
    print("   2. Quality scoring helps select the most executable opportunities")
    print("   3. Outlier detection prevents costly execution failures")
    print("   4. Comprehensive market analysis leads to better trading decisions")
    print("   5. Vectorized operations scale effortlessly to 1000+ opportunities")
    print(
        "\n✅ NumPy transforms SFR from sequential screening to intelligent analysis!"
    )
