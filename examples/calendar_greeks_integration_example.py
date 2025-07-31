#!/usr/bin/env python3
"""
Calendar Greeks Integration Example

This example demonstrates how to use the new CalendarGreeks module
with the existing CalendarSpread, CalendarPnL, and TermStructure infrastructure.

Usage:
    python examples/calendar_greeks_integration_example.py
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import List

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ib_async import Contract

from modules.Arbitrage.CalendarGreeks import (
    AdjustmentType,
    CalendarGreeksCalculator,
    GreeksRiskLevel,
    calculate_calendar_greeks,
    monitor_portfolio_greeks,
)
from modules.Arbitrage.CalendarSpread import (
    CalendarSpreadConfig,
    CalendarSpreadLeg,
    CalendarSpreadOpportunity,
)


def create_sample_calendar_opportunity() -> CalendarSpreadOpportunity:
    """Create a sample calendar spread opportunity for testing"""

    # Create sample contracts
    front_contract = Contract(
        symbol="SPY",
        secType="OPT",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20241215",
        strike=540.0,
        right="C",
    )
    front_contract.conId = 12345

    back_contract = Contract(
        symbol="SPY",
        secType="OPT",
        exchange="SMART",
        currency="USD",
        lastTradeDateOrContractMonth="20250115",
        strike=540.0,
        right="C",
    )
    back_contract.conId = 67890

    # Create calendar spread legs
    front_leg = CalendarSpreadLeg(
        contract=front_contract,
        strike=540.0,
        expiry="20241215",
        right="C",
        price=8.50,
        bid=8.45,
        ask=8.55,
        volume=450,
        iv=22.5,  # 22.5% IV
        theta=-0.08,  # -$0.08 per day
        days_to_expiry=35,
    )

    back_leg = CalendarSpreadLeg(
        contract=back_contract,
        strike=540.0,
        expiry="20250115",
        right="C",
        price=12.80,
        bid=12.75,
        ask=12.85,
        volume=280,
        iv=19.2,  # 19.2% IV (inverted term structure)
        theta=-0.04,  # -$0.04 per day
        days_to_expiry=66,
    )

    # Create calendar spread opportunity
    opportunity = CalendarSpreadOpportunity(
        symbol="SPY",
        strike=540.0,
        option_type="CALL",
        front_leg=front_leg,
        back_leg=back_leg,
        iv_spread=front_leg.iv - back_leg.iv,  # 3.3% inversion
        theta_ratio=abs(front_leg.theta / back_leg.theta),  # 2.0 ratio
        net_debit=back_leg.price - front_leg.price,  # $4.30 debit
        max_profit=2.80,  # Estimated max profit
        max_loss=4.30,  # Net debit
        front_bid_ask_spread=0.01,
        back_bid_ask_spread=0.01,
        combined_liquidity_score=0.75,
        term_structure_inversion=True,
        net_delta=0.15,  # Slight positive delta
        net_gamma=0.08,
        net_vega=3.2,
        composite_score=0.82,
    )

    return opportunity


def demonstrate_basic_greeks_calculation():
    """Demonstrate basic Greeks calculation for a calendar spread"""
    print("\n" + "=" * 60)
    print("BASIC GREEKS CALCULATION DEMO")
    print("=" * 60)

    # Create sample opportunity
    opportunity = create_sample_calendar_opportunity()

    print(
        f"\nCalendar Spread: {opportunity.symbol} {opportunity.strike} {opportunity.option_type}"
    )
    print(
        f"Front Expiry: {opportunity.front_leg.expiry} ({opportunity.front_leg.days_to_expiry} days)"
    )
    print(
        f"Back Expiry: {opportunity.back_leg.expiry} ({opportunity.back_leg.days_to_expiry} days)"
    )
    print(f"Net Debit: ${opportunity.net_debit:.2f}")
    print(
        f"IV Spread: {opportunity.iv_spread:.1f}% (Front: {opportunity.front_leg.iv:.1f}%, Back: {opportunity.back_leg.iv:.1f}%)"
    )

    # Calculate Greeks using convenience function
    calendar_greeks = calculate_calendar_greeks(
        opportunity=opportunity,
        position_size=5,  # 5 contracts
        underlying_price=538.50,  # Current SPY price
        delta_threshold=0.20,  # 20 cents delta threshold
    )

    print(f"\n--- INDIVIDUAL LEG GREEKS ---")
    print(
        f"Front Leg - Delta: {calendar_greeks.front_delta:.3f}, Gamma: {calendar_greeks.front_gamma:.3f}, Vega: {calendar_greeks.front_vega:.2f}, Theta: {calendar_greeks.front_theta:.3f}"
    )
    print(
        f"Back Leg  - Delta: {calendar_greeks.back_delta:.3f}, Gamma: {calendar_greeks.back_gamma:.3f}, Vega: {calendar_greeks.back_vega:.2f}, Theta: {calendar_greeks.back_theta:.3f}"
    )

    print(f"\n--- NET POSITION GREEKS ---")
    print(
        f"Net Delta: {calendar_greeks.net_delta:.3f} {'(THRESHOLD EXCEEDED!)' if calendar_greeks.delta_threshold_exceeded else '(Within limits)'}"
    )
    print(
        f"Net Gamma: {calendar_greeks.net_gamma:.3f} {'(HIGH!)' if calendar_greeks.gamma_threshold_exceeded else '(Normal)'}"
    )
    print(
        f"Net Vega:  {calendar_greeks.net_vega:.2f} {'(HIGH!)' if calendar_greeks.vega_threshold_exceeded else '(Normal)'}"
    )
    print(f"Net Theta: {calendar_greeks.net_theta:.3f}")
    print(f"Net Rho:   {calendar_greeks.net_rho:.3f}")

    print(f"\n--- RISK ASSESSMENT ---")
    print(f"Overall Risk Level: {calendar_greeks.risk_level.value}")
    print(f"Overall Risk Score: {calendar_greeks.overall_risk_score:.3f}")
    print(f"Delta Risk Score: {calendar_greeks.delta_risk_score:.3f}")
    print(f"Gamma Risk Score: {calendar_greeks.gamma_risk_score:.3f}")
    print(f"Vega Risk Score: {calendar_greeks.vega_risk_score:.3f}")
    print(f"Theta Efficiency: {calendar_greeks.theta_efficiency_score:.3f}")

    print(f"\n--- POSITION HEALTH METRICS ---")
    delta_range = calendar_greeks.delta_neutral_range
    print(f"Delta Neutral Range: ${delta_range[0]:.2f} - ${delta_range[1]:.2f}")
    print(
        f"Gamma Acceleration Threshold: ${calendar_greeks.gamma_acceleration_threshold:.2f}"
    )
    vega_range = calendar_greeks.vega_sensitivity_range
    print(f"Vega Sensitivity IV Range: {vega_range[0]:.1f}% - {vega_range[1]:.1f}%")
    print(f"Theta Capture Efficiency: {calendar_greeks.theta_capture_efficiency:.3f}")

    # Show recommendations
    if calendar_greeks.recommended_adjustments:
        print(f"\n--- POSITION ADJUSTMENT RECOMMENDATIONS ---")
        for i, adj in enumerate(calendar_greeks.recommended_adjustments, 1):
            priority_text = ["", "URGENT", "IMPORTANT", "OPTIONAL"][adj.priority]
            print(f"{i}. [{priority_text}] {adj.adjustment_type.value}")
            print(f"   Reason: {adj.reason}")
            print(f"   Action: {adj.recommended_action}")
            print(f"   Timing: {adj.time_sensitivity}")
            if adj.expected_cost:
                print(f"   Est. Cost: ${adj.expected_cost:.2f}")
            print()
    else:
        print(f"\n--- NO IMMEDIATE ADJUSTMENTS NEEDED ---")

    return calendar_greeks


def demonstrate_portfolio_greeks():
    """Demonstrate portfolio-level Greeks aggregation"""
    print("\n" + "=" * 60)
    print("PORTFOLIO GREEKS AGGREGATION DEMO")
    print("=" * 60)

    # Create multiple calendar spread positions
    positions = []

    # Position 1: SPY Calendar
    spy_opportunity = create_sample_calendar_opportunity()
    spy_greeks = calculate_calendar_greeks(
        spy_opportunity, position_size=3, underlying_price=538.50
    )
    positions.append(spy_greeks)

    # Position 2: QQQ Calendar (simulate different characteristics)
    qqq_opportunity = create_sample_calendar_opportunity()
    qqq_opportunity.symbol = "QQQ"
    qqq_opportunity.strike = 450.0
    qqq_opportunity.front_leg.iv = 25.8  # Higher IV
    qqq_opportunity.back_leg.iv = 21.5
    qqq_greeks = calculate_calendar_greeks(
        qqq_opportunity, position_size=2, underlying_price=448.20
    )
    positions.append(qqq_greeks)

    # Position 3: TSLA Calendar (simulate high volatility)
    tsla_opportunity = create_sample_calendar_opportunity()
    tsla_opportunity.symbol = "TSLA"
    tsla_opportunity.strike = 250.0
    tsla_opportunity.front_leg.iv = 45.2  # Very high IV
    tsla_opportunity.back_leg.iv = 38.7
    tsla_greeks = calculate_calendar_greeks(
        tsla_opportunity, position_size=1, underlying_price=252.80
    )
    positions.append(tsla_greeks)

    # Aggregate portfolio Greeks
    portfolio_greeks = monitor_portfolio_greeks(
        positions, portfolio_id="sample_portfolio"
    )

    print(f"\nPortfolio ID: {portfolio_greeks.portfolio_id}")
    print(f"Total Positions: {len(portfolio_greeks.positions)}")
    print(f"\n--- PORTFOLIO LEVEL GREEKS ---")
    print(f"Total Delta: {portfolio_greeks.total_delta:.3f}")
    print(f"Total Gamma: {portfolio_greeks.total_gamma:.3f}")
    print(f"Total Vega:  {portfolio_greeks.total_vega:.2f}")
    print(f"Total Theta: {portfolio_greeks.total_theta:.3f}")
    print(f"Total Rho:   {portfolio_greeks.total_rho:.3f}")

    print(f"\n--- PORTFOLIO RISK METRICS ---")
    print(f"Portfolio Risk Score: {portfolio_greeks.portfolio_risk_score:.3f}")
    print(
        f"Correlation Adjusted Risk: {portfolio_greeks.correlation_adjusted_risk:.3f}"
    )

    print(f"\n--- CONCENTRATION ANALYSIS ---")
    for symbol, concentration in portfolio_greeks.concentration_risk.items():
        print(f"{symbol}: {concentration:.1%}")

    # Individual position summary
    print(f"\n--- INDIVIDUAL POSITION SUMMARY ---")
    for pos in positions:
        print(
            f"{pos.symbol}: Delta={pos.net_delta:.3f}, Risk={pos.risk_level.value}, Size={pos.position_size}"
        )

    # Portfolio-level recommendations
    if portfolio_greeks.portfolio_adjustments:
        print(f"\n--- PORTFOLIO ADJUSTMENT RECOMMENDATIONS ---")
        for i, adj in enumerate(portfolio_greeks.portfolio_adjustments, 1):
            priority_text = ["", "URGENT", "IMPORTANT", "OPTIONAL"][adj.priority]
            print(f"{i}. [{priority_text}] {adj.adjustment_type.value}")
            print(f"   Reason: {adj.reason}")
            print(f"   Action: {adj.recommended_action}")
            print(f"   Timing: {adj.time_sensitivity}")
            print()


def demonstrate_greeks_evolution():
    """Demonstrate Greeks evolution modeling"""
    print("\n" + "=" * 60)
    print("GREEKS EVOLUTION MODELING DEMO")
    print("=" * 60)

    # Create calculator
    calculator = CalendarGreeksCalculator()

    # Get calendar Greeks
    opportunity = create_sample_calendar_opportunity()
    calendar_greeks = calculator.calculate_calendar_greeks(
        opportunity, position_size=2, underlying_price=538.50
    )

    # Model evolution over 30 days
    evolution = calculator.model_greeks_evolution(
        calendar_greeks, time_horizon_days=30, num_price_points=10
    )

    print(
        f"\nEvolution Analysis for {calendar_greeks.symbol} {calendar_greeks.strike} {calendar_greeks.option_type}"
    )
    print(f"Time Horizon: {evolution.time_horizon_days} days")
    print(f"Price Scenarios: {len(evolution.underlying_price_scenarios)}")

    print(f"\n--- EXPECTED GREEKS EVOLUTION (30 days) ---")
    expected_delta = evolution.expected_values["delta"]
    expected_theta = evolution.expected_values["theta"]

    print(f"Day 0 Expected Delta: {expected_delta[0]:.3f}")
    print(f"Day 15 Expected Delta: {expected_delta[15]:.3f}")
    print(f"Day 30 Expected Delta: {expected_delta[30]:.3f}")

    print(f"Day 0 Expected Theta: {expected_theta[0]:.3f}")
    print(f"Day 15 Expected Theta: {expected_theta[15]:.3f}")
    print(f"Day 30 Expected Theta: {expected_theta[30]:.3f}")

    # Show some price scenario details
    print(f"\n--- PRICE SCENARIO ANALYSIS ---")
    prices = evolution.underlying_price_scenarios
    min_price, max_price = min(prices), max(prices)
    mid_price = prices[len(prices) // 2]

    print(f"Scenario Prices: ${min_price:.2f} to ${max_price:.2f}")
    print(f"Current Price Scenario (${mid_price:.2f}):")

    delta_path = evolution.delta_evolution[mid_price]
    vega_path = evolution.vega_evolution[mid_price]

    print(f"  Day 0 Delta: {delta_path[0]:.3f}, Vega: {vega_path[0]:.2f}")
    print(f"  Day 15 Delta: {delta_path[15]:.3f}, Vega: {vega_path[15]:.2f}")
    print(f"  Day 30 Delta: {delta_path[30]:.3f}, Vega: {vega_path[30]:.2f}")


def demonstrate_advanced_features():
    """Demonstrate advanced Greeks calculator features"""
    print("\n" + "=" * 60)
    print("ADVANCED FEATURES DEMO")
    print("=" * 60)

    # Create calculator with custom parameters
    calculator = CalendarGreeksCalculator(
        delta_threshold=0.15,  # Tighter delta threshold
        gamma_threshold=0.08,  # Lower gamma threshold
        vega_threshold=4.0,  # Lower vega threshold
        theta_min_efficiency=0.75,  # Higher theta requirement
    )

    opportunity = create_sample_calendar_opportunity()

    # Calculate with custom thresholds
    calendar_greeks = calculator.calculate_calendar_greeks(
        opportunity, position_size=4, underlying_price=540.25
    )

    print(f"\nCustom Threshold Analysis:")
    print(f"Delta Threshold: {calculator.delta_threshold} (vs standard 0.20)")
    print(
        f"Current Delta: {calendar_greeks.net_delta:.3f} {'[EXCEEDED]' if calendar_greeks.delta_threshold_exceeded else '[OK]'}"
    )

    # Test suggestion engine
    adjustments = calculator.suggest_position_adjustments(
        calendar_greeks, current_underlying_price=540.25
    )

    print(f"\n--- ADVANCED ADJUSTMENT SUGGESTIONS ---")
    for adj in adjustments:
        print(f"Type: {adj.adjustment_type.value}")
        print(
            f"Priority: {adj.priority} ({'URGENT' if adj.priority == 1 else 'NORMAL'})"
        )
        print(f"Reason: {adj.reason}")
        print(f"Action: {adj.recommended_action}")
        if adj.expected_cost:
            print(f"Cost: ${adj.expected_cost:.2f}")
        if adj.risk_reduction:
            print(f"Risk Reduction: {adj.risk_reduction:.3f}")
        print()

    # Cache statistics
    cache_stats = calculator.get_cache_stats()
    print(f"--- CACHE STATISTICS ---")
    for cache_type, size in cache_stats.items():
        print(f"{cache_type}: {size}")

    # Clear cache demonstration
    calculator.clear_cache()
    print(f"\nCache cleared successfully")


def main():
    """Main demonstration function"""
    print("Calendar Greeks Integration Module - Demonstration")
    print("This example shows how to use the new CalendarGreeks module")
    print("with existing calendar spread infrastructure.")

    try:
        # Run demonstrations
        calendar_greeks = demonstrate_basic_greeks_calculation()
        demonstrate_portfolio_greeks()
        demonstrate_greeks_evolution()
        demonstrate_advanced_features()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Integration Points:")
        print(
            "• CalendarGreeks works seamlessly with CalendarSpreadOpportunity objects"
        )
        print("• Portfolio-level Greeks aggregation for risk management")
        print("• Real-time position adjustment recommendations")
        print("• Greeks evolution modeling for scenario analysis")
        print("• Performance optimization with intelligent caching")
        print("• Comprehensive risk scoring and threshold monitoring")

        print(f"\nNext Steps:")
        print("• Integrate with live market data feeds")
        print("• Connect to Interactive Brokers for real-time Greeks")
        print("• Implement automated hedging based on recommendations")
        print("• Add historical Greeks tracking and analysis")
        print("• Enhance with machine learning risk models")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Check that all required modules are properly installed")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
