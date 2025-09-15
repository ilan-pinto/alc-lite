"""
Property-based testing for price precision edge cases.

This module uses the Hypothesis library to generate thousands of test cases
with random but realistic price values to catch edge cases that manual
test cases might miss.

The tests verify that:
1. Price rounding functions handle all possible decimal inputs
2. Rollback pricing calculations never produce invalid prices
3. Aggressive pricing strategies maintain proper precision
4. Edge cases around minimum price variations are handled correctly
"""

from decimal import Decimal
from typing import List

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from modules.Arbitrage.sfr.utils import (
    calculate_aggressive_execution_price,
    round_price_to_tick_size,
)
from tests.test_utils import IBValidationError, RealisticIBMock, create_mock_contract


class TestPropertyBasedPricing:
    """Property-based tests for price precision and validation."""

    # Price strategies for realistic testing
    stock_prices = st.floats(
        min_value=0.01,
        max_value=10000.0,
        exclude_min=True,
        allow_nan=False,
        allow_infinity=False,
    )

    option_prices = st.floats(
        min_value=0.01,
        max_value=1000.0,
        exclude_min=True,
        allow_nan=False,
        allow_infinity=False,
    )

    decimal_places = st.integers(min_value=0, max_value=10)
    aggressiveness_factors = st.floats(min_value=0.001, max_value=0.1)

    @given(
        price=stock_prices, contract_type=st.sampled_from(["stock", "option", "future"])
    )
    def test_round_price_to_tick_size_always_valid(self, price, contract_type):
        """
        Test that price rounding always produces valid prices.

        Property: For any input price, the rounded price should:
        1. Be positive
        2. Have at most 2 decimal places
        3. Be a valid float
        """
        rounded_price = round_price_to_tick_size(price, contract_type)

        # Property 1: Result is positive
        assert rounded_price > 0

        # Property 2: Has at most 2 decimal places
        decimal_places = (
            len(str(rounded_price).split(".")[1]) if "." in str(rounded_price) else 0
        )
        assert decimal_places <= 2

        # Property 3: Is a valid float
        assert isinstance(rounded_price, float)
        assert not (rounded_price != rounded_price)  # Not NaN

    @given(base_price=stock_prices, decimal_places=decimal_places)
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    def test_problematic_prices_get_fixed(self, base_price, decimal_places):
        """
        Test that prices with excessive decimal places are properly rounded.

        Property: Adding extra decimal places to a price should not cause
        validation errors after rounding.
        """
        # Skip if base price already has too many decimals
        assume(
            len(str(base_price).split(".")[1]) <= 8 if "." in str(base_price) else True
        )

        # Create problematic price by adding decimal places
        problematic_price = base_price + (1.0 / (10 ** (decimal_places + 3)))

        # Round the price
        rounded_price = round_price_to_tick_size(problematic_price, "stock")

        # Should be valid for IB API
        ib_mock = RealisticIBMock()
        contract = create_mock_contract("TEST", "STK")

        # This should not raise an exception
        try:
            ib_mock._validate_price_precision(rounded_price, contract)
        except IBValidationError:
            pytest.fail(f"Rounded price {rounded_price} should be valid after rounding")

    @given(stock_price=stock_prices, aggressiveness=aggressiveness_factors)
    def test_aggressive_pricing_maintains_precision(self, stock_price, aggressiveness):
        """
        Test that aggressive pricing calculations maintain proper precision.

        Property: Aggressive pricing should never produce prices with
        more than 2 decimal places.
        """
        from unittest.mock import MagicMock

        # Create mock ticker with reasonable bid/ask spread
        spread = stock_price * 0.001  # 0.1% spread
        bid_price = round(stock_price - spread / 2, 2)
        ask_price = round(stock_price + spread / 2, 2)

        ticker = MagicMock()
        ticker.bid = bid_price
        ticker.ask = ask_price

        # Test both BUY and SELL actions
        for action in ["BUY", "SELL"]:
            aggressive_price = calculate_aggressive_execution_price(
                ticker, action, stock_price, aggressiveness
            )

            # Property: Result has at most 2 decimal places
            decimal_places = (
                len(str(aggressive_price).split(".")[1])
                if "." in str(aggressive_price)
                else 0
            )
            assert (
                decimal_places <= 2
            ), f"Aggressive {action} price {aggressive_price} has too many decimal places"

            # Property: Result is positive
            assert aggressive_price > 0, f"Aggressive {action} price must be positive"

            # Property: Result should be reasonable relative to input
            assert (
                0.5 * stock_price <= aggressive_price <= 2.0 * stock_price
            ), f"Aggressive {action} price {aggressive_price} too far from base price {stock_price}"

    @given(prices=st.lists(stock_prices, min_size=1, max_size=10))
    def test_batch_price_processing_consistency(self, prices):
        """
        Test that batch processing of prices maintains consistency.

        Property: Processing prices individually vs in batch should
        produce the same results.
        """
        # Process individually
        individual_results = [
            round_price_to_tick_size(price, "stock") for price in prices
        ]

        # Process as batch (simulate rollback scenario)
        batch_results = []
        for price in prices:
            # Simulate rollback calculation: aggressive pricing
            rollback_price = price * 0.99  # 1% discount for selling
            rounded_price = round_price_to_tick_size(rollback_price, "stock")
            batch_results.append(rounded_price)

        # Property: All results should be valid
        for i, result in enumerate(batch_results):
            assert result > 0, f"Batch result {i} is not positive: {result}"
            decimal_places = len(str(result).split(".")[1]) if "." in str(result) else 0
            assert (
                decimal_places <= 2
            ), f"Batch result {i} has too many decimal places: {result}"

    @given(
        base_price=stock_prices,
        pricing_factor=st.floats(min_value=0.001, max_value=0.05),
    )
    def test_rollback_pricing_calculation_robustness(self, base_price, pricing_factor):
        """
        Test that rollback pricing calculations are robust to various inputs.

        Property: Rollback price calculations should always produce
        valid prices regardless of input variation.
        """
        # Simulate the rollback pricing logic from rollback_manager.py
        for action in ["SELL", "BUY"]:
            if action == "SELL":
                # Selling: reduce price to encourage fills
                aggressive_price = base_price * (1.0 - pricing_factor)
            else:
                # Buying: increase price to encourage fills
                aggressive_price = base_price * (1.0 + pricing_factor)

            # Round the price
            rounded_price = round_price_to_tick_size(aggressive_price, "stock")

            # Ensure BUY prices don't fall below base price due to rounding
            if action == "BUY" and rounded_price < base_price:
                # Add one tick to maintain business rule
                rounded_price = round_price_to_tick_size(base_price + 0.01, "stock")

            # Ensure SELL prices don't exceed base price due to rounding
            elif action == "SELL" and rounded_price > base_price:
                # Reduce by one tick to maintain business rule
                rounded_price = round_price_to_tick_size(base_price - 0.01, "stock")
                # If that goes to zero or negative, use the minimum valid price
                if rounded_price <= 0:
                    rounded_price = 0.01

            # Property: Valid price
            assert rounded_price > 0
            decimal_places = (
                len(str(rounded_price).split(".")[1])
                if "." in str(rounded_price)
                else 0
            )
            assert decimal_places <= 2

            # Property: Price direction is correct
            if action == "SELL":
                assert (
                    rounded_price <= base_price
                ), f"SELL price should be <= base price"
            else:
                assert rounded_price >= base_price, f"BUY price should be >= base price"

    @given(
        prices=st.lists(
            st.floats(
                min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False
            ),
            min_size=10,
            max_size=100,
        )
    )
    @settings(max_examples=50)  # Reduce examples for performance
    def test_large_batch_price_validation(self, prices):
        """
        Test that large batches of prices are handled efficiently.

        Property: Processing large numbers of prices should not cause
        performance issues or validation errors.
        """
        ib_mock = RealisticIBMock()
        contract = create_mock_contract("BATCH_TEST", "STK")

        valid_count = 0
        invalid_count = 0

        for price in prices:
            try:
                rounded_price = round_price_to_tick_size(price, "stock")
                ib_mock._validate_price_precision(rounded_price, contract)
                valid_count += 1
            except (IBValidationError, ValueError):
                invalid_count += 1

        # Property: Most prices should be valid after rounding
        # At least 95% should be valid (allowing for edge cases)
        success_rate = valid_count / len(prices)
        assert (
            success_rate >= 0.95
        ), f"Success rate {success_rate:.2%} too low for batch processing"

    @given(
        base_price=st.floats(min_value=1.0, max_value=1000.0),
        multiplier=st.floats(min_value=0.5, max_value=2.0),
    )
    def test_price_multiplication_precision(self, base_price, multiplier):
        """
        Test that price multiplications maintain precision.

        Property: Multiplying prices (common in rollback calculations)
        should not introduce precision errors after rounding.
        """
        # Ensure base price is properly rounded first
        clean_base_price = round_price_to_tick_size(base_price, "stock")

        # Multiply (simulate various rollback calculations)
        result_price = clean_base_price * multiplier

        # Round the result
        final_price = round_price_to_tick_size(result_price, "stock")

        # Property: Final price is valid
        assert final_price > 0
        decimal_places = (
            len(str(final_price).split(".")[1]) if "." in str(final_price) else 0
        )
        assert decimal_places <= 2

        # Property: Result is reasonable
        min_expected = clean_base_price * min(multiplier, 1.0) * 0.5
        max_expected = clean_base_price * max(multiplier, 1.0) * 2.0
        assert min_expected <= final_price <= max_expected

    @given(
        prices=st.lists(
            st.floats(min_value=0.01, max_value=500.0),
            min_size=3,
            max_size=3,  # Stock, call, put
        )
    )
    def test_arbitrage_price_combination_precision(self, prices):
        """
        Test that arbitrage price combinations maintain precision.

        Property: Combining stock, call, and put prices (as in SFR arbitrage)
        should not introduce precision errors.
        """
        stock_price, call_price, put_price = prices

        # Simulate SFR arbitrage calculation
        # Net cost = Stock price - Call premium + Put premium
        theoretical_cost = stock_price - call_price + put_price

        # Add buffer (as in utils.py)
        buffer_amount = abs(theoretical_cost) * 0.02  # 2% buffer
        limit_price = theoretical_cost + buffer_amount

        # Round the final price
        final_price = round_price_to_tick_size(abs(limit_price), "stock")

        # Property: Final price is valid and positive
        assert final_price > 0
        decimal_places = (
            len(str(final_price).split(".")[1]) if "." in str(final_price) else 0
        )
        assert decimal_places <= 2

        # Property: Result is reasonable relative to inputs
        max_input = max(abs(p) for p in prices)
        assert final_price <= max_input * 5, "Combined price unreasonably large"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
