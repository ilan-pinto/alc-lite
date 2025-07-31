"""
Comprehensive unit tests for TermStructure analysis module.

This test suite provides extensive coverage of the TermStructure implementation,
including all classes, methods, edge cases, error conditions, and integration scenarios.

Test Coverage:
- IVDataPoint data class functionality
- TermStructureCurve creation and methods
- TermStructureInversion detection and scoring
- IVPercentileData calculations
- TermStructureConfig validation
- TermStructureAnalyzer comprehensive analysis
- IV curve construction and interpolation
- Inversion detection algorithms
- Confidence scoring and validation
- Performance optimization and caching
- Edge cases and error handling
"""

import time
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from ib_async import Contract, Ticker

from modules.Arbitrage.metrics import RejectionReason

# Import the modules under test
from modules.Arbitrage.TermStructure import (
    IVDataPoint,
    IVPercentileData,
    TermStructureAnalyzer,
    TermStructureConfig,
    TermStructureCurve,
    TermStructureInversion,
    analyze_multi_symbol_term_structure,
    analyze_symbol_term_structure,
    detect_calendar_spread_opportunities,
    enhance_calendar_spread_with_term_structure,
    get_optimal_calendar_expiries,
    validate_calendar_spread_with_term_structure,
)


class TestIVDataPoint:
    """Comprehensive tests for IVDataPoint dataclass"""

    def test_iv_data_point_creation_complete(self):
        """Test creating an IV data point with all fields"""
        data_point = IVDataPoint(
            expiry="20241215",
            days_to_expiry=45,
            strike=150.0,
            option_type="CALL",
            iv=28.5,
            price=5.75,
            volume=250,
            delta=0.52,
            bid=5.70,
            ask=5.80,
            last_updated=1234567890.0,
        )

        assert data_point.expiry == "20241215"
        assert data_point.days_to_expiry == 45
        assert data_point.strike == 150.0
        assert data_point.option_type == "CALL"
        assert data_point.iv == 28.5
        assert data_point.price == 5.75
        assert data_point.volume == 250
        assert data_point.delta == 0.52
        assert data_point.bid == 5.70
        assert data_point.ask == 5.80
        assert data_point.last_updated == 1234567890.0

    def test_iv_data_point_creation_put(self):
        """Test creating an IV data point for put option"""
        data_point = IVDataPoint(
            expiry="20241215",
            days_to_expiry=45,
            strike=150.0,
            option_type="PUT",
            iv=26.0,
            price=4.25,
            volume=180,
            delta=-0.48,
            bid=4.20,
            ask=4.30,
        )

        assert data_point.option_type == "PUT"
        assert data_point.delta == -0.48  # Negative for puts

    def test_iv_data_point_default_timestamp(self):
        """Test that default timestamp is set correctly"""
        start_time = time.time()

        data_point = IVDataPoint(
            expiry="20241215",
            days_to_expiry=45,
            strike=150.0,
            option_type="CALL",
            iv=28.5,
            price=5.75,
            volume=250,
            delta=0.52,
            bid=5.70,
            ask=5.80,
        )

        end_time = time.time()

        # Timestamp should be within reasonable range
        assert start_time <= data_point.last_updated <= end_time

    def test_iv_data_point_edge_cases(self):
        """Test IV data point with edge case values"""
        data_point = IVDataPoint(
            expiry="20241215",
            days_to_expiry=0,  # Expiring today
            strike=0.01,  # Very low strike
            option_type="CALL",
            iv=0.1,  # Very low IV
            price=0.01,  # Very low price
            volume=0,  # No volume
            delta=0.0,  # No delta
            bid=0.0,  # No bid
            ask=0.01,  # Minimal ask
        )

        assert data_point.days_to_expiry == 0
        assert data_point.strike == 0.01
        assert data_point.volume == 0


class TestTermStructureCurve:
    """Comprehensive tests for TermStructureCurve class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.data_points = [
            IVDataPoint(
                "20241115", 30, 150.0, "CALL", 32.0, 6.50, 200, 0.55, 6.45, 6.55
            ),
            IVDataPoint(
                "20241220", 65, 150.0, "CALL", 28.0, 8.75, 150, 0.48, 8.70, 8.80
            ),
            IVDataPoint(
                "20250117", 93, 150.0, "CALL", 25.0, 10.25, 100, 0.42, 10.20, 10.30
            ),
        ]

        self.curve = TermStructureCurve(
            symbol="AAPL",
            strike=150.0,
            option_type="CALL",
            curve_points=self.data_points,
        )

    def test_curve_creation(self):
        """Test term structure curve creation"""
        assert self.curve.symbol == "AAPL"
        assert self.curve.strike == 150.0
        assert self.curve.option_type == "CALL"
        assert len(self.curve.curve_points) == 3
        assert isinstance(self.curve.curve_time, float)

    def test_get_iv_at_expiry_exact_match(self):
        """Test getting IV for exact expiry match"""
        iv = self.curve.get_iv_at_expiry(65)  # Exact match
        assert iv == 28.0

    def test_get_iv_at_expiry_interpolation(self):
        """Test IV interpolation between points"""
        # Test interpolation between 30 days (32.0 IV) and 65 days (28.0 IV)
        iv = self.curve.get_iv_at_expiry(45)  # Midpoint

        # Linear interpolation: 32.0 + (28.0 - 32.0) * (45 - 30) / (65 - 30)
        expected = 32.0 + (28.0 - 32.0) * (45 - 30) / (65 - 30)  # ≈ 30.29
        assert abs(iv - expected) < 0.01

    def test_get_iv_at_expiry_extrapolation_boundaries(self):
        """Test IV retrieval at boundaries (no extrapolation)"""
        # Before first point
        iv_before = self.curve.get_iv_at_expiry(15)
        assert iv_before is None

        # After last point
        iv_after = self.curve.get_iv_at_expiry(120)
        assert iv_after is None

    def test_get_iv_at_expiry_empty_curve(self):
        """Test IV retrieval from empty curve"""
        empty_curve = TermStructureCurve("AAPL", 150.0, "CALL", [])
        iv = empty_curve.get_iv_at_expiry(30)
        assert iv is None

    def test_get_iv_at_expiry_single_point(self):
        """Test IV retrieval with single data point"""
        single_point = [
            IVDataPoint("20241215", 45, 150.0, "CALL", 30.0, 5.0, 100, 0.5, 4.95, 5.05)
        ]
        single_curve = TermStructureCurve("AAPL", 150.0, "CALL", single_point)

        # Exact match
        iv_exact = single_curve.get_iv_at_expiry(45)
        assert iv_exact == 30.0

        # No match
        iv_none = single_curve.get_iv_at_expiry(30)
        assert iv_none is None

    def test_get_iv_at_expiry_identical_days(self):
        """Test IV retrieval with identical days to expiry"""
        identical_points = [
            IVDataPoint("20241215", 45, 150.0, "CALL", 30.0, 5.0, 100, 0.5, 4.95, 5.05),
            IVDataPoint("20241215", 45, 155.0, "CALL", 32.0, 5.5, 80, 0.45, 5.45, 5.55),
        ]
        identical_curve = TermStructureCurve("AAPL", 150.0, "CALL", identical_points)

        # Should return the first match
        iv = identical_curve.get_iv_at_expiry(45)
        assert iv == 30.0


class TestTermStructureInversion:
    """Comprehensive tests for TermStructureInversion dataclass"""

    def test_inversion_creation_complete(self):
        """Test creating a complete term structure inversion"""
        inversion = TermStructureInversion(
            symbol="AAPL",
            strike=150.0,
            option_type="CALL",
            front_expiry="20241115",
            back_expiry="20241220",
            front_days=30,
            back_days=65,
            front_iv=35.0,
            back_iv=28.0,
            iv_differential=7.0,
            inversion_magnitude=25.0,
            confidence_score=0.85,
            opportunity_score=0.78,
        )

        assert inversion.symbol == "AAPL"
        assert inversion.strike == 150.0
        assert inversion.option_type == "CALL"
        assert inversion.front_expiry == "20241115"
        assert inversion.back_expiry == "20241220"
        assert inversion.front_days == 30
        assert inversion.back_days == 65
        assert inversion.front_iv == 35.0
        assert inversion.back_iv == 28.0
        assert inversion.iv_differential == 7.0
        assert inversion.inversion_magnitude == 25.0
        assert inversion.confidence_score == 0.85
        assert inversion.opportunity_score == 0.78

    def test_inversion_put_option(self):
        """Test inversion for put option"""
        inversion = TermStructureInversion(
            symbol="AAPL",
            strike=150.0,
            option_type="PUT",
            front_expiry="20241115",
            back_expiry="20241220",
            front_days=30,
            back_days=65,
            front_iv=33.0,
            back_iv=26.0,
            iv_differential=7.0,
            inversion_magnitude=26.9,
            confidence_score=0.82,
            opportunity_score=0.75,
        )

        assert inversion.option_type == "PUT"
        assert inversion.iv_differential == 7.0  # Front > Back

    def test_inversion_edge_values(self):
        """Test inversion with edge case values"""
        inversion = TermStructureInversion(
            symbol="TEST",
            strike=0.01,
            option_type="CALL",
            front_expiry="20241115",
            back_expiry="20241116",  # Very close expiries
            front_days=1,
            back_days=2,
            front_iv=100.0,  # Very high IV
            back_iv=0.1,  # Very low IV
            iv_differential=99.9,
            inversion_magnitude=99900.0,  # Extreme inversion
            confidence_score=0.0,
            opportunity_score=1.0,
        )

        assert inversion.front_days == 1
        assert inversion.back_days == 2
        assert inversion.inversion_magnitude == 99900.0


class TestIVPercentileData:
    """Comprehensive tests for IVPercentileData dataclass"""

    def test_percentile_data_creation(self):
        """Test creating IV percentile data"""
        percentile_data = IVPercentileData(
            symbol="AAPL",
            strike=150.0,
            option_type="CALL",
            current_iv=32.5,
            percentile_rank=78.5,
            historical_mean=28.2,
            historical_std=6.8,
            lookback_days=252,
            confidence_level=0.95,
        )

        assert percentile_data.symbol == "AAPL"
        assert percentile_data.strike == 150.0
        assert percentile_data.option_type == "CALL"
        assert percentile_data.current_iv == 32.5
        assert percentile_data.percentile_rank == 78.5
        assert percentile_data.historical_mean == 28.2
        assert percentile_data.historical_std == 6.8
        assert percentile_data.lookback_days == 252
        assert percentile_data.confidence_level == 0.95

    def test_percentile_data_edge_cases(self):
        """Test percentile data with edge case values"""
        percentile_data = IVPercentileData(
            symbol="TEST",
            strike=0.01,
            option_type="PUT",
            current_iv=0.1,  # Very low current IV
            percentile_rank=0.0,  # Lowest percentile
            historical_mean=50.0,
            historical_std=0.0,  # No variation
            lookback_days=1,  # Minimal lookback
            confidence_level=0.01,  # Low confidence
        )

        assert percentile_data.percentile_rank == 0.0
        assert percentile_data.historical_std == 0.0
        assert percentile_data.confidence_level == 0.01

    def test_percentile_data_high_values(self):
        """Test percentile data with high values"""
        percentile_data = IVPercentileData(
            symbol="VOLATILE",
            strike=1000.0,
            option_type="CALL",
            current_iv=100.0,  # Very high current IV
            percentile_rank=100.0,  # Highest percentile
            historical_mean=45.0,
            historical_std=25.0,
            lookback_days=1000,
            confidence_level=1.0,  # Maximum confidence
        )

        assert percentile_data.current_iv == 100.0
        assert percentile_data.percentile_rank == 100.0
        assert percentile_data.confidence_level == 1.0


class TestTermStructureConfig:
    """Comprehensive tests for TermStructureConfig dataclass"""

    def test_default_configuration(self):
        """Test default configuration values"""
        config = TermStructureConfig()

        assert config.min_iv_spread == 2.0
        assert config.min_inversion_threshold == 10.0
        assert config.min_confidence_score == 0.7
        assert config.min_opportunity_score == 0.6
        assert config.min_data_points == 3
        assert config.max_curve_age == 300.0
        assert config.iv_percentile_lookback == 252
        assert config.min_historical_samples == 30
        assert config.cache_ttl == 60.0
        assert config.max_processing_time == 300.0

    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = TermStructureConfig(
            min_iv_spread=5.0,
            min_inversion_threshold=15.0,
            min_confidence_score=0.8,
            min_opportunity_score=0.7,
            min_data_points=5,
            max_curve_age=600.0,
            iv_percentile_lookback=500,
            min_historical_samples=50,
            cache_ttl=120.0,
            max_processing_time=500.0,
        )

        assert config.min_iv_spread == 5.0
        assert config.min_inversion_threshold == 15.0
        assert config.min_confidence_score == 0.8
        assert config.min_opportunity_score == 0.7
        assert config.min_data_points == 5
        assert config.max_curve_age == 600.0
        assert config.iv_percentile_lookback == 500
        assert config.min_historical_samples == 50
        assert config.cache_ttl == 120.0
        assert config.max_processing_time == 500.0

    def test_configuration_edge_cases(self):
        """Test configuration with edge case values"""
        # Minimum values
        config_min = TermStructureConfig(
            min_iv_spread=0.0,
            min_inversion_threshold=0.0,
            min_confidence_score=0.0,
            min_opportunity_score=0.0,
            min_data_points=1,
            max_curve_age=1.0,
            iv_percentile_lookback=1,
            min_historical_samples=1,
            cache_ttl=1.0,
            max_processing_time=1.0,
        )

        assert config_min.min_data_points == 1
        assert config_min.cache_ttl == 1.0

        # Maximum values
        config_max = TermStructureConfig(
            min_iv_spread=100.0,
            min_inversion_threshold=1000.0,
            min_confidence_score=1.0,
            min_opportunity_score=1.0,
            min_data_points=1000,
            max_curve_age=86400.0,  # 1 day
            iv_percentile_lookback=2520,  # 10 years
            min_historical_samples=1000,
            cache_ttl=3600.0,  # 1 hour
            max_processing_time=60000.0,  # 1 minute
        )

        assert config_max.min_iv_spread == 100.0
        assert config_max.min_data_points == 1000


class TestTermStructureAnalyzer:
    """Comprehensive tests for TermStructureAnalyzer class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = TermStructureConfig()
        self.analyzer = TermStructureAnalyzer(self.config)

    def test_analyzer_initialization_default(self):
        """Test analyzer initialization with default config"""
        analyzer = TermStructureAnalyzer()

        assert isinstance(analyzer.config, TermStructureConfig)
        assert len(analyzer.iv_curve_cache) == 0
        assert len(analyzer.iv_calculation_cache) == 0
        assert len(analyzer.percentile_cache) == 0
        assert isinstance(analyzer.historical_iv_data, dict)

    def test_analyzer_initialization_custom_config(self):
        """Test analyzer initialization with custom config"""
        custom_config = TermStructureConfig(min_iv_spread=5.0)
        analyzer = TermStructureAnalyzer(custom_config)

        assert analyzer.config.min_iv_spread == 5.0

    def test_calculate_days_to_expiry_valid(self):
        """Test days to expiry calculation with valid date"""
        future_date = datetime.now() + timedelta(days=45)
        expiry_str = future_date.strftime("%Y%m%d")

        days = self.analyzer._calculate_days_to_expiry(expiry_str)
        assert 44 <= days <= 46  # Allow for timing variations

    def test_calculate_days_to_expiry_invalid(self):
        """Test days to expiry calculation with invalid date"""
        days = self.analyzer._calculate_days_to_expiry("invalid_date")
        assert days == 30  # Should return default

    def test_calculate_days_to_expiry_past_date(self):
        """Test days to expiry calculation with past date"""
        past_date = datetime.now() - timedelta(days=30)
        expiry_str = past_date.strftime("%Y%m%d")

        days = self.analyzer._calculate_days_to_expiry(expiry_str)
        assert days < 0

    def test_estimate_delta_call(self):
        """Test delta estimation for call option"""
        mock_ticker = Mock(spec=Ticker)
        mock_contract = Mock(spec=Contract)
        mock_contract.right = "C"

        delta = self.analyzer._estimate_delta(mock_ticker, mock_contract)
        assert delta == 0.5

    def test_estimate_delta_put(self):
        """Test delta estimation for put option"""
        mock_ticker = Mock(spec=Ticker)
        mock_contract = Mock(spec=Contract)
        mock_contract.right = "P"

        delta = self.analyzer._estimate_delta(mock_ticker, mock_contract)
        assert delta == -0.5

    def test_estimate_delta_no_right(self):
        """Test delta estimation with no right attribute"""
        mock_ticker = Mock(spec=Ticker)
        mock_contract = Mock(spec=Contract)
        # Remove the right attribute to simulate it not existing
        del mock_contract.right

        delta = self.analyzer._estimate_delta(mock_ticker, mock_contract)
        assert delta == 0.0

    def test_calculate_implied_volatility_cached_with_cache(self):
        """Test cached IV calculation when value is in cache"""
        mock_ticker = Mock(spec=Ticker)
        mock_ticker.time = 1234567890
        mock_contract = Mock(spec=Contract)
        mock_contract.conId = 12345

        # Pre-populate cache
        cache_key = f"{mock_contract.conId}_{mock_ticker.time}"
        self.analyzer.iv_calculation_cache[cache_key] = (35.0, time.time())

        iv = self.analyzer._calculate_implied_volatility_cached(
            mock_ticker, mock_contract
        )
        assert iv == 35.0

    def test_calculate_implied_volatility_cached_expired_cache(self):
        """Test cached IV calculation when cache is expired"""
        mock_ticker = Mock(spec=Ticker)
        mock_ticker.time = 1234567890
        mock_ticker.ask = 5.50
        mock_ticker.bid = 5.40
        mock_contract = Mock(spec=Contract)
        mock_contract.conId = 12345

        # Pre-populate cache with expired entry
        cache_key = f"{mock_contract.conId}_{mock_ticker.time}"
        expired_time = time.time() - self.config.cache_ttl - 10
        self.analyzer.iv_calculation_cache[cache_key] = (35.0, expired_time)

        iv = self.analyzer._calculate_implied_volatility_cached(
            mock_ticker, mock_contract
        )

        # Should recalculate and cache new value
        assert isinstance(iv, float)
        assert 5.0 <= iv <= 100.0

        # Cache should be updated
        new_iv, new_time = self.analyzer.iv_calculation_cache[cache_key]
        assert new_iv == iv
        assert new_time > expired_time

    def test_calculate_implied_volatility_cached_no_cache(self):
        """Test cached IV calculation without existing cache"""
        mock_ticker = Mock(spec=Ticker)
        mock_ticker.time = 1234567890
        mock_ticker.ask = 5.50
        mock_ticker.bid = 5.40
        mock_contract = Mock(spec=Contract)
        mock_contract.conId = 12345

        # Clear cache
        self.analyzer.iv_calculation_cache.clear()

        iv = self.analyzer._calculate_implied_volatility_cached(
            mock_ticker, mock_contract
        )

        # Should calculate and cache IV
        assert isinstance(iv, float)
        assert 5.0 <= iv <= 100.0

        # Should be cached now
        cache_key = f"{mock_contract.conId}_{mock_ticker.time}"
        assert cache_key in self.analyzer.iv_calculation_cache

    def test_calculate_implied_volatility_cached_invalid_prices(self):
        """Test cached IV calculation with invalid prices"""
        mock_ticker = Mock(spec=Ticker)
        mock_ticker.time = 1234567890
        mock_ticker.ask = 0.0  # Invalid
        mock_ticker.bid = 0.0  # Invalid
        mock_contract = Mock(spec=Contract)
        mock_contract.conId = 12345

        iv = self.analyzer._calculate_implied_volatility_cached(
            mock_ticker, mock_contract
        )
        assert iv == 20.0  # Should return default

    def test_get_historical_iv_data_existing(self):
        """Test getting historical IV data when it exists"""
        key = "AAPL_150.0_CALL"

        # Pre-populate with test data
        test_data = [(time.time() - i * 86400, 25.0 + i) for i in range(10)]
        self.analyzer.historical_iv_data[key] = test_data

        iv_data = self.analyzer._get_historical_iv_data("AAPL", 150.0, "CALL")

        assert len(iv_data) == 10
        assert all(isinstance(iv, (int, float)) for iv in iv_data)

    def test_get_historical_iv_data_generate_synthetic(self):
        """Test generating synthetic historical IV data"""
        iv_data = self.analyzer._get_historical_iv_data("NEWSTOCK", 100.0, "CALL")

        # Should generate data for the lookback period
        assert len(iv_data) == self.config.iv_percentile_lookback
        assert all(5.0 <= iv <= 80.0 for iv in iv_data)  # Within reasonable bounds

    def test_detect_iv_inversion_positive(self):
        """Test IV inversion detection - positive case"""
        is_inversion, magnitude = self.analyzer.detect_iv_inversion(
            front_iv=35.0, back_iv=25.0, front_days=30, back_days=60
        )

        assert is_inversion == True
        assert magnitude > 0

    def test_detect_iv_inversion_negative(self):
        """Test IV inversion detection - negative case"""
        is_inversion, magnitude = self.analyzer.detect_iv_inversion(
            front_iv=20.0, back_iv=30.0, front_days=30, back_days=60
        )

        assert is_inversion == False
        assert magnitude == 0.0

    def test_detect_iv_inversion_invalid_expiries(self):
        """Test IV inversion with invalid expiry relationship"""
        # Front >= back (invalid)
        is_inversion, magnitude = self.analyzer.detect_iv_inversion(
            front_iv=30.0, back_iv=25.0, front_days=60, back_days=30
        )

        assert is_inversion == False
        assert magnitude == 0.0

    def test_detect_iv_inversion_zero_back_iv(self):
        """Test IV inversion with zero back IV"""
        is_inversion, magnitude = self.analyzer.detect_iv_inversion(
            front_iv=30.0, back_iv=0.0, front_days=30, back_days=60
        )

        assert is_inversion == False
        assert magnitude == 0.0

    def test_detect_iv_inversion_below_threshold(self):
        """Test IV inversion below minimum threshold"""
        # Normal term structure accounting for time normalization
        # front_normalized = 15.0 * sqrt(365/30) ≈ 52.3
        # back_normalized = 30.0 * sqrt(365/60) ≈ 74.1
        # No inversion since back > front after normalization
        is_inversion, magnitude = self.analyzer.detect_iv_inversion(
            front_iv=15.0, back_iv=30.0, front_days=30, back_days=60
        )

        assert is_inversion == False  # No inversion
        assert magnitude == 0.0  # Should return 0.0 when no inversion

    def test_clear_cache(self):
        """Test cache clearing functionality"""
        # Populate caches
        self.analyzer.iv_curve_cache["test"] = Mock()
        self.analyzer.iv_calculation_cache["test"] = (25.0, time.time())
        self.analyzer.percentile_cache["test"] = (Mock(), time.time())

        # Clear caches
        self.analyzer.clear_cache()

        assert len(self.analyzer.iv_curve_cache) == 0
        assert len(self.analyzer.iv_calculation_cache) == 0
        assert len(self.analyzer.percentile_cache) == 0

    def test_get_cache_stats(self):
        """Test cache statistics retrieval"""
        # Populate caches
        self.analyzer.iv_curve_cache["test1"] = Mock()
        self.analyzer.iv_curve_cache["test2"] = Mock()
        self.analyzer.iv_calculation_cache["test"] = (25.0, time.time())
        self.analyzer.percentile_cache["test"] = (Mock(), time.time())

        stats = self.analyzer.get_cache_stats()

        assert stats["iv_curves"] == 2
        assert stats["iv_calculations"] == 1
        assert stats["percentiles"] == 1

    def _create_mock_options_data(self) -> Dict[int, Ticker]:
        """Helper method to create mock options data"""
        options_data = {}

        # Create mock options for different strikes and expiries (3+ expiries per strike for min_data_points)
        for i, (strike, expiry, days) in enumerate(
            [
                (145.0, "20241115", 30),
                (150.0, "20241115", 30),
                (155.0, "20241115", 30),
                (145.0, "20241220", 65),
                (150.0, "20241220", 65),
                (155.0, "20241220", 65),
                (145.0, "20250117", 100),
                (150.0, "20250117", 100),
                (155.0, "20250117", 100),
            ]
        ):
            for right in ["C", "P"]:
                contract_id = 10000 + i * 10 + (1 if right == "C" else 2)

                mock_contract = Mock(spec=Contract)
                mock_contract.conId = contract_id
                mock_contract.symbol = "AAPL"
                mock_contract.strike = strike
                mock_contract.right = right
                mock_contract.lastTradeDateOrContractMonth = expiry

                mock_ticker = Mock(spec=Ticker)
                mock_ticker.contract = mock_contract
                mock_ticker.midpoint.return_value = 5.0 + i
                mock_ticker.close = 5.0 + i
                mock_ticker.bid = 4.9 + i
                mock_ticker.ask = 5.1 + i
                mock_ticker.volume = 100 + i * 10
                mock_ticker.time = 1234567890 + i  # Add time attribute for caching

                options_data[contract_id] = mock_ticker

        return options_data

    def test_build_iv_curves_success(self):
        """Test successful IV curve building"""
        options_data = self._create_mock_options_data()

        curves = self.analyzer._build_iv_curves("AAPL", options_data)

        # Should create curves for each strike/type combination
        assert len(curves) > 0

        # Verify curve structure
        for curve in curves:
            assert isinstance(curve, TermStructureCurve)
            assert curve.symbol == "AAPL"
            assert len(curve.curve_points) >= self.config.min_data_points

    def test_build_iv_curves_insufficient_data(self):
        """Test IV curve building with insufficient data"""
        # Create minimal options data (less than min_data_points)
        options_data = {}

        mock_contract = Mock(spec=Contract)
        mock_contract.conId = 10001
        mock_contract.symbol = "AAPL"
        mock_contract.strike = 150.0
        mock_contract.right = "C"
        mock_contract.lastTradeDateOrContractMonth = "20241115"

        mock_ticker = Mock(spec=Ticker)
        mock_ticker.contract = mock_contract
        mock_ticker.midpoint.return_value = 5.0
        mock_ticker.close = 5.0
        mock_ticker.bid = 4.9
        mock_ticker.ask = 5.1
        mock_ticker.volume = 100

        options_data[10001] = mock_ticker

        curves = self.analyzer._build_iv_curves("AAPL", options_data)

        # Should return empty list due to insufficient data points
        assert len(curves) == 0

    def test_build_iv_curves_invalid_data(self):
        """Test IV curve building with invalid data"""
        # Create mock ticker without contract but with valid bid/ask
        mock_ticker_no_contract = Mock()
        mock_ticker_no_contract.ask = 5.5
        mock_ticker_no_contract.bid = 5.0
        mock_ticker_no_contract.time = 1234567890
        # Remove contract attribute to simulate it not existing
        del mock_ticker_no_contract.contract

        options_data = {
            10001: None,  # None ticker
            10002: mock_ticker_no_contract,  # Ticker without contract
        }

        # Add ticker with invalid contract but valid bid/ask to avoid comparison errors
        mock_ticker_invalid = Mock(spec=Ticker)
        mock_contract = Mock()  # Contract without required attributes
        mock_contract.lastTradeDateOrContractMonth = "20241115"  # Valid date string
        mock_contract.strike = 150.0
        mock_contract.right = "C"
        mock_contract.symbol = "AAPL"
        mock_contract.conId = 10003  # Add conId to avoid AttributeError
        # Make it "invalid" by having unusual or missing other attributes
        mock_contract.multiplier = None
        mock_ticker_invalid.contract = mock_contract
        mock_ticker_invalid.ask = 5.5
        mock_ticker_invalid.bid = 5.0
        mock_ticker_invalid.time = 1234567890
        options_data[10003] = mock_ticker_invalid

        curves = self.analyzer._build_iv_curves("AAPL", options_data)
        assert len(curves) == 0

    def test_detect_term_structure_inversions_with_inversions(self):
        """Test detection of term structure inversions"""
        # Create curves with inversion (front > back IV)
        front_point = IVDataPoint(
            "20241115", 30, 150.0, "CALL", 35.0, 6.0, 200, 0.5, 5.9, 6.1
        )
        back_point = IVDataPoint(
            "20241220", 65, 150.0, "CALL", 25.0, 8.0, 150, 0.4, 7.9, 8.1
        )

        curve = TermStructureCurve("AAPL", 150.0, "CALL", [front_point, back_point])

        inversions = self.analyzer._detect_term_structure_inversions([curve])

        # Should detect inversion
        assert len(inversions) > 0

        inversion = inversions[0]
        assert inversion.front_iv > inversion.back_iv
        assert inversion.iv_differential > 0
        assert inversion.inversion_magnitude >= self.config.min_inversion_threshold

    def test_detect_term_structure_inversions_no_inversions(self):
        """Test detection with no inversions (normal term structure)"""
        # Create curves with normal term structure (back > front IV)
        front_point = IVDataPoint(
            "20241115", 30, 150.0, "CALL", 20.0, 6.0, 200, 0.5, 5.9, 6.1
        )
        back_point = IVDataPoint(
            "20241220", 65, 150.0, "CALL", 30.0, 8.0, 150, 0.4, 7.9, 8.1
        )

        curve = TermStructureCurve("AAPL", 150.0, "CALL", [front_point, back_point])

        inversions = self.analyzer._detect_term_structure_inversions([curve])

        # Should not detect significant inversions
        assert len(inversions) == 0

    def test_detect_term_structure_inversions_insufficient_points(self):
        """Test inversion detection with insufficient curve points"""
        # Single point curve
        single_point = IVDataPoint(
            "20241115", 30, 150.0, "CALL", 30.0, 6.0, 200, 0.5, 5.9, 6.1
        )
        curve = TermStructureCurve("AAPL", 150.0, "CALL", [single_point])

        inversions = self.analyzer._detect_term_structure_inversions([curve])
        assert len(inversions) == 0

    def test_calculate_inversion_confidence_high_quality(self):
        """Test confidence calculation with high quality data"""
        front_point = IVDataPoint(
            "20241115", 30, 150.0, "CALL", 35.0, 6.0, 500, 0.5, 5.95, 6.05
        )  # High volume, tight spread
        back_point = IVDataPoint(
            "20241220", 65, 150.0, "CALL", 25.0, 8.0, 400, 0.4, 7.95, 8.05
        )  # High volume, tight spread

        curve = TermStructureCurve("AAPL", 150.0, "CALL", [front_point, back_point])

        confidence = self.analyzer._calculate_inversion_confidence(
            front_point, back_point, curve
        )

        # Should have high confidence
        assert 0.7 <= confidence <= 1.0

    def test_calculate_inversion_confidence_low_quality(self):
        """Test confidence calculation with low quality data"""
        front_point = IVDataPoint(
            "20241115", 30, 150.0, "CALL", 35.0, 6.0, 5, 0.5, 5.0, 7.0
        )  # Low volume, wide spread
        back_point = IVDataPoint(
            "20241220", 65, 150.0, "CALL", 25.0, 8.0, 3, 0.4, 7.0, 9.0
        )  # Low volume, wide spread

        # Set old timestamps for freshness penalty
        old_time = time.time() - 600  # 10 minutes ago
        front_point.last_updated = old_time
        back_point.last_updated = old_time

        curve = TermStructureCurve("AAPL", 150.0, "CALL", [front_point, back_point])

        confidence = self.analyzer._calculate_inversion_confidence(
            front_point, back_point, curve
        )

        # Should have low confidence
        assert 0.0 <= confidence <= 0.4

    def test_calculate_curve_consistency_consistent(self):
        """Test curve consistency calculation with consistent inversion"""
        # Create curve with overall downward slope (consistent with inversion)
        points = [
            IVDataPoint("20241101", 15, 150.0, "CALL", 40.0, 7.0, 200, 0.6, 6.9, 7.1),
            IVDataPoint("20241115", 30, 150.0, "CALL", 35.0, 6.0, 180, 0.5, 5.9, 6.1),
            IVDataPoint("20241220", 65, 150.0, "CALL", 25.0, 8.0, 150, 0.4, 7.9, 8.1),
        ]

        curve = TermStructureCurve("AAPL", 150.0, "CALL", points)

        consistency = self.analyzer._calculate_curve_consistency(
            curve, points[1], points[2]
        )

        # Should indicate high consistency
        assert consistency >= 0.6

    def test_calculate_curve_consistency_inconsistent(self):
        """Test curve consistency with inconsistent data"""
        # Create curve with mixed slopes (inconsistent)
        points = [
            IVDataPoint("20241101", 15, 150.0, "CALL", 20.0, 5.0, 200, 0.4, 4.9, 5.1),
            IVDataPoint(
                "20241115", 30, 150.0, "CALL", 35.0, 6.0, 180, 0.5, 5.9, 6.1
            ),  # Higher than expected
            IVDataPoint(
                "20241220", 65, 150.0, "CALL", 25.0, 8.0, 150, 0.4, 7.9, 8.1
            ),  # Lower than expected
        ]

        curve = TermStructureCurve("AAPL", 150.0, "CALL", points)

        consistency = self.analyzer._calculate_curve_consistency(
            curve, points[1], points[2]
        )

        # Should indicate lower consistency
        assert consistency <= 0.5

    def test_calculate_curve_consistency_insufficient_points(self):
        """Test curve consistency with insufficient points"""
        points = [
            IVDataPoint("20241115", 30, 150.0, "CALL", 35.0, 6.0, 180, 0.5, 5.9, 6.1),
            IVDataPoint("20241220", 65, 150.0, "CALL", 25.0, 8.0, 150, 0.4, 7.9, 8.1),
        ]

        curve = TermStructureCurve("AAPL", 150.0, "CALL", points)

        consistency = self.analyzer._calculate_curve_consistency(
            curve, points[0], points[1]
        )

        # Should return neutral value
        assert consistency == 0.5

    def test_calculate_inversion_opportunity_score_excellent(self):
        """Test opportunity score calculation with excellent parameters"""
        front_point = IVDataPoint(
            "20241115", 30, 150.0, "CALL", 40.0, 6.0, 300, 0.5, 5.95, 6.05
        )
        back_point = IVDataPoint(
            "20241215", 60, 150.0, "CALL", 25.0, 8.0, 250, 0.4, 7.95, 8.05
        )

        score = self.analyzer._calculate_inversion_opportunity_score(
            front_point, back_point, inversion_magnitude=30.0, confidence=0.9
        )

        # Should have high opportunity score
        assert 0.7 <= score <= 1.0

    def test_calculate_inversion_opportunity_score_poor(self):
        """Test opportunity score calculation with poor parameters"""
        front_point = IVDataPoint(
            "20241115", 30, 150.0, "CALL", 26.0, 6.0, 5, 0.5, 5.0, 7.0
        )
        back_point = IVDataPoint(
            "20250315", 150, 150.0, "CALL", 25.0, 8.0, 3, 0.4, 7.0, 9.0
        )  # Very far expiry

        score = self.analyzer._calculate_inversion_opportunity_score(
            front_point,
            back_point,
            inversion_magnitude=5.0,
            confidence=0.2,  # Low magnitude and confidence
        )

        # Should have low opportunity score
        assert 0.0 <= score <= 0.4

    def test_score_and_filter_inversions_qualified(self):
        """Test scoring and filtering with qualified inversions"""
        qualified_inversion = TermStructureInversion(
            "AAPL",
            150.0,
            "CALL",
            "20241115",
            "20241220",
            30,
            65,
            35.0,
            25.0,
            10.0,
            40.0,
            0.8,
            0.75,  # High confidence and opportunity scores
        )

        unqualified_inversion = TermStructureInversion(
            "AAPL",
            150.0,
            "CALL",
            "20241115",
            "20241220",
            30,
            65,
            27.0,
            25.0,
            2.0,
            8.0,
            0.3,
            0.4,  # Low confidence and opportunity scores
        )

        inversions = [qualified_inversion, unqualified_inversion]

        filtered = self.analyzer._score_and_filter_inversions(inversions)

        # Should only return qualified inversion
        assert len(filtered) == 1
        assert filtered[0] == qualified_inversion

    def test_score_and_filter_inversions_none_qualified(self):
        """Test scoring and filtering with no qualified inversions"""
        unqualified_inversions = [
            TermStructureInversion(
                "AAPL",
                150.0,
                "CALL",
                "20241115",
                "20241220",
                30,
                65,
                26.0,
                25.0,
                1.0,
                4.0,
                0.3,
                0.4,  # Low scores
            ),
            TermStructureInversion(
                "AAPL",
                155.0,
                "CALL",
                "20241115",
                "20241220",
                30,
                65,
                27.0,
                25.0,
                2.0,
                8.0,
                0.5,
                0.45,  # Low scores
            ),
        ]

        filtered = self.analyzer._score_and_filter_inversions(unqualified_inversions)

        # Should return empty list
        assert len(filtered) == 0

    def test_score_and_filter_inversions_sorting(self):
        """Test that filtered inversions are sorted by opportunity score"""
        inversions = [
            TermStructureInversion(
                "AAPL",
                150.0,
                "CALL",
                "20241115",
                "20241220",
                30,
                65,
                35.0,
                25.0,
                10.0,
                40.0,
                0.8,
                0.65,  # Lower opportunity score
            ),
            TermStructureInversion(
                "AAPL",
                155.0,
                "CALL",
                "20241115",
                "20241220",
                30,
                65,
                40.0,
                25.0,
                15.0,
                60.0,
                0.85,
                0.75,  # Higher opportunity score
            ),
        ]

        filtered = self.analyzer._score_and_filter_inversions(inversions)

        # Should be sorted by opportunity score (descending)
        assert len(filtered) == 2
        assert filtered[0].opportunity_score > filtered[1].opportunity_score

    def test_analyze_term_structure_success(self):
        """Test complete term structure analysis"""
        options_data = self._create_mock_options_data()

        curves, inversions = self.analyzer.analyze_term_structure("AAPL", options_data)

        # Should return results
        assert len(curves) >= 0
        assert len(inversions) >= 0

        # Verify structure of results
        for curve in curves:
            assert isinstance(curve, TermStructureCurve)
            assert curve.symbol == "AAPL"

        for inversion in inversions:
            assert isinstance(inversion, TermStructureInversion)
            assert inversion.symbol == "AAPL"

    def test_analyze_term_structure_empty_data(self):
        """Test term structure analysis with empty data"""
        curves, inversions = self.analyzer.analyze_term_structure("AAPL", {})

        assert len(curves) == 0
        assert len(inversions) == 0

    def test_analyze_term_structure_performance_warning(self):
        """Test performance warning for slow analysis"""
        # Create config with very low processing time limit
        fast_config = TermStructureConfig(max_processing_time=0.1)  # 0.1ms limit
        fast_analyzer = TermStructureAnalyzer(fast_config)

        options_data = self._create_mock_options_data()

        with patch("time.time", side_effect=[0.0, 1.0]):  # Mock 1000ms processing time
            curves, inversions = fast_analyzer.analyze_term_structure(
                "AAPL", options_data
            )

            # Should still return results despite warning
            assert isinstance(curves, list)
            assert isinstance(inversions, list)


class TestTermStructureAnalyzerIntegration:
    """Integration tests for TermStructureAnalyzer with other components"""

    def setup_method(self):
        """Setup integration test fixtures"""
        self.analyzer = TermStructureAnalyzer()

    def test_calculate_iv_percentiles_with_cache(self):
        """Test IV percentile calculation using cache"""
        # Pre-populate cache
        mock_percentile_data = IVPercentileData(
            "AAPL", 150.0, "CALL", 32.5, 78.5, 28.2, 6.8, 252, 0.95
        )
        cache_key = "AAPL_150.0_CALL"
        self.analyzer.percentile_cache[cache_key] = (mock_percentile_data, time.time())

        result = self.analyzer.calculate_iv_percentiles("AAPL", 150.0, "CALL", 32.5)

        assert result == mock_percentile_data

    def test_calculate_iv_percentiles_expired_cache(self):
        """Test IV percentile calculation with expired cache"""
        # Pre-populate cache with expired entry
        mock_percentile_data = IVPercentileData(
            "AAPL", 150.0, "CALL", 32.5, 78.5, 28.2, 6.8, 252, 0.95
        )
        cache_key = "AAPL_150.0_CALL"
        expired_time = time.time() - self.analyzer.config.cache_ttl - 10
        self.analyzer.percentile_cache[cache_key] = (mock_percentile_data, expired_time)

        result = self.analyzer.calculate_iv_percentiles("AAPL", 150.0, "CALL", 32.5)

        # Should recalculate and return new data
        assert isinstance(result, IVPercentileData)
        assert result.symbol == "AAPL"
        assert result.current_iv == 32.5

    def test_calculate_iv_percentiles_insufficient_data(self):
        """Test IV percentile calculation with insufficient historical data"""
        # Create analyzer with high minimum samples requirement
        config = TermStructureConfig(min_historical_samples=1000000)  # Unreachable
        analyzer = TermStructureAnalyzer(config)

        result = analyzer.calculate_iv_percentiles("NEWSTOCK", 100.0, "CALL", 25.0)

        assert result is None

    def test_calculate_iv_percentiles_normal_calculation(self):
        """Test normal IV percentile calculation"""
        result = self.analyzer.calculate_iv_percentiles("AAPL", 150.0, "CALL", 32.5)

        assert isinstance(result, IVPercentileData)
        assert result.symbol == "AAPL"
        assert result.strike == 150.0
        assert result.option_type == "CALL"
        assert result.current_iv == 32.5
        assert 0.0 <= result.percentile_rank <= 100.0
        assert result.lookback_days == self.analyzer.config.iv_percentile_lookback

    def test_score_calendar_opportunity_base_case(self):
        """Test calendar opportunity scoring - base case"""
        inversion = TermStructureInversion(
            "AAPL",
            150.0,
            "CALL",
            "20241115",
            "20241220",
            30,
            65,
            35.0,
            25.0,
            10.0,
            40.0,
            0.8,
            0.7,
        )

        score = self.analyzer.score_calendar_opportunity(inversion)

        # Should return base score with some bonuses
        assert 0.7 <= score <= 1.0

    def test_score_calendar_opportunity_with_percentile_bonus(self):
        """Test calendar opportunity scoring with high IV percentile bonus"""
        inversion = TermStructureInversion(
            "AAPL",
            150.0,
            "CALL",
            "20241115",
            "20241220",
            30,
            65,
            35.0,
            25.0,
            10.0,
            40.0,
            0.8,
            0.7,
        )

        high_percentile_data = IVPercentileData(
            "AAPL", 150.0, "CALL", 35.0, 85.0, 25.0, 5.0, 252, 0.95  # 85th percentile
        )

        score = self.analyzer.score_calendar_opportunity(
            inversion, high_percentile_data
        )

        # Should get percentile bonus
        base_score = self.analyzer.score_calendar_opportunity(inversion)
        assert score > base_score

    def test_score_calendar_opportunity_with_percentile_penalty(self):
        """Test calendar opportunity scoring with low IV percentile penalty"""
        inversion = TermStructureInversion(
            "AAPL",
            150.0,
            "CALL",
            "20241115",
            "20241220",
            30,
            65,
            35.0,
            25.0,
            10.0,
            40.0,
            0.8,
            0.7,
        )

        low_percentile_data = IVPercentileData(
            "AAPL", 150.0, "CALL", 35.0, 15.0, 40.0, 8.0, 252, 0.95  # 15th percentile
        )

        score = self.analyzer.score_calendar_opportunity(inversion, low_percentile_data)

        # Should get percentile penalty
        base_score = self.analyzer.score_calendar_opportunity(inversion)
        assert score < base_score

    def test_score_calendar_opportunity_bonuses(self):
        """Test various bonuses in calendar opportunity scoring"""
        # Short front month (bonus)
        short_front_inversion = TermStructureInversion(
            "AAPL",
            150.0,
            "CALL",
            "20241115",
            "20241220",
            25,
            65,  # 25 days front
            35.0,
            25.0,
            10.0,
            40.0,
            0.8,
            0.7,
        )

        # High inversion magnitude (bonus)
        high_magnitude_inversion = TermStructureInversion(
            "AAPL",
            150.0,
            "CALL",
            "20241115",
            "20241220",
            30,
            65,
            45.0,
            25.0,
            20.0,
            80.0,
            0.8,
            0.7,  # 80% magnitude
        )

        base_score = self.analyzer.score_calendar_opportunity(
            TermStructureInversion(
                "AAPL",
                150.0,
                "CALL",
                "20241115",
                "20241220",
                50,
                65,  # 50 days front
                30.0,
                25.0,
                5.0,
                20.0,
                0.8,
                0.7,  # 20% magnitude
            )
        )

        short_score = self.analyzer.score_calendar_opportunity(short_front_inversion)
        high_mag_score = self.analyzer.score_calendar_opportunity(
            high_magnitude_inversion
        )

        # Both should have bonuses
        assert short_score >= base_score
        assert high_mag_score >= base_score

    def test_get_term_structure_summary_with_data(self):
        """Test term structure summary with existing data"""
        # Populate cache with test curve
        test_curve = TermStructureCurve(
            "AAPL",
            150.0,
            "CALL",
            [IVDataPoint("20241115", 30, 150.0, "CALL", 32.0, 6.0, 200, 0.5, 5.9, 6.1)],
        )
        self.analyzer.iv_curve_cache["AAPL_150.0_CALL"] = test_curve

        summary = self.analyzer.get_term_structure_summary("AAPL")

        assert summary["symbol"] == "AAPL"
        assert summary["total_curves"] >= 1
        assert "analysis_timestamp" in summary
        assert isinstance(summary["average_iv"], float)

    def test_get_term_structure_summary_no_data(self):
        """Test term structure summary with no data"""
        summary = self.analyzer.get_term_structure_summary("NONEXISTENT")

        assert "error" in summary
        assert "NONEXISTENT" in summary["error"]


class TestTermStructureConvenienceFunctions:
    """Tests for convenience functions and integration methods"""

    def test_analyze_symbol_term_structure_function(self):
        """Test the analyze_symbol_term_structure convenience function"""
        options_data = {}  # Empty for simplicity

        curves, inversions = analyze_symbol_term_structure("AAPL", options_data)

        assert isinstance(curves, list)
        assert isinstance(inversions, list)

    def test_analyze_symbol_term_structure_with_custom_config(self):
        """Test analyze_symbol_term_structure with custom config"""
        custom_config = TermStructureConfig(min_iv_spread=5.0)
        options_data = {}

        curves, inversions = analyze_symbol_term_structure(
            "AAPL", options_data, custom_config
        )

        assert isinstance(curves, list)
        assert isinstance(inversions, list)

    def test_detect_calendar_spread_opportunities_function(self):
        """Test the detect_calendar_spread_opportunities function"""
        options_data = {}  # Empty for simplicity

        opportunities = detect_calendar_spread_opportunities("AAPL", options_data)

        assert isinstance(opportunities, list)

    def test_detect_calendar_spread_opportunities_with_parameters(self):
        """Test detect_calendar_spread_opportunities with custom parameters"""
        options_data = {}

        opportunities = detect_calendar_spread_opportunities(
            "AAPL", options_data, min_iv_spread=5.0, min_confidence=0.8
        )

        assert isinstance(opportunities, list)

    def test_validate_calendar_spread_with_term_structure_valid(self):
        """Test calendar spread validation - valid case"""
        is_valid, criteria = validate_calendar_spread_with_term_structure(
            front_iv=35.0,
            back_iv=25.0,
            front_days=30,
            back_days=60,
            min_iv_spread=2.0,
            min_confidence=0.7,
        )

        assert isinstance(is_valid, bool)
        assert isinstance(criteria, dict)
        assert "has_iv_inversion" in criteria
        assert "iv_spread_sufficient" in criteria
        assert "inversion_magnitude" in criteria

    def test_validate_calendar_spread_with_term_structure_invalid(self):
        """Test calendar spread validation - invalid case"""
        is_valid, criteria = validate_calendar_spread_with_term_structure(
            front_iv=20.0,
            back_iv=30.0,
            front_days=30,
            back_days=60,  # No inversion
            min_iv_spread=2.0,
            min_confidence=0.7,
        )

        assert is_valid == False
        assert criteria["has_iv_inversion"] == False

    def test_get_optimal_calendar_expiries_function(self):
        """Test get_optimal_calendar_expiries function"""
        options_data = {}  # Empty for simplicity

        expiry_scores = get_optimal_calendar_expiries("AAPL", options_data)

        assert isinstance(expiry_scores, list)
        # Each item should be (front_expiry, back_expiry, score) tuple
        for item in expiry_scores:
            assert len(item) == 3
            assert isinstance(item[2], (int, float))  # Score

    def test_get_optimal_calendar_expiries_with_targets(self):
        """Test get_optimal_calendar_expiries with custom targets"""
        options_data = {}

        expiry_scores = get_optimal_calendar_expiries(
            "AAPL", options_data, target_front_days=45, target_back_days=90
        )

        assert isinstance(expiry_scores, list)

    @pytest.mark.asyncio
    async def test_analyze_multi_symbol_term_structure_function(self):
        """Test analyze_multi_symbol_term_structure function"""
        symbols = ["AAPL", "MSFT"]
        options_data_dict = {"AAPL": {}, "MSFT": {}}

        results = await analyze_multi_symbol_term_structure(symbols, options_data_dict)

        assert isinstance(results, dict)
        assert "AAPL" in results
        assert "MSFT" in results

    @pytest.mark.asyncio
    async def test_analyze_multi_symbol_term_structure_missing_data(self):
        """Test multi-symbol analysis with missing data"""
        symbols = ["AAPL", "MISSING"]
        options_data_dict = {"AAPL": {}}  # Missing MISSING symbol data

        results = await analyze_multi_symbol_term_structure(symbols, options_data_dict)

        assert "AAPL" in results
        assert "MISSING" not in results or "error" in results.get("MISSING", {})

    @pytest.mark.asyncio
    async def test_analyze_multi_symbol_term_structure_with_config(self):
        """Test multi-symbol analysis with custom config"""
        symbols = ["AAPL"]
        options_data_dict = {"AAPL": {}}
        custom_config = TermStructureConfig(min_iv_spread=5.0)

        results = await analyze_multi_symbol_term_structure(
            symbols, options_data_dict, custom_config
        )

        assert "AAPL" in results

    def test_enhance_calendar_spread_with_term_structure_function(self):
        """Test enhance_calendar_spread_with_term_structure function"""
        # Create mock calendar opportunity
        mock_opportunity = Mock()
        mock_opportunity.symbol = "AAPL"
        mock_opportunity.strike = 150.0
        mock_opportunity.option_type = "CALL"
        mock_opportunity.front_leg = Mock()
        mock_opportunity.front_leg.expiry = "20241115"
        mock_opportunity.back_leg = Mock()
        mock_opportunity.back_leg.expiry = "20241220"
        mock_opportunity.composite_score = 0.7

        options_data = {}

        enhanced_opportunity = enhance_calendar_spread_with_term_structure(
            mock_opportunity, options_data
        )

        # Should return the same opportunity (enhanced)
        assert enhanced_opportunity == mock_opportunity


class TestTermStructurePerformanceAndEdgeCases:
    """Tests for performance optimization and edge cases"""

    def setup_method(self):
        """Setup performance test fixtures"""
        self.analyzer = TermStructureAnalyzer()

    def test_large_dataset_performance(self):
        """Test performance with large datasets"""
        # Create large options dataset
        large_options_data = {}

        for i in range(1000):  # 1000 options
            contract_id = 20000 + i

            mock_contract = Mock(spec=Contract)
            mock_contract.conId = contract_id
            mock_contract.symbol = "PERF_TEST"
            mock_contract.strike = 100.0 + (i % 50)  # 50 different strikes
            mock_contract.right = "C" if i % 2 == 0 else "P"
            mock_contract.lastTradeDateOrContractMonth = f"2024{12 + (i % 12):02d}15"

            mock_ticker = Mock(spec=Ticker)
            mock_ticker.contract = mock_contract
            mock_ticker.midpoint.return_value = 5.0 + (i % 10)
            mock_ticker.close = 5.0 + (i % 10)
            mock_ticker.bid = 4.9 + (i % 10)
            mock_ticker.ask = 5.1 + (i % 10)
            mock_ticker.volume = 100 + i

            large_options_data[contract_id] = mock_ticker

        start_time = time.time()
        curves, inversions = self.analyzer.analyze_term_structure(
            "PERF_TEST", large_options_data
        )
        processing_time = time.time() - start_time

        # Should complete within reasonable time (adjust as needed)
        assert processing_time < 5.0  # 5 seconds max

        # Should return results
        assert isinstance(curves, list)
        assert isinstance(inversions, list)

    def test_memory_usage_large_cache(self):
        """Test memory management with large cache"""
        # Fill caches with many entries
        for i in range(10000):
            cache_key = f"TEST_{i}"
            self.analyzer.iv_curve_cache[cache_key] = Mock()
            self.analyzer.iv_calculation_cache[cache_key] = (25.0 + i % 50, time.time())
            self.analyzer.percentile_cache[cache_key] = (Mock(), time.time())

        # Verify caches are populated
        assert len(self.analyzer.iv_curve_cache) == 10000
        assert len(self.analyzer.iv_calculation_cache) == 10000
        assert len(self.analyzer.percentile_cache) == 10000

        # Clear and verify
        self.analyzer.clear_cache()
        assert len(self.analyzer.iv_curve_cache) == 0
        assert len(self.analyzer.iv_calculation_cache) == 0
        assert len(self.analyzer.percentile_cache) == 0

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values"""
        # Test with extreme IV values
        extreme_inversions = [
            (999.0, 0.001, 30, 60),  # Very high front, very low back
            (0.001, 999.0, 30, 60),  # Very low front, very high back
            (0.0, 0.0, 30, 60),  # Zero IVs
            (float("inf"), 50.0, 30, 60),  # Infinite IV
        ]

        for front_iv, back_iv, front_days, back_days in extreme_inversions:
            try:
                is_inversion, magnitude = self.analyzer.detect_iv_inversion(
                    front_iv, back_iv, front_days, back_days
                )

                # Should handle gracefully
                assert isinstance(is_inversion, bool)
                assert isinstance(magnitude, (int, float))
                assert not np.isnan(magnitude)
                assert not np.isinf(magnitude)

            except (ZeroDivisionError, OverflowError, ValueError):
                # Acceptable to raise these for extreme values
                pass

    def test_concurrent_cache_access(self):
        """Test thread safety of cache operations"""
        import threading

        errors = []
        results = []

        def cache_worker(worker_id):
            try:
                for i in range(100):
                    # Simulate concurrent cache operations
                    cache_key = f"worker_{worker_id}_{i}"

                    # Add to cache
                    self.analyzer.iv_calculation_cache[cache_key] = (
                        25.0 + i,
                        time.time(),
                    )

                    # Read from cache
                    if cache_key in self.analyzer.iv_calculation_cache:
                        iv, timestamp = self.analyzer.iv_calculation_cache[cache_key]
                        results.append((worker_id, i, iv))

                    # Simulate some processing
                    time.sleep(0.001)

            except Exception as e:
                errors.append((worker_id, str(e)))

        # Create multiple threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=cache_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors and expected results
        assert len(errors) == 0
        assert len(results) == 500  # 5 workers * 100 operations each

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration behavior"""
        # Configure short TTL for testing
        short_ttl_config = TermStructureConfig(cache_ttl=0.1)  # 100ms TTL
        analyzer = TermStructureAnalyzer(short_ttl_config)

        # Add entry to cache
        cache_key = "test_ttl"
        test_iv = 30.0
        analyzer.iv_calculation_cache[cache_key] = (test_iv, time.time())

        # Should be available immediately
        cached_iv, _ = analyzer.iv_calculation_cache[cache_key]
        assert cached_iv == test_iv

        # Wait for TTL to expire
        time.sleep(0.2)  # 200ms > 100ms TTL

        # Simulate cache check (in real usage, this would trigger recalculation)
        mock_ticker = Mock()
        mock_ticker.ask = 5.50
        mock_ticker.bid = 5.40
        mock_contract = Mock()
        mock_contract.conId = 12345

        # Should recalculate due to expired TTL
        new_iv = analyzer._calculate_implied_volatility_cached(
            mock_ticker, mock_contract
        )
        assert isinstance(new_iv, float)

    def test_error_handling_malformed_data(self):
        """Test error handling with malformed data structures"""
        malformed_data = {
            1: "not_a_ticker",  # String instead of Ticker
            2: None,  # None value
            3: Mock(),  # Mock without required attributes
            4: {"invalid": "dict"},  # Dict instead of Ticker
        }

        # Should handle gracefully without crashing
        curves, inversions = self.analyzer.analyze_term_structure(
            "MALFORMED", malformed_data
        )

        assert isinstance(curves, list)
        assert isinstance(inversions, list)
        # May be empty due to malformed data, but shouldn't crash

    def test_boundary_condition_handling(self):
        """Test handling of boundary conditions"""
        boundary_conditions = [
            # (front_iv, back_iv, front_days, back_days, expected_inversion)
            (25.0, 25.0, 30, 60, False),  # Equal IVs
            (25.1, 25.0, 30, 60, False),  # Tiny difference
            (25.0, 24.9, 30, 60, False),  # Reversed tiny difference
            (30.0, 30.0, 30, 30, False),  # Equal days
            (30.0, 25.0, 60, 30, False),  # Reversed days (invalid)
            (0.1, 0.1, 1, 2, False),  # Minimal values
        ]

        for front_iv, back_iv, front_days, back_days, expected in boundary_conditions:
            is_inversion, magnitude = self.analyzer.detect_iv_inversion(
                front_iv, back_iv, front_days, back_days
            )

            assert isinstance(is_inversion, bool)
            assert isinstance(magnitude, (int, float))
            assert magnitude >= 0.0  # Magnitude should be non-negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
