"""
Configuration loader for SFR backtesting engine.

This module provides functionality to load and validate configuration from YAML files,
environment variables, and programmatic settings.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging
import yaml

from backtesting.sfr_backtest_engine import SFRBacktestConfig, SlippageModel, VixRegime

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""

    pass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    host: str = "localhost"
    port: int = 5433
    database: str = "options_arbitrage"
    user: str = "trading_user"
    password: str = "secure_trading_password"
    min_pool_size: int = 5
    max_pool_size: int = 20

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class SFRConfigLoader:
    """
    Configuration loader for SFR backtesting engine.

    Supports loading from:
    1. YAML configuration files
    2. Environment variables
    3. Programmatic configuration
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path or self._find_default_config()
        self.config_data: Dict[str, Any] = {}
        self._load_config()

    def _find_default_config(self) -> str:
        """Find default configuration file."""
        # Look in multiple locations
        possible_paths = [
            "backtesting/config/sfr_backtest_config.yaml",
            "config/sfr_backtest_config.yaml",
            "sfr_backtest_config.yaml",
            os.path.join(os.path.dirname(__file__), "sfr_backtest_config.yaml"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found configuration file at: {path}")
                return path

        # If no config found, return default path (will be created if needed)
        return os.path.join(os.path.dirname(__file__), "sfr_backtest_config.yaml")

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    self.config_data = yaml.safe_load(f)
                logger.info(f"Configuration loaded from: {self.config_path}")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self.config_data = {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise ConfigurationError(
                f"Failed to load config from {self.config_path}: {e}"
            )

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration with environment variable overrides."""
        db_config = self.config_data.get("database", {})

        return DatabaseConfig(
            host=os.getenv("DB_HOST", db_config.get("host", "localhost")),
            port=int(os.getenv("DB_PORT", db_config.get("port", 5433))),
            database=os.getenv(
                "DB_NAME", db_config.get("database", "options_arbitrage")
            ),
            user=os.getenv("DB_USER", db_config.get("user", "trading_user")),
            password=os.getenv(
                "DB_PASSWORD", db_config.get("password", "secure_trading_password")
            ),
            min_pool_size=db_config.get("min_pool_size", 5),
            max_pool_size=db_config.get("max_pool_size", 20),
        )

    def get_target_symbols(self) -> List[str]:
        """Get target symbols for backtesting."""
        symbols = self.config_data.get(
            "target_symbols",
            [
                "SPY",
                "QQQ",
                "AAPL",
                "MSFT",
                "NVDA",
                "TSLA",
                "AMZN",
                "META",
                "GOOGL",
                "JPM",
            ],
        )

        # Allow environment variable override
        env_symbols = os.getenv("BACKTEST_SYMBOLS")
        if env_symbols:
            symbols = env_symbols.split(",")

        return symbols

    def get_periods(self) -> Dict[str, int]:
        """Get predefined backtesting periods."""
        return self.config_data.get(
            "periods",
            {
                "quick_test": 30,
                "short_term": 90,
                "medium_term": 365,
                "long_term": 1825,
                "full_cycle": 3650,
            },
        )

    def get_sfr_config(self, config_name: str = "balanced") -> SFRBacktestConfig:
        """
        Get SFR backtesting configuration by name.

        Args:
            config_name: Name of configuration preset

        Returns:
            SFRBacktestConfig instance
        """
        configurations = self.config_data.get("configurations", {})

        if config_name not in configurations:
            available = list(configurations.keys())
            raise ConfigurationError(
                f"Configuration '{config_name}' not found. Available: {available}"
            )

        config_dict = configurations[config_name]

        try:
            # Parse slippage model
            slippage_model_str = config_dict.get("slippage_model", "LINEAR")
            slippage_model = SlippageModel(slippage_model_str)

            # Parse VIX regime filter
            vix_regime_filter = None
            if config_dict.get("vix_regime_filter"):
                vix_regime_filter = VixRegime(config_dict["vix_regime_filter"])

            # Create configuration with environment variable overrides
            return SFRBacktestConfig(
                profit_target=float(
                    os.getenv("PROFIT_TARGET", config_dict.get("profit_target", 0.50))
                ),
                cost_limit=float(
                    os.getenv("COST_LIMIT", config_dict.get("cost_limit", 120.0))
                ),
                volume_limit=int(
                    os.getenv("VOLUME_LIMIT", config_dict.get("volume_limit", 100))
                ),
                quantity=int(os.getenv("QUANTITY", config_dict.get("quantity", 1))),
                call_strike_range_days=config_dict.get("call_strike_range_days", 25),
                put_strike_range_days=config_dict.get("put_strike_range_days", 25),
                expiry_min_days=config_dict.get("expiry_min_days", 19),
                expiry_max_days=config_dict.get("expiry_max_days", 45),
                max_strike_combinations=config_dict.get("max_strike_combinations", 4),
                max_expiry_options=config_dict.get("max_expiry_options", 8),
                max_bid_ask_spread_call=config_dict.get(
                    "max_bid_ask_spread_call", 20.0
                ),
                max_bid_ask_spread_put=config_dict.get("max_bid_ask_spread_put", 20.0),
                combo_buffer_percent=config_dict.get("combo_buffer_percent", 0.00),
                data_timeout_seconds=config_dict.get("data_timeout_seconds", 45),
                slippage_model=slippage_model,
                base_slippage_bps=config_dict.get("base_slippage_bps", 2),
                liquidity_penalty_factor=config_dict.get(
                    "liquidity_penalty_factor", 1.0
                ),
                commission_per_contract=config_dict.get(
                    "commission_per_contract", 1.00
                ),
                vix_regime_filter=vix_regime_filter,
                min_vix_level=config_dict.get("min_vix_level"),
                max_vix_level=config_dict.get("max_vix_level"),
                exclude_vix_spikes=config_dict.get("exclude_vix_spikes", False),
            )

        except (ValueError, KeyError) as e:
            raise ConfigurationError(f"Invalid configuration '{config_name}': {e}")

    def get_available_configurations(self) -> List[str]:
        """Get list of available configuration names."""
        return list(self.config_data.get("configurations", {}).keys())

    def get_slippage_model_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about available slippage models."""
        return self.config_data.get("slippage_models", {})

    def get_vix_regime_info(self) -> Dict[str, Dict[str, str]]:
        """Get information about VIX regimes."""
        return self.config_data.get("vix_regimes", {})

    def get_benchmarks(self) -> Dict[str, Dict[str, str]]:
        """Get performance benchmarks."""
        return self.config_data.get("benchmarks", {})

    def get_analysis_template(self, template_name: str) -> Dict[str, Any]:
        """
        Get analysis template configuration.

        Args:
            template_name: Name of analysis template

        Returns:
            Template configuration dictionary
        """
        templates = self.config_data.get("analysis_templates", {})

        if template_name not in templates:
            available = list(templates.keys())
            raise ConfigurationError(
                f"Template '{template_name}' not found. Available: {available}"
            )

        return templates[template_name]

    def get_available_templates(self) -> List[str]:
        """Get list of available analysis templates."""
        return list(self.config_data.get("analysis_templates", {}).keys())

    def validate_configuration(self, config: SFRBacktestConfig) -> List[str]:
        """
        Validate SFR configuration against rules.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        validation_rules = self.config_data.get("validation", {})

        # Profit target validation
        if config.profit_target <= 0:
            errors.append("Profit target must be positive")
        if config.profit_target > 10:
            errors.append("Profit target appears too high (>10%)")

        # Cost limit validation
        if config.cost_limit <= 0:
            errors.append("Cost limit must be positive")
        if config.cost_limit > validation_rules.get("max_portfolio_exposure", 100000):
            errors.append("Cost limit exceeds maximum exposure")

        # Volume validation
        if config.volume_limit < 0:
            errors.append("Volume limit cannot be negative")

        # Quantity validation
        if config.quantity <= 0:
            errors.append("Quantity must be positive")

        # Expiry range validation
        if config.expiry_min_days >= config.expiry_max_days:
            errors.append("Minimum expiry days must be less than maximum")
        if config.expiry_min_days < validation_rules.get("min_days_to_expiry", 1):
            errors.append("Expiry minimum days too low")
        if config.expiry_max_days > validation_rules.get("max_days_to_expiry", 365):
            errors.append("Expiry maximum days too high")

        # Spread validation
        if config.max_bid_ask_spread_call <= 0 or config.max_bid_ask_spread_put <= 0:
            errors.append("Bid-ask spread limits must be positive")

        # Slippage validation
        if config.base_slippage_bps < 0:
            errors.append("Base slippage cannot be negative")
        if config.base_slippage_bps > 100:  # 1%
            errors.append("Base slippage appears too high (>100bps)")

        # Commission validation
        if config.commission_per_contract < 0:
            errors.append("Commission cannot be negative")
        if config.commission_per_contract > 10:
            errors.append("Commission appears too high (>$10)")

        # VIX level validation
        if config.min_vix_level is not None and config.max_vix_level is not None:
            if config.min_vix_level >= config.max_vix_level:
                errors.append("Minimum VIX level must be less than maximum")

        return errors

    def create_custom_config(
        self, base_config: str = "balanced", overrides: Dict[str, Any] = None
    ) -> SFRBacktestConfig:
        """
        Create custom configuration based on preset with overrides.

        Args:
            base_config: Base configuration name
            overrides: Dictionary of parameter overrides

        Returns:
            Custom SFRBacktestConfig instance
        """
        # Start with base configuration
        config = self.get_sfr_config(base_config)

        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    # Handle enum conversions
                    if key == "slippage_model" and isinstance(value, str):
                        value = SlippageModel(value)
                    elif key == "vix_regime_filter" and isinstance(value, str):
                        value = VixRegime(value) if value else None

                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown configuration parameter: {key}")

        # Validate the custom configuration
        errors = self.validate_configuration(config)
        if errors:
            raise ConfigurationError(
                f"Invalid custom configuration: {'; '.join(errors)}"
            )

        return config

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance optimization configuration."""
        return self.config_data.get(
            "performance",
            {
                "batch_size": 1000,
                "connection_pool_size": 20,
                "query_timeout_seconds": 30,
                "max_opportunities_in_memory": 10000,
                "max_trades_in_memory": 5000,
                "periodic_memory_cleanup": True,
                "max_concurrent_symbols": 4,
                "max_concurrent_days": 2,
                "async_processing": True,
            },
        )

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration settings."""
        return self.config_data.get(
            "output",
            {
                "console": {
                    "use_rich_formatting": True,
                    "show_progress_bars": True,
                    "detailed_tables": True,
                    "color_coding": True,
                },
                "export": {
                    "formats": ["json", "csv", "xlsx"],
                    "include_raw_data": False,
                    "include_trade_details": True,
                    "include_rejection_analysis": True,
                },
                "database_storage": {
                    "store_opportunities": True,
                    "store_trades": True,
                    "store_rejections": True,
                    "store_analytics": True,
                    "cleanup_old_runs": False,
                },
            },
        )

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config_data.get(
            "logging",
            {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_logging": False,
                "log_file_path": "logs/sfr_backtest.log",
                "max_file_size": "10MB",
                "backup_count": 5,
            },
        )

    def export_config_template(self, output_path: str) -> None:
        """
        Export a template configuration file.

        Args:
            output_path: Path where to save the template
        """
        template_config = {
            "# SFR Backtesting Configuration Template": None,
            "database": {
                "host": "localhost",
                "port": 5433,
                "database": "options_arbitrage",
                "user": "trading_user",
                "password": "secure_trading_password",
            },
            "target_symbols": ["SPY", "QQQ", "AAPL", "MSFT"],
            "configurations": {
                "custom": {
                    "profit_target": 0.50,
                    "cost_limit": 150.0,
                    "slippage_model": "LINEAR",
                    "commission_per_contract": 1.00,
                }
            },
        }

        try:
            with open(output_path, "w") as f:
                yaml.dump(template_config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration template exported to: {output_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to export template to {output_path}: {e}")


def load_config(config_path: Optional[str] = None) -> SFRConfigLoader:
    """
    Convenience function to load configuration.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configured SFRConfigLoader instance
    """
    return SFRConfigLoader(config_path)


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_loader = load_config()

    # Get database config
    db_config = config_loader.get_database_config()
    print(f"Database: {db_config.connection_string}")

    # Get available configurations
    configs = config_loader.get_available_configurations()
    print(f"Available configurations: {configs}")

    # Load specific configuration
    conservative_config = config_loader.get_sfr_config("conservative")
    print(f"Conservative config profit target: {conservative_config.profit_target}%")

    # Validate configuration
    errors = config_loader.validate_configuration(conservative_config)
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Configuration is valid")

    # Create custom configuration
    custom_config = config_loader.create_custom_config(
        base_config="balanced",
        overrides={
            "profit_target": 0.75,
            "cost_limit": 200.0,
            "slippage_model": "IMPACT",
        },
    )
    print(
        f"Custom config: profit_target={custom_config.profit_target}%, cost_limit=${custom_config.cost_limit}"
    )
