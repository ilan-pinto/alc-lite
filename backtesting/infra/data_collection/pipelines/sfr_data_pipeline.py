"""
Automated SFR Data Pipeline.

Orchestrates end-to-end data collection, validation, and preparation
specifically for SFR (Synthetic Free Risk) backtesting requirements.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, TaskID
from rich.table import Table

try:
    from ..config.config import DatabaseConfig, HistoricalConfig
    from ..data_sources.csv_data_source import CSVDataSource
    from ..data_sources.data_source_adapter import DataSourceManager
    from ..data_sources.external_api_source import ExternalAPISource
    from ..data_sources.ib_data_source import IBDataSource
    from ..historical_sfr_loader import (
        HistoricalSFRLoader,
        SFRDataLoadConfig,
        SFRDataLoadResult,
    )
    from ..validation.quality_metrics import DataQualityMetrics, DataQualityReport
    from ..validation.sfr_validator import SFRDataValidator, SFRValidationResult
except ImportError:
    from backtesting.infra.data_collection.config.config import (
        DatabaseConfig,
        HistoricalConfig,
    )
    from backtesting.infra.data_collection.data_sources.csv_data_source import (
        CSVDataSource,
    )
    from backtesting.infra.data_collection.data_sources.data_source_adapter import (
        DataSourceManager,
    )
    from backtesting.infra.data_collection.data_sources.external_api_source import (
        ExternalAPISource,
    )
    from backtesting.infra.data_collection.data_sources.ib_data_source import (
        IBDataSource,
    )
    from backtesting.infra.data_collection.historical_sfr_loader import (
        HistoricalSFRLoader,
        SFRDataLoadConfig,
        SFRDataLoadResult,
    )
    from backtesting.infra.data_collection.validation.quality_metrics import (
        DataQualityMetrics,
        DataQualityReport,
    )
    from backtesting.infra.data_collection.validation.sfr_validator import (
        SFRDataValidator,
        SFRValidationResult,
    )

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class SFRPipelineConfig:
    """Configuration for SFR data pipeline."""

    # Pipeline identification
    pipeline_name: str = "SFR Backtesting Data Pipeline"
    pipeline_version: str = "1.0.0"

    # Data collection settings
    symbols: List[str] = field(
        default_factory=lambda: [
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
        ]
    )
    lookback_days: int = 365
    include_vix_data: bool = True

    # Data source priorities (1 = highest priority)
    data_source_priorities: Dict[str, int] = field(
        default_factory=lambda: {
            "ib": 1,  # Interactive Brokers (highest quality)
            "csv": 2,  # Local CSV files
            "yahoo": 3,  # Yahoo Finance API
            "alphavantage": 4,  # Alpha Vantage API
        }
    )

    # Validation settings
    enable_validation: bool = True
    validation_detailed: bool = True
    quality_threshold: float = 0.7

    # Pipeline behavior
    fail_on_validation_error: bool = False
    retry_failed_symbols: bool = True
    max_retries: int = 2
    parallel_processing: bool = True
    max_concurrent_symbols: int = 4

    # Output settings
    generate_reports: bool = True
    save_intermediate_results: bool = True
    output_directory: str = "./sfr_pipeline_output"

    # Performance settings
    batch_size: int = 50
    connection_timeout: int = 30
    request_delay: float = 0.1


@dataclass
class SFRPipelineResult:
    """Result of SFR data pipeline execution."""

    pipeline_id: str
    execution_start: datetime
    execution_end: Optional[datetime] = None
    success: bool = False

    # Summary statistics
    total_symbols: int = 0
    successful_loads: int = 0
    failed_loads: int = 0
    validation_passes: int = 0
    validation_failures: int = 0

    # Detailed results
    load_results: Dict[str, SFRDataLoadResult] = field(default_factory=dict)
    validation_results: Dict[str, SFRValidationResult] = field(default_factory=dict)
    quality_reports: Dict[str, DataQualityReport] = field(default_factory=dict)

    # Issues and recommendations
    pipeline_errors: List[str] = field(default_factory=list)
    pipeline_warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Performance metrics
    total_records_loaded: int = 0
    total_load_time_seconds: float = 0.0
    records_per_second: float = 0.0

    @property
    def execution_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.execution_end:
            return (self.execution_end - self.execution_start).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_symbols == 0:
            return 0.0
        return (self.successful_loads / self.total_symbols) * 100

    @property
    def validation_pass_rate(self) -> float:
        """Get validation pass rate percentage."""
        total_validations = self.validation_passes + self.validation_failures
        if total_validations == 0:
            return 0.0
        return (self.validation_passes / total_validations) * 100


class SFRDataPipeline:
    """
    Automated SFR Data Collection Pipeline.

    Orchestrates the complete workflow for collecting, validating, and preparing
    historical data for SFR arbitrage backtesting:

    1. Data Source Management - Configure and prioritize data sources
    2. Historical Data Collection - Load stock and options data
    3. Data Validation - Comprehensive SFR-specific validation
    4. Quality Assessment - Generate quality metrics and reports
    5. Results Aggregation - Compile and save results
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool,
        config: SFRPipelineConfig = None,
        db_config: DatabaseConfig = None,
        historical_config: HistoricalConfig = None,
    ):
        self.db_pool = db_pool
        self.config = config or SFRPipelineConfig()
        self.db_config = db_config or DatabaseConfig()
        self.historical_config = historical_config or HistoricalConfig()

        # Pipeline components
        self.data_source_manager = DataSourceManager()
        self.sfr_loader: Optional[HistoricalSFRLoader] = None
        self.sfr_validator = SFRDataValidator(db_pool)
        self.quality_metrics = DataQualityMetrics(db_pool)

        # Runtime state
        self.pipeline_id = str(uuid.uuid4())
        self.output_path = Path(self.config.output_directory)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.execution_stats = {
            "symbols_processed": 0,
            "data_sources_used": set(),
            "total_api_calls": 0,
            "cache_hits": 0,
        }

    async def execute_pipeline(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        symbols: Optional[List[str]] = None,
    ) -> SFRPipelineResult:
        """
        Execute the complete SFR data pipeline.

        Args:
            start_date: Start date for data collection (default: lookback from today)
            end_date: End date for data collection (default: today)
            symbols: List of symbols to process (default: from config)

        Returns:
            SFRPipelineResult with comprehensive execution results
        """
        # Initialize execution
        result = SFRPipelineResult(
            pipeline_id=self.pipeline_id, execution_start=datetime.now()
        )

        # Set defaults
        end_date = end_date or date.today()
        start_date = start_date or (
            end_date - timedelta(days=self.config.lookback_days)
        )
        symbols = symbols or self.config.symbols

        result.total_symbols = len(symbols)

        console.print(
            Panel(
                f"[bold blue]SFR Data Pipeline[/bold blue]\n"
                f"Pipeline ID: {self.pipeline_id}\n"
                f"Symbols: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})\n"
                f"Date Range: {start_date} to {end_date}\n"
                f"Lookback: {(end_date - start_date).days} days",
                title="Pipeline Configuration",
            )
        )

        try:
            # Phase 1: Initialize data sources
            await self._initialize_data_sources(result)

            # Phase 2: Load historical data
            with Progress() as progress:
                await self._execute_data_loading_phase(
                    symbols, start_date, end_date, progress, result
                )

            # Phase 3: Validate data quality
            if self.config.enable_validation:
                with Progress() as progress:
                    await self._execute_validation_phase(
                        symbols, start_date, end_date, progress, result
                    )

            # Phase 4: Generate quality reports
            if self.config.generate_reports:
                await self._execute_reporting_phase(
                    symbols, start_date, end_date, result
                )

            # Phase 5: Finalize and save results
            await self._finalize_pipeline_execution(result)

            result.success = True
            console.print(f"[green]Pipeline completed successfully![/green]")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            result.pipeline_errors.append(f"Pipeline execution error: {str(e)}")
            result.success = False
        finally:
            result.execution_end = datetime.now()
            await self._cleanup_resources()

        return result

    async def _initialize_data_sources(self, result: SFRPipelineResult):
        """Initialize and configure data sources."""
        console.print("[bold cyan]Phase 1: Initializing Data Sources[/bold cyan]")

        # Initialize data sources based on configuration and availability
        data_sources_initialized = []

        # Interactive Brokers
        if "ib" in self.config.data_source_priorities:
            try:
                ib_source = IBDataSource()
                connected = await ib_source.connect()
                if connected:
                    self.data_source_manager.add_source(
                        ib_source,
                        priority=self.config.data_source_priorities["ib"],
                        is_primary=True,
                    )
                    data_sources_initialized.append("Interactive Brokers")
                    self.execution_stats["data_sources_used"].add("ib")
            except Exception as e:
                logger.warning(f"Failed to initialize IB data source: {e}")
                result.pipeline_warnings.append(f"IB connection failed: {str(e)}")

        # CSV Data Source
        if "csv" in self.config.data_source_priorities:
            try:
                csv_data_path = self.output_path / "csv_data"
                if csv_data_path.exists():
                    csv_source = CSVDataSource(str(csv_data_path))
                    connected = await csv_source.connect()
                    if connected:
                        self.data_source_manager.add_source(
                            csv_source,
                            priority=self.config.data_source_priorities["csv"],
                        )
                        data_sources_initialized.append("CSV Files")
                        self.execution_stats["data_sources_used"].add("csv")
            except Exception as e:
                logger.warning(f"Failed to initialize CSV data source: {e}")

        # External APIs
        for api_name in ["yahoo", "alphavantage"]:
            if api_name in self.config.data_source_priorities:
                try:
                    if api_name == "yahoo":
                        api_source = ExternalAPISource("yahoo")
                    else:
                        # Would need API key from environment or config
                        continue

                    connected = await api_source.connect()
                    if connected:
                        self.data_source_manager.add_source(
                            api_source,
                            priority=self.config.data_source_priorities[api_name],
                        )
                        data_sources_initialized.append(f"{api_name.title()} API")
                        self.execution_stats["data_sources_used"].add(api_name)
                except Exception as e:
                    logger.warning(f"Failed to initialize {api_name} API: {e}")

        if not data_sources_initialized:
            raise RuntimeError("No data sources could be initialized")

        console.print(
            f"✓ Initialized {len(data_sources_initialized)} data sources: {', '.join(data_sources_initialized)}"
        )

    async def _execute_data_loading_phase(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        progress: Progress,
        result: SFRPipelineResult,
    ):
        """Execute the data loading phase."""
        console.print("[bold cyan]Phase 2: Loading Historical Data[/bold cyan]")

        # Initialize SFR loader with primary data source
        primary_source = self.data_source_manager.primary_source
        if primary_source and hasattr(primary_source, "ib"):
            ib_connection = primary_source.ib
        else:
            ib_connection = None

        sfr_config = SFRDataLoadConfig(
            max_concurrent_symbols=self.config.max_concurrent_symbols,
            batch_size_options=self.config.batch_size,
        )

        self.sfr_loader = HistoricalSFRLoader(
            db_pool=self.db_pool,
            ib_connection=ib_connection,
            config=self.historical_config,
            sfr_config=sfr_config,
            data_source_adapter=self.data_source_manager.primary_source,
        )

        # Execute data loading
        load_results = await self.sfr_loader.load_sfr_dataset(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            include_vix=self.config.include_vix_data,
            progress=progress,
        )

        # Process results
        for symbol, load_result in load_results.items():
            result.load_results[symbol] = load_result

            if load_result.success:
                result.successful_loads += 1
                result.total_records_loaded += (
                    load_result.stock_bars_loaded + load_result.option_bars_loaded
                )
                result.total_load_time_seconds += load_result.load_duration_seconds
            else:
                result.failed_loads += 1
                result.pipeline_warnings.append(
                    f"Failed to load data for {symbol}: {load_result.error_message}"
                )

        # Calculate performance metrics
        if result.total_load_time_seconds > 0:
            result.records_per_second = (
                result.total_records_loaded / result.total_load_time_seconds
            )

        console.print(
            f"✓ Data loading completed: {result.successful_loads}/{result.total_symbols} symbols successful"
        )

    async def _execute_validation_phase(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        progress: Progress,
        result: SFRPipelineResult,
    ):
        """Execute the data validation phase."""
        console.print("[bold cyan]Phase 3: Validating Data Quality[/bold cyan]")

        # Only validate symbols that were successfully loaded
        symbols_to_validate = [
            symbol
            for symbol in symbols
            if symbol in result.load_results and result.load_results[symbol].success
        ]

        if not symbols_to_validate:
            console.print(
                "[yellow]No symbols to validate (no successful loads)[/yellow]"
            )
            return

        # Execute SFR-specific validation
        validation_results = await self.sfr_validator.validate_multiple_symbols(
            symbols=symbols_to_validate,
            start_date=start_date,
            end_date=end_date,
            detailed_analysis=self.config.validation_detailed,
        )

        # Process validation results
        for symbol, validation_result in validation_results.items():
            result.validation_results[symbol] = validation_result

            if validation_result.passed:
                result.validation_passes += 1
            else:
                result.validation_failures += 1

                # Check if we should fail the pipeline
                if self.config.fail_on_validation_error:
                    if validation_result.overall_score < self.config.quality_threshold:
                        result.pipeline_errors.append(
                            f"Validation failed for {symbol}: score {validation_result.overall_score:.3f} "
                            f"below threshold {self.config.quality_threshold}"
                        )

        console.print(
            f"✓ Validation completed: {result.validation_passes}/{len(symbols_to_validate)} symbols passed"
        )

    async def _execute_reporting_phase(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        result: SFRPipelineResult,
    ):
        """Execute the quality reporting phase."""
        console.print("[bold cyan]Phase 4: Generating Quality Reports[/bold cyan]")

        # Generate quality reports for successfully loaded symbols
        symbols_for_reports = [
            symbol
            for symbol in symbols
            if symbol in result.load_results and result.load_results[symbol].success
        ]

        for symbol in symbols_for_reports:
            try:
                # Determine data type based on what was loaded
                load_result = result.load_results[symbol]
                if load_result.option_contracts_loaded > 0:
                    data_type = "mixed"  # Both stock and options
                else:
                    data_type = "stock"

                # Generate comprehensive quality report
                quality_report = (
                    await self.quality_metrics.generate_comprehensive_quality_report(
                        symbol=symbol,
                        data_type=data_type,
                        start_date=start_date,
                        end_date=end_date,
                        detailed_analysis=True,
                    )
                )

                result.quality_reports[symbol] = quality_report

                # Save individual report
                if self.config.save_intermediate_results:
                    report_path = (
                        self.output_path
                        / f"quality_report_{symbol}_{self.pipeline_id[:8]}.json"
                    )
                    await self.quality_metrics.save_quality_report(
                        quality_report, str(report_path)
                    )

            except Exception as e:
                logger.error(f"Error generating quality report for {symbol}: {e}")
                result.pipeline_warnings.append(
                    f"Quality report generation failed for {symbol}: {str(e)}"
                )

        console.print(f"✓ Generated {len(result.quality_reports)} quality reports")

    async def _finalize_pipeline_execution(self, result: SFRPipelineResult):
        """Finalize pipeline execution and save comprehensive results."""
        console.print("[bold cyan]Phase 5: Finalizing Results[/bold cyan]")

        # Generate pipeline recommendations
        self._generate_pipeline_recommendations(result)

        # Save comprehensive pipeline result
        if self.config.save_intermediate_results:
            await self._save_pipeline_results(result)

        # Generate summary report
        self._display_execution_summary(result)

        # Update execution statistics
        self.execution_stats["symbols_processed"] = result.total_symbols

        console.print("✓ Pipeline execution finalized")

    def _generate_pipeline_recommendations(self, result: SFRPipelineResult):
        """Generate pipeline-level recommendations."""
        success_rate = result.success_rate
        validation_pass_rate = result.validation_pass_rate

        if success_rate < 80:
            result.recommendations.append(
                f"Low success rate ({success_rate:.1f}%) - consider improving data source reliability"
            )

        if validation_pass_rate < 70:
            result.recommendations.append(
                f"Low validation pass rate ({validation_pass_rate:.1f}%) - review data quality requirements"
            )

        if result.records_per_second < 100:
            result.recommendations.append(
                "Low data loading performance - consider optimizing data collection or using faster sources"
            )

        # Symbol-specific recommendations
        low_quality_symbols = [
            symbol
            for symbol, quality_report in result.quality_reports.items()
            if quality_report.overall_quality_score < 0.7
        ]

        if low_quality_symbols:
            result.recommendations.append(
                f"Low quality data for symbols: {', '.join(low_quality_symbols[:5])}"
                f"{'...' if len(low_quality_symbols) > 5 else ''}"
            )

        if (
            result.successful_loads == result.total_symbols
            and result.validation_passes == len(result.validation_results)
        ):
            result.recommendations.append(
                "Excellent pipeline execution - data is ready for SFR backtesting"
            )

    async def _save_pipeline_results(self, result: SFRPipelineResult):
        """Save comprehensive pipeline results."""
        # Convert result to serializable dictionary
        result_dict = {
            "pipeline_id": result.pipeline_id,
            "execution_start": result.execution_start.isoformat(),
            "execution_end": (
                result.execution_end.isoformat() if result.execution_end else None
            ),
            "success": result.success,
            "total_symbols": result.total_symbols,
            "successful_loads": result.successful_loads,
            "failed_loads": result.failed_loads,
            "validation_passes": result.validation_passes,
            "validation_failures": result.validation_failures,
            "total_records_loaded": result.total_records_loaded,
            "total_load_time_seconds": result.total_load_time_seconds,
            "records_per_second": result.records_per_second,
            "success_rate": result.success_rate,
            "validation_pass_rate": result.validation_pass_rate,
            "execution_duration": result.execution_duration,
            "pipeline_errors": result.pipeline_errors,
            "pipeline_warnings": result.pipeline_warnings,
            "recommendations": result.recommendations,
            "config": {
                "symbols": self.config.symbols,
                "lookback_days": self.config.lookback_days,
                "data_sources_used": list(self.execution_stats["data_sources_used"]),
                "validation_enabled": self.config.enable_validation,
                "quality_threshold": self.config.quality_threshold,
            },
            "load_results_summary": {
                symbol: {
                    "success": lr.success,
                    "stock_bars_loaded": lr.stock_bars_loaded,
                    "option_contracts_loaded": lr.option_contracts_loaded,
                    "option_bars_loaded": lr.option_bars_loaded,
                    "data_quality_score": lr.data_quality_score,
                    "load_duration_seconds": lr.load_duration_seconds,
                    "error_message": lr.error_message,
                }
                for symbol, lr in result.load_results.items()
            },
            "validation_results_summary": {
                symbol: {
                    "passed": vr.passed,
                    "overall_score": vr.overall_score,
                    "completeness_score": vr.completeness_score,
                    "liquidity_score": vr.liquidity_score,
                    "arbitrage_suitability_score": vr.arbitrage_suitability_score,
                    "critical_issues_count": len(vr.critical_issues),
                    "warnings_count": len(vr.warnings),
                }
                for symbol, vr in result.validation_results.items()
            },
        }

        # Save main pipeline result
        result_path = (
            self.output_path / f"sfr_pipeline_result_{self.pipeline_id[:8]}.json"
        )
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        console.print(f"✓ Pipeline results saved to {result_path}")

    def _display_execution_summary(self, result: SFRPipelineResult):
        """Display execution summary table."""
        # Create summary table
        table = Table(title="SFR Pipeline Execution Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")

        # Overall metrics
        table.add_row("Pipeline ID", result.pipeline_id[:12] + "...", "")
        table.add_row(
            "Execution Time",
            f"{result.execution_duration:.1f}s" if result.execution_duration else "N/A",
            "",
        )
        table.add_row("Total Symbols", str(result.total_symbols), "")

        # Loading metrics
        success_status = (
            "✓"
            if result.success_rate >= 90
            else "⚠" if result.success_rate >= 70 else "✗"
        )
        table.add_row(
            "Load Success Rate", f"{result.success_rate:.1f}%", success_status
        )
        table.add_row("Records Loaded", f"{result.total_records_loaded:,}", "")
        table.add_row(
            "Load Performance", f"{result.records_per_second:.0f} records/sec", ""
        )

        # Validation metrics
        if result.validation_passes + result.validation_failures > 0:
            val_status = (
                "✓"
                if result.validation_pass_rate >= 80
                else "⚠" if result.validation_pass_rate >= 60 else "✗"
            )
            table.add_row(
                "Validation Pass Rate",
                f"{result.validation_pass_rate:.1f}%",
                val_status,
            )

        # Issues
        error_status = "✓" if len(result.pipeline_errors) == 0 else "✗"
        table.add_row("Pipeline Errors", str(len(result.pipeline_errors)), error_status)
        table.add_row("Warnings", str(len(result.pipeline_warnings)), "")

        console.print(table)

        # Display top recommendations
        if result.recommendations:
            console.print("\n[bold yellow]Key Recommendations:[/bold yellow]")
            for i, rec in enumerate(result.recommendations[:3], 1):
                console.print(f"{i}. {rec}")

    async def _cleanup_resources(self):
        """Clean up resources used during pipeline execution."""
        try:
            # Disconnect data sources
            await self.data_source_manager.disconnect_all()

            # Clear loader cache
            if self.sfr_loader and hasattr(self.sfr_loader, "data_cache"):
                self.sfr_loader.data_cache.clear()

        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and statistics."""
        return {
            "pipeline_id": self.pipeline_id,
            "config": {
                "symbols_count": len(self.config.symbols),
                "lookback_days": self.config.lookback_days,
                "validation_enabled": self.config.enable_validation,
                "parallel_processing": self.config.parallel_processing,
            },
            "execution_stats": dict(self.execution_stats),
            "data_sources": {
                "available": self.data_source_manager.get_available_sources(),
                "connection_status": self.data_source_manager.get_connection_status(),
            },
            "output_directory": str(self.output_path),
        }


# Utility functions for pipeline execution


async def run_sfr_pipeline_simple(
    symbols: List[str] = None, days_back: int = 90, db_config: DatabaseConfig = None
) -> SFRPipelineResult:
    """
    Simple function to run SFR pipeline with default settings.

    Args:
        symbols: Symbols to process (default: top SFR targets)
        days_back: Days of historical data to collect
        db_config: Database configuration

    Returns:
        SFRPipelineResult
    """
    if symbols is None:
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]

    if db_config is None:
        db_config = DatabaseConfig()

    # Create database pool
    db_pool = await asyncpg.create_pool(
        db_config.connection_string,
        min_size=db_config.min_pool_size,
        max_size=db_config.max_pool_size,
    )

    try:
        # Configure pipeline
        config = SFRPipelineConfig(
            symbols=symbols,
            lookback_days=days_back,
            max_concurrent_symbols=2,  # Conservative for simple execution
        )

        # Create and execute pipeline
        pipeline = SFRDataPipeline(db_pool, config)
        result = await pipeline.execute_pipeline()

        return result

    finally:
        await db_pool.close()


async def run_full_sfr_pipeline(config_file: Optional[str] = None) -> SFRPipelineResult:
    """
    Run full SFR pipeline with configuration from file or defaults.

    Args:
        config_file: Optional JSON configuration file

    Returns:
        SFRPipelineResult
    """
    # Load configuration
    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            config_data = json.load(f)

        config = SFRPipelineConfig(**config_data)
    else:
        config = SFRPipelineConfig()

    # Database setup
    db_config = DatabaseConfig()
    db_pool = await asyncpg.create_pool(
        db_config.connection_string,
        min_size=db_config.min_pool_size,
        max_size=db_config.max_pool_size,
    )

    try:
        # Create and execute pipeline
        pipeline = SFRDataPipeline(db_pool, config)

        console.print(
            f"[bold green]Starting SFR Pipeline with {len(config.symbols)} symbols[/bold green]"
        )

        result = await pipeline.execute_pipeline()

        return result

    finally:
        await db_pool.close()


if __name__ == "__main__":
    # Example usage
    async def main():
        console.print("[bold blue]SFR Data Pipeline Example[/bold blue]")

        # Run simple pipeline for testing
        result = await run_sfr_pipeline_simple(symbols=["SPY", "QQQ"], days_back=30)

        if result.success:
            console.print(f"[green]Pipeline completed successfully![/green]")
            console.print(
                f"Loaded data for {result.successful_loads}/{result.total_symbols} symbols"
            )
            console.print(f"Total records: {result.total_records_loaded:,}")
        else:
            console.print(
                f"[red]Pipeline failed with {len(result.pipeline_errors)} errors[/red]"
            )

    # Run the example
    asyncio.run(main())
