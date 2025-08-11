"""
CSV data source adapter.

Provides access to historical data stored in CSV files,
supporting various formats and data types for backtesting.
"""

import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import logging
import pandas as pd

from .data_source_adapter import (
    DataFrequency,
    DataSourceAdapter,
    DataType,
    MarketDataPoint,
    MarketDataRequest,
)

logger = logging.getLogger(__name__)


class CSVDataSource(DataSourceAdapter):
    """
    CSV file data source adapter.

    Supports various CSV formats for historical data:
    - Standard OHLCV format
    - Options data with Greeks
    - Custom formats with configurable columns
    """

    def __init__(
        self,
        data_directory: str,
        config: Dict[str, Any] = None,
    ):
        super().__init__("CSV Files", config)

        self.data_directory = Path(data_directory)
        self.data_cache = {}  # Cache for loaded data

        # Default column mappings
        self.default_stock_columns = {
            "timestamp": ["timestamp", "date", "datetime", "Date", "DateTime"],
            "open": ["open", "Open", "OPEN"],
            "high": ["high", "High", "HIGH"],
            "low": ["low", "Low", "LOW"],
            "close": ["close", "Close", "CLOSE"],
            "volume": ["volume", "Volume", "VOLUME"],
            "price": ["price", "Price", "close", "Close"],
        }

        self.default_option_columns = {
            **self.default_stock_columns,
            "bid": ["bid", "Bid", "BID"],
            "ask": ["ask", "Ask", "ASK"],
            "strike": ["strike", "Strike", "STRIKE"],
            "expiry": ["expiry", "expiration", "Expiry", "Expiration"],
            "option_type": ["type", "right", "option_type", "Right", "Type"],
            "delta": ["delta", "Delta", "DELTA"],
            "gamma": ["gamma", "Gamma", "GAMMA"],
            "theta": ["theta", "Theta", "THETA"],
            "vega": ["vega", "Vega", "VEGA"],
            "implied_vol": ["iv", "implied_vol", "ImpliedVol", "IV"],
        }

        # File naming patterns
        self.file_patterns = {
            "stock": [
                "{symbol}_stock.csv",
                "{symbol}_daily.csv",
                "{symbol}.csv",
                "stocks/{symbol}.csv",
                "equities/{symbol}.csv",
            ],
            "options": [
                "{symbol}_options.csv",
                "{symbol}_opts.csv",
                "options/{symbol}.csv",
                "options/{symbol}_options.csv",
            ],
            "vix": ["VIX.csv", "vix.csv", "VIX_daily.csv", "indices/VIX.csv"],
        }

        # Set capabilities
        self._capabilities = {
            DataType.TRADES,
            DataType.QUOTES,
            DataType.BID_ASK,
            DataType.VOLUME,
            DataType.GREEKS,
            DataType.IMPLIED_VOL,
        }

        # Supported frequencies (depends on data)
        self._supported_frequencies = [
            DataFrequency.MINUTE,
            DataFrequency.FIVE_MINUTES,
            DataFrequency.FIFTEEN_MINUTES,
            DataFrequency.HOUR,
            DataFrequency.DAILY,
            DataFrequency.WEEKLY,
            DataFrequency.MONTHLY,
        ]

    async def connect(self) -> bool:
        """Connect to CSV data source (validate directory)."""
        try:
            if not self.data_directory.exists():
                logger.error(f"Data directory does not exist: {self.data_directory}")
                return False

            if not self.data_directory.is_dir():
                logger.error(f"Data path is not a directory: {self.data_directory}")
                return False

            # Check for some data files
            csv_files = list(self.data_directory.glob("**/*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found in {self.data_directory}")

            self._connected = True
            logger.info(f"Connected to CSV data source: {self.data_directory}")
            logger.info(f"Found {len(csv_files)} CSV files")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to CSV data source: {e}")
            self._connected = False
            return False

    async def disconnect(self):
        """Disconnect from CSV data source."""
        self.data_cache.clear()
        self._connected = False
        logger.info("Disconnected from CSV data source")

    async def get_stock_data(self, request: MarketDataRequest) -> List[MarketDataPoint]:
        """Get stock data from CSV files."""
        if not self.is_connected:
            raise RuntimeError("Not connected to CSV data source")

        # Find stock data file
        stock_file = self._find_data_file(request.symbol, "stock")
        if not stock_file:
            logger.warning(f"No stock data file found for {request.symbol}")
            return []

        try:
            # Load and parse CSV
            df = self._load_csv_file(stock_file, "stock")
            if df is None or df.empty:
                return []

            # Filter by date range
            df = self._filter_by_date_range(df, request.start_date, request.end_date)

            # Apply frequency filtering if needed
            if request.frequency != DataFrequency.TICK:
                df = self._resample_data(df, request.frequency)

            # Convert to MarketDataPoint objects
            data_points = []
            for _, row in df.iterrows():
                point = MarketDataPoint(
                    timestamp=row["timestamp"],
                    symbol=request.symbol,
                    price=row.get("price", row.get("close")),
                    volume=row.get("volume"),
                    bid=row.get("bid"),
                    ask=row.get("ask"),
                    open=row.get("open"),
                    high=row.get("high"),
                    low=row.get("low"),
                    close=row.get("close"),
                    data_source=self.name,
                    data_quality=0.8,  # CSV data quality score
                )

                # Apply filters
                if self._passes_filters(point, request):
                    data_points.append(point)

            logger.debug(
                f"Loaded {len(data_points)} stock data points for {request.symbol} "
                f"from {stock_file.name}"
            )

            return data_points

        except Exception as e:
            logger.error(f"Error loading stock data for {request.symbol}: {e}")
            return []

    async def get_option_data(
        self, request: MarketDataRequest
    ) -> List[MarketDataPoint]:
        """Get option data from CSV files."""
        if not self.is_connected:
            raise RuntimeError("Not connected to CSV data source")

        if not all([request.strike, request.expiry, request.option_type]):
            raise ValueError("Option data requires strike, expiry, and option_type")

        # Find options data file
        options_file = self._find_data_file(request.symbol, "options")
        if not options_file:
            logger.warning(f"No options data file found for {request.symbol}")
            return []

        try:
            # Load and parse CSV
            df = self._load_csv_file(options_file, "options")
            if df is None or df.empty:
                return []

            # Filter by option contract
            df = df[
                (df["strike"] == request.strike)
                & (df["expiry"] == request.expiry)
                & (df["option_type"] == request.option_type)
            ]

            if df.empty:
                logger.debug(
                    f"No data found for {request.symbol} "
                    f"{request.strike}{request.option_type} {request.expiry}"
                )
                return []

            # Filter by date range
            df = self._filter_by_date_range(df, request.start_date, request.end_date)

            # Apply frequency filtering if needed
            if request.frequency != DataFrequency.TICK:
                df = self._resample_data(df, request.frequency)

            # Convert to MarketDataPoint objects
            data_points = []
            for _, row in df.iterrows():
                point = MarketDataPoint(
                    timestamp=row["timestamp"],
                    symbol=request.symbol,
                    price=row.get("price", row.get("close")),
                    volume=row.get("volume"),
                    bid=row.get("bid"),
                    ask=row.get("ask"),
                    open=row.get("open"),
                    high=row.get("high"),
                    low=row.get("low"),
                    close=row.get("close"),
                    strike=row.get("strike"),
                    expiry=row.get("expiry"),
                    option_type=row.get("option_type"),
                    delta=row.get("delta"),
                    gamma=row.get("gamma"),
                    theta=row.get("theta"),
                    vega=row.get("vega"),
                    implied_vol=row.get("implied_vol"),
                    data_source=self.name,
                    data_quality=0.8,
                )

                # Apply filters
                if self._passes_filters(point, request):
                    data_points.append(point)

            logger.debug(
                f"Loaded {len(data_points)} option data points for "
                f"{request.symbol} {request.strike}{request.option_type} {request.expiry} "
                f"from {options_file.name}"
            )

            return data_points

        except Exception as e:
            logger.error(
                f"Error loading option data for {request.symbol} "
                f"{request.strike}{request.option_type} {request.expiry}: {e}"
            )
            return []

    async def get_option_chain(
        self, symbol: str, expiry: date, date_range: Tuple[date, date] = None
    ) -> List[Dict[str, Any]]:
        """Get option chain from CSV data."""
        options_file = self._find_data_file(symbol, "options")
        if not options_file:
            return []

        try:
            df = self._load_csv_file(options_file, "options")
            if df is None or df.empty:
                return []

            # Filter by expiry
            chain_df = df[df["expiry"] == expiry]

            if chain_df.empty:
                return []

            # Get unique strike/type combinations
            chain_contracts = []
            for _, row in (
                chain_df[["strike", "option_type"]].drop_duplicates().iterrows()
            ):
                contract_info = {
                    "symbol": symbol,
                    "expiry": expiry,
                    "strike": row["strike"],
                    "right": row["option_type"],
                    "exchange": "CSV",
                    "multiplier": 100,
                    "tradingClass": symbol,
                }
                chain_contracts.append(contract_info)

            return chain_contracts

        except Exception as e:
            logger.error(f"Error loading option chain for {symbol} {expiry}: {e}")
            return []

    async def get_vix_data(
        self,
        start_date: date,
        end_date: date,
        frequency: DataFrequency = DataFrequency.DAILY,
    ) -> List[MarketDataPoint]:
        """Get VIX data from CSV files."""
        vix_file = self._find_data_file("VIX", "vix")
        if not vix_file:
            logger.warning("No VIX data file found")
            return []

        vix_request = MarketDataRequest(
            symbol="VIX",
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
        )

        return await self.get_stock_data(vix_request)

    def _find_data_file(self, symbol: str, data_type: str) -> Optional[Path]:
        """Find data file for a symbol and data type."""
        patterns = self.file_patterns.get(data_type, [])

        for pattern in patterns:
            file_path = self.data_directory / pattern.format(symbol=symbol)
            if file_path.exists():
                return file_path

        # Try case-insensitive search
        for pattern in patterns:
            file_pattern = pattern.format(symbol=symbol.lower())
            file_path = self.data_directory / file_pattern
            if file_path.exists():
                return file_path

            file_pattern = pattern.format(symbol=symbol.upper())
            file_path = self.data_directory / file_pattern
            if file_path.exists():
                return file_path

        return None

    def _load_csv_file(self, file_path: Path, data_type: str) -> Optional[pd.DataFrame]:
        """Load and standardize CSV file."""
        cache_key = str(file_path)

        if cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()

        try:
            # Try different CSV reading strategies
            df = None

            # Strategy 1: Standard CSV
            try:
                df = pd.read_csv(file_path)
            except Exception:
                pass

            # Strategy 2: Different separator
            if df is None or df.empty:
                try:
                    df = pd.read_csv(file_path, sep=";")
                except Exception:
                    pass

            # Strategy 3: Different date format
            if df is None or df.empty:
                try:
                    df = pd.read_csv(
                        file_path, parse_dates=True, infer_datetime_format=True
                    )
                except Exception:
                    pass

            if df is None or df.empty:
                logger.error(f"Could not load CSV file: {file_path}")
                return None

            # Standardize columns
            df = self._standardize_columns(df, data_type)

            # Parse timestamp column
            df = self._parse_timestamps(df)

            # Cache the result
            self.data_cache[cache_key] = df.copy()

            return df

        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Standardize column names."""
        column_mappings = (
            self.default_option_columns
            if data_type == "options"
            else self.default_stock_columns
        )

        # Create reverse mapping
        reverse_mapping = {}
        for standard_name, possible_names in column_mappings.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    reverse_mapping[possible_name] = standard_name
                    break

        # Rename columns
        df = df.rename(columns=reverse_mapping)

        # Ensure we have essential columns
        if "timestamp" not in df.columns:
            if "Date" in df.columns:
                df["timestamp"] = df["Date"]
            elif df.index.name in ["Date", "date", "timestamp"]:
                df = df.reset_index()
                df["timestamp"] = df[df.index.name]

        return df

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp column to datetime."""
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            df = df.sort_values("timestamp")

        return df

    def _filter_by_date_range(
        self, df: pd.DataFrame, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if "timestamp" not in df.columns:
            return df

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        mask = (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
        return df[mask]

    def _resample_data(
        self, df: pd.DataFrame, frequency: DataFrequency
    ) -> pd.DataFrame:
        """Resample data to specified frequency."""
        if "timestamp" not in df.columns:
            return df

        frequency_map = {
            DataFrequency.MINUTE: "1T",
            DataFrequency.FIVE_MINUTES: "5T",
            DataFrequency.FIFTEEN_MINUTES: "15T",
            DataFrequency.HOUR: "1H",
            DataFrequency.DAILY: "1D",
            DataFrequency.WEEKLY: "1W",
            DataFrequency.MONTHLY: "1M",
        }

        freq_str = frequency_map.get(frequency)
        if not freq_str:
            return df

        df = df.set_index("timestamp")

        # Define aggregation rules
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "price": "last",
            "volume": "sum",
            "bid": "last",
            "ask": "last",
        }

        # Add option-specific columns
        for col in ["delta", "gamma", "theta", "vega", "implied_vol"]:
            if col in df.columns:
                agg_rules[col] = "last"

        # Only aggregate columns that exist
        existing_agg_rules = {
            col: rule for col, rule in agg_rules.items() if col in df.columns
        }

        if existing_agg_rules:
            df_resampled = df.resample(freq_str).agg(existing_agg_rules)
            df_resampled = df_resampled.dropna()
            df_resampled = df_resampled.reset_index()
            return df_resampled

        return df.reset_index()

    def _passes_filters(
        self, point: MarketDataPoint, request: MarketDataRequest
    ) -> bool:
        """Check if data point passes request filters."""
        # Volume filter
        if request.min_volume > 0:
            if point.volume is None or point.volume < request.min_volume:
                return False

        # Spread filter
        if request.max_spread_percent is not None:
            if point.bid is not None and point.ask is not None and point.price:
                spread_pct = ((point.ask - point.bid) / point.price) * 100
                if spread_pct > request.max_spread_percent:
                    return False

        return True

    async def validate_data_availability(
        self, symbol: str, start_date: date, end_date: date
    ) -> Dict[str, Any]:
        """Validate data availability in CSV files."""
        stock_file = self._find_data_file(symbol, "stock")
        options_file = self._find_data_file(symbol, "options")

        stock_available = stock_file is not None and stock_file.exists()
        options_available = options_file is not None and options_file.exists()

        # Calculate basic quality score
        quality_score = 0.0
        if stock_available:
            quality_score += 0.5
        if options_available:
            quality_score += 0.5

        notes = []
        if stock_file:
            notes.append(f"Stock data: {stock_file.name}")
        if options_file:
            notes.append(f"Options data: {options_file.name}")

        return {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "stock_data_available": stock_available,
            "option_data_available": options_available,
            "data_quality_score": quality_score,
            "missing_dates": [],  # Would need detailed analysis
            "notes": "; ".join(notes) if notes else "No data files found",
        }

    def list_available_symbols(self) -> List[str]:
        """List all available symbols in the CSV data directory."""
        symbols = set()

        # Look for stock files
        for pattern in self.file_patterns["stock"]:
            # Extract symbol pattern
            if "{symbol}" in pattern:
                pattern_regex = pattern.replace("{symbol}", "([A-Z]+)")
                for csv_file in self.data_directory.glob("**/*.csv"):
                    match = re.search(pattern_regex, csv_file.name)
                    if match:
                        symbols.add(match.group(1))

        return sorted(list(symbols))

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data."""
        symbols = self.list_available_symbols()

        summary = {
            "total_symbols": len(symbols),
            "symbols": symbols,
            "data_directory": str(self.data_directory),
            "total_csv_files": len(list(self.data_directory.glob("**/*.csv"))),
        }

        # Check for special files
        vix_file = self._find_data_file("VIX", "vix")
        summary["vix_available"] = vix_file is not None

        return summary
