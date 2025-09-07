# ADR-003: Parallel Leg Execution with Global Lock for SFR Arbitrage

**Status**: Approved
**Date**: 2025-09-06
**Author**: Claude Code
**Reviewer**: Options Trading Advisor Agent

## Context

### Problem Statement

The current SFR (Synthetic Free Risk) arbitrage implementation uses Interactive Brokers combo orders for conversion strategies (Buy Stock + Sell Call + Buy Put). While combo orders guarantee the total transaction price, they do not guarantee individual leg prices, leading to unprofitable trades despite meeting the overall limit price.

### Real-World Example

**Expected Execution:**
- Stock: $153.70 (buy)
- Call (150 strike): $8.65 (sell)
- Put (149 strike): $3.50 (buy)
- Combo Limit: $150.04
- Expected Net Credit: $5.15
- Expected Profit: $0.45

**Actual Execution:**
- Stock: $153.70 (buy) ✓
- Call (150 strike): $8.17 (sell) ❌ (-$0.48 vs expected)
- Put (149 strike): $3.78 (buy) ❌ (+$0.28 vs expected)
- Actual Net Credit: $4.39
- Actual Loss: -$0.31

The combo order filled within the $150.04 limit ($149.31 total) but individual leg slippage eliminated profitability.

### Root Cause Analysis

1. **Combo Order Limitation**: IB combo orders only enforce the total price, not individual leg prices
2. **Market Maker Internalization**: Combo orders may be internalized by market makers who optimize for their profit
3. **Price Discovery Failure**: Individual leg markets may offer better prices than combo market
4. **Lack of Control**: Cannot specify precise limits for each leg within a combo

## Decision

**We will implement Parallel Leg Execution with Global Lock for SFR arbitrage strategies.**

### Core Components:

1. **Parallel Execution**: Place all three legs as separate limit orders simultaneously
2. **Global Execution Lock**: Prevent multiple symbols from executing concurrently
3. **Individual Leg Limits**: Set precise limit prices for each leg
4. **Smart Fill Monitoring**: Track fills across all legs with timeout management
5. **Partial Fill Handling**: Sophisticated rollback for incomplete executions

## Rationale

### Why Parallel (Not Sequential)?

Sequential execution was considered but rejected based on expert trading advisor analysis:

- **Market Risk**: 10-30 seconds between legs introduces unacceptable directional risk
- **Opportunity Decay**: Arbitrage opportunities disappear quickly; speed is critical
- **Slippage Multiplication**: Later legs face worse prices as market makers adjust
- **Professional Standard**: Real arbitrage desks use parallel execution exclusively

### Why Global Lock?

Multiple concurrent executions can cause:

- **Capital Overuse**: Exceeding available margin/capital limits
- **Position Conflicts**: Multiple positions interfering with each other
- **Rollback Complexity**: Partial fills across symbols creating cleanup problems
- **Risk Management**: Inability to track total exposure during execution

## Architecture

### High-Level Flow

```
Symbol A finds opportunity → Acquire Global Lock → Execute 3 legs in parallel
    ↓
Pause all other executors → Monitor fills → Handle results → Release lock
    ↓
SUCCESS: Stop all executors → Disconnect from IB → Exit
    OR
FAILURE: Resume other executors → Continue searching
```

### Core Classes

#### 1. Global Execution Lock
```python
class GlobalExecutionLock:
    """Singleton managing global execution state"""

    def __init__(self):
        self.execution_in_progress = False
        self.executing_symbol = None
        self.execution_start_time = None
        self.execution_lock = asyncio.Lock()

    async def acquire_execution_lock(self, symbol: str) -> bool:
        """Try to acquire exclusive execution rights"""
        async with self.execution_lock:
            if self.execution_in_progress:
                return False

            self.execution_in_progress = True
            self.executing_symbol = symbol
            self.execution_start_time = time.time()
            return True

    async def release_execution_lock(self):
        """Release execution lock and cleanup state"""
        async with self.execution_lock:
            self.execution_in_progress = False
            self.executing_symbol = None
            self.execution_start_time = None
```

#### 2. Parallel Executor
```python
class ParallelLegExecutor:
    """Executes all three legs simultaneously"""

    async def execute_parallel_legs(self, opportunity):
        """Main parallel execution logic"""

        # 1. Calculate individual leg limits
        limits = self.calculate_parallel_limits(opportunity)

        # 2. Place all orders simultaneously
        orders = await asyncio.gather(
            self.place_stock_order(stock_contract, limits['stock']),
            self.place_call_order(call_contract, limits['call']),
            self.place_put_order(put_contract, limits['put']),
            return_exceptions=True
        )

        # 3. Monitor fills in parallel
        filled_legs = await self.monitor_parallel_fills(orders)

        # 4. Handle results
        if len(filled_legs) == 3:
            return self.handle_complete_fill(filled_legs)
        else:
            return await self.handle_partial_fills(filled_legs, orders)
```

### Execution Flow

#### 1. Order Placement (Simultaneous)
```python
async def execute_opportunity(self, opportunity):
    # Acquire global lock
    if not await global_execution_lock.acquire_execution_lock(self.symbol):
        logger.info(f"[{self.symbol}] Another execution in progress")
        self.finish_collection_without_execution("blocked_by_global_lock")
        return

    try:
        # Pause all other executors
        await self.strategy.pause_all_other_executors(self.symbol)

        # Execute all legs in parallel
        success = await self.parallel_executor.execute_parallel_legs(opportunity)

        if success:
            logger.info(f"[{self.symbol}] All legs executed successfully")
            # On success - stop everything and disconnect
            await self.strategy.stop_all_executors()
            logger.info("Successful execution - disconnecting from IB")
            self.ib.disconnect()
            return True
        else:
            logger.warning(f"[{self.symbol}] Execution failed or rolled back")
            # On failure - resume searching with other symbols

    finally:
        # Always release lock
        await global_execution_lock.release_execution_lock()

        # Only resume if execution failed
        if not success:
            await self.strategy.resume_all_executors()
            logger.info("Execution failed - resuming search with other symbols")
```

#### 2. Fill Monitoring
```python
async def monitor_parallel_fills(self, orders):
    """Monitor all orders with sophisticated timeout logic"""
    filled = []
    start_time = time.time()
    first_fill_time = None

    while time.time() - start_time < PARALLEL_TIMEOUT:
        # Check each order
        for order in orders:
            if order.filled and order not in filled:
                filled.append(order)
                if first_fill_time is None:
                    first_fill_time = time.time()

        # Partial fill timeout check
        if first_fill_time and time.time() - first_fill_time > PARTIAL_FILL_TIMEOUT:
            if len(filled) < 3:
                logger.warning("Partial fill timeout reached")
                break

        # Complete fill check
        if len(filled) == 3:
            return filled

        await asyncio.sleep(0.1)  # 100ms check interval

    return filled
```

#### 3. Partial Fill Handling
```python
async def handle_partial_fills(self, filled_legs, all_orders):
    """Smart partial fill recovery"""

    if len(filled_legs) == 0:
        # No fills - clean exit
        return False

    elif len(filled_legs) == 1:
        # Single leg filled - immediate liquidation
        await self.liquidate_single_leg(filled_legs[0])
        logger.info("Single leg liquidated")

    elif len(filled_legs) == 2:
        # Two legs filled - evaluate completion vs liquidation
        remaining_profit = self.calculate_remaining_profit(filled_legs)

        if remaining_profit > MIN_REMAINING_PROFIT:
            # Try aggressive completion with market order
            success = await self.complete_third_leg_aggressively()
            if success:
                return True
            else:
                await self.liquidate_partial_position(filled_legs)
        else:
            # Not profitable to complete - liquidate
            await self.liquidate_partial_position(filled_legs)

    # Cancel any unfilled orders
    await self.cancel_unfilled_orders(all_orders, filled_legs)
    return False
```

#### 4. Execution Reporting System

A comprehensive reporting system tracks and analyzes execution performance:

```python
@dataclass
class ExecutionReport:
    """Complete execution report with all metrics"""

    # Basic Information
    symbol: str
    execution_id: str
    timestamp: datetime
    execution_mode: str  # "parallel", "combo", "sequential"
    status: str  # "complete", "partial", "failed", "rolled_back"

    # Expected vs Actual Prices
    expected_stock_price: float
    actual_stock_price: float
    expected_call_price: float
    actual_call_price: float
    expected_put_price: float
    actual_put_price: float

    # Slippage Analysis
    stock_slippage: float  # actual - expected
    call_slippage: float   # actual - expected (negative = worse for selling)
    put_slippage: float    # actual - expected (positive = worse for buying)
    total_slippage: float

    # Profitability
    expected_profit: float
    actual_profit: float
    profit_variance: float
    profit_variance_pct: float

    # Execution Metrics
    total_execution_time: float  # seconds
    time_to_first_fill: float
    time_to_complete: float
    legs_attempted: int
    legs_filled: int
    fill_rate: float  # legs_filled / legs_attempted

    # Fill Timeline
    stock_fill_time: Optional[float]
    call_fill_time: Optional[float]
    put_fill_time: Optional[float]
    fill_sequence: List[str]  # ["stock", "call", "put"]

    # Costs and Fees
    commission_stock: float
    commission_call: float
    commission_put: float
    total_commission: float
    rollback_cost: float  # cost of unwinding partial fills

    # Final Results
    gross_pnl: float  # before commissions
    net_pnl: float    # after all costs
    roi: float        # return on invested capital

    # Error Information
    errors: List[str]
    warnings: List[str]
    rollback_reason: Optional[str]
```

#### Real-time Progress Reporting

```python
async def execute_with_reporting(self, opportunity):
    """Execute with real-time progress reporting"""

    # Initialize report
    report = ExecutionReport(
        symbol=self.symbol,
        execution_id=f"{self.symbol}_{int(time.time())}",
        timestamp=datetime.now(),
        execution_mode="parallel",
        status="executing"
    )

    # Real-time progress updates
    logger.info(f"[{self.symbol}] 🚀 EXECUTION STARTED")
    logger.info(f"[{self.symbol}] Expected: Stock=${report.expected_stock_price:.2f}, "
                f"Call=${report.expected_call_price:.2f}, Put=${report.expected_put_price:.2f}")
    logger.info(f"[{self.symbol}] Target Profit: ${report.expected_profit:.2f}")

    start_time = time.time()

    try:
        # Place all orders
        logger.info(f"[{self.symbol}] 📤 Placing 3 orders simultaneously...")
        orders = await self.place_parallel_orders(opportunity)

        # Monitor fills with progress updates
        filled_legs = []
        while len(filled_legs) < 3 and time.time() - start_time < PARALLEL_TIMEOUT:
            # Check for new fills
            new_fills = self.check_new_fills(orders, filled_legs)

            for fill in new_fills:
                filled_legs.append(fill)
                logger.info(f"[{self.symbol}] ✅ {fill.leg_type.upper()} filled @ ${fill.price:.2f} "
                           f"({len(filled_legs)}/3 complete)")

                # Update report with fill details
                self.update_report_with_fill(report, fill)

            await asyncio.sleep(0.1)

        # Final status
        if len(filled_legs) == 3:
            report.status = "complete"
            self.generate_success_report(report)
        else:
            report.status = "partial"
            await self.handle_partial_with_reporting(filled_legs, report)

    except Exception as e:
        report.status = "failed"
        report.errors.append(str(e))
        self.generate_failure_report(report)

    # Always log final summary
    self.log_execution_summary(report)

    # Save to persistent storage
    await self.save_execution_report(report)

    return report
```

#### Post-Execution Summary Report

```python
def generate_execution_summary(self, report: ExecutionReport) -> str:
    """Generate formatted execution summary"""

    status_emoji = {
        "complete": "✅",
        "partial": "⚠️",
        "failed": "❌",
        "rolled_back": "🔄"
    }

    summary = f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                           EXECUTION REPORT - {report.symbol.upper()}                           ║
║ {report.execution_id} - {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')} ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║ STATUS: {status_emoji.get(report.status, '❓')} {report.status.upper():<12} │ MODE: {report.execution_mode.upper():<10} │ DURATION: {report.total_execution_time:.1f}s      ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                   PRICE ANALYSIS                                  ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║ LEG     │ EXPECTED │ ACTUAL   │ SLIPPAGE │ IMPACT   │ FILL TIME                   ║
╠─────────┼──────────┼──────────┼──────────┼──────────┼─────────────────────────────╣
║ STOCK   │ ${report.expected_stock_price:>7.2f} │ ${report.actual_stock_price:>7.2f} │ ${report.stock_slippage:>+7.2f} │ {self.format_impact(report.stock_slippage):>7} │ {report.stock_fill_time or 'NO FILL':<27} ║
║ CALL    │ ${report.expected_call_price:>7.2f} │ ${report.actual_call_price:>7.2f} │ ${report.call_slippage:>+7.2f} │ {self.format_impact(report.call_slippage):>7} │ {report.call_fill_time or 'NO FILL':<27} ║
║ PUT     │ ${report.expected_put_price:>7.2f} │ ${report.actual_put_price:>7.2f} │ ${report.put_slippage:>+7.2f} │ {self.format_impact(report.put_slippage):>7} │ {report.put_fill_time or 'NO FILL':<27} ║
╠═════════╪══════════╪══════════╪══════════╪══════════╪═════════════════════════════╣
║ TOTAL   │    -     │    -     │ ${report.total_slippage:>+7.2f} │ {self.format_impact(report.total_slippage):>7} │ FILLS: {report.legs_filled}/{report.legs_attempted}                   ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║                                 PROFITABILITY                                     ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║ Expected Profit:  ${report.expected_profit:>8.2f}    │    Actual Profit:   ${report.actual_profit:>8.2f}         ║
║ Variance:         ${report.profit_variance:>+8.2f}    │    Variance %:      {report.profit_variance_pct:>+7.1f}%         ║
║ Commissions:      ${report.total_commission:>8.2f}    │    Rollback Cost:   ${report.rollback_cost:>8.2f}         ║
║ Gross P&L:        ${report.gross_pnl:>8.2f}    │    Net P&L:         ${report.net_pnl:>8.2f}         ║
║ ROI:              {report.roi:>7.2f}%     │    Fill Rate:       {report.fill_rate:>6.1%}           ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
"""

    if report.errors:
        summary += "\n❌ ERRORS:\n"
        for error in report.errors:
            summary += f"   • {error}\n"

    if report.warnings:
        summary += "\n⚠️  WARNINGS:\n"
        for warning in report.warnings:
            summary += f"   • {warning}\n"

    return summary

def format_impact(self, slippage: float) -> str:
    """Format slippage impact with color coding"""
    if abs(slippage) < 0.01:
        return "GOOD"
    elif abs(slippage) < 0.05:
        return "OK"
    elif abs(slippage) < 0.10:
        return "POOR"
    else:
        return "BAD"
```

#### Persistent Reporting and Analytics

```python
class ExecutionReportManager:
    """Manages execution report storage and analytics"""

    def __init__(self, db_connection):
        self.db = db_connection
        self.create_tables()

    async def save_execution_report(self, report: ExecutionReport):
        """Save execution report to database"""
        query = """
        INSERT INTO execution_reports (
            symbol, execution_id, timestamp, execution_mode, status,
            expected_profit, actual_profit, profit_variance,
            total_slippage, total_execution_time, legs_filled,
            net_pnl, roi, commission_paid, rollback_cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.db.execute(query, (
            report.symbol, report.execution_id, report.timestamp,
            report.execution_mode, report.status, report.expected_profit,
            report.actual_profit, report.profit_variance, report.total_slippage,
            report.total_execution_time, report.legs_filled, report.net_pnl,
            report.roi, report.total_commission, report.rollback_cost
        ))

    async def generate_daily_summary(self, date: datetime.date) -> str:
        """Generate daily execution summary"""
        reports = await self.get_reports_by_date(date)

        total_executions = len(reports)
        successful = len([r for r in reports if r.status == "complete"])
        partial = len([r for r in reports if r.status == "partial"])
        failed = len([r for r in reports if r.status == "failed"])

        total_pnl = sum(r.net_pnl for r in reports)
        total_slippage = sum(r.total_slippage for r in reports)
        avg_execution_time = sum(r.total_execution_time for r in reports) / total_executions if reports else 0

        return f"""
        📊 DAILY EXECUTION SUMMARY - {date}

        Executions: {total_executions} total
        Success Rate: {successful}/{total_executions} ({successful/total_executions*100:.1f}%)
        Partial Fills: {partial} ({partial/total_executions*100:.1f}%)
        Failures: {failed} ({failed/total_executions*100:.1f}%)

        Performance:
        Total P&L: ${total_pnl:.2f}
        Average Slippage: ${total_slippage/total_executions:.3f}
        Average Execution Time: {avg_execution_time:.1f}s

        Top Performers: {self.get_top_performers(reports)}
        Issues: {self.get_common_issues(reports)}
        """

    async def check_performance_alerts(self, report: ExecutionReport):
        """Check for performance degradation and send alerts"""

        # Alert thresholds
        if report.total_slippage > 0.20:  # >$0.20 total slippage
            await self.send_alert(f"HIGH SLIPPAGE: {report.symbol} slippage ${report.total_slippage:.2f}")

        if report.profit_variance_pct < -50:  # >50% profit reduction
            await self.send_alert(f"PROFIT DEGRADATION: {report.symbol} profit down {abs(report.profit_variance_pct):.1f}%")

        if report.total_execution_time > 30:  # >30 second execution
            await self.send_alert(f"SLOW EXECUTION: {report.symbol} took {report.total_execution_time:.1f}s")

        if report.status in ["partial", "failed"]:
            await self.send_alert(f"EXECUTION ISSUE: {report.symbol} status={report.status}")
```

### Post-Execution Behavior

The system behaves differently based on execution outcome:

#### **Successful Execution (All 3 Legs Filled)**
```python
async def handle_successful_execution(self):
    """Complete shutdown after successful arbitrage capture"""

    # 1. Generate and save execution report
    report = await self.generate_execution_report()
    await self.save_report(report)

    # 2. Log success with details
    logger.info(f"✅ ARBITRAGE CAPTURED: {report.symbol}")
    logger.info(f"   Net P&L: ${report.net_pnl:.2f}")
    logger.info(f"   Execution Time: {report.total_execution_time:.1f}s")

    # 3. Stop all executors (no need to search anymore)
    await self.strategy.stop_all_executors()
    logger.info("Stopping all executors - opportunity captured")

    # 4. Clean up market data subscriptions
    await self.cleanup_market_data()

    # 5. Disconnect from IB
    self.ib.disconnect()
    logger.info("Disconnected from IB - session complete")

    # 6. Exit program
    sys.exit(0)
```

#### **Partial Fill (1-2 Legs Filled) with Rollback Limits**
```python
async def handle_partial_fill(self, filled_legs):
    """Handle partial fills with rollback attempt tracking"""

    # Check rollback limits before attempting
    if self.rollback_manager.should_stop_rollbacks(self.symbol):
        logger.error(f"[{self.symbol}] Maximum rollback attempts reached - stopping execution")
        await self.strategy.stop_all_executors()
        self.ib.disconnect()
        sys.exit(1)

    # Record rollback attempt
    rollback_id = self.rollback_manager.start_rollback(
        symbol=self.symbol,
        filled_legs=filled_legs,
        reason="partial_fill_timeout"
    )

    try:
        # Attempt rollback
        rollback_cost = await self.rollback_partial_position_with_tracking(
            filled_legs, rollback_id
        )

        # Record successful rollback
        await self.rollback_manager.complete_rollback(
            rollback_id,
            cost=rollback_cost,
            success=True
        )

        if STOP_ON_PARTIAL_FILL:
            # Stop after rollback
            await self.strategy.stop_all_executors()
            self.ib.disconnect()
            logger.info("Partial fill rolled back - stopping per configuration")
            sys.exit(1)
        else:
            # Check if we should continue or pause this symbol
            if self.rollback_manager.should_pause_symbol(self.symbol):
                logger.warning(f"[{self.symbol}] Too many rollbacks - pausing symbol")
                await self.strategy.pause_symbol(self.symbol)
            else:
                # Continue searching
                await self.strategy.resume_all_executors()
                logger.info("Partial fill rolled back - continuing search")

    except RollbackError as e:
        # Record failed rollback
        await self.rollback_manager.complete_rollback(
            rollback_id,
            cost=0.0,
            success=False,
            error=str(e)
        )
        logger.error(f"[{self.symbol}] Rollback failed: {e}")

        # Stop on rollback failure
        await self.strategy.stop_all_executors()
        self.ib.disconnect()
        sys.exit(1)
```

#### **Failed Execution (No Fills)**
```python
async def handle_failed_execution(self):
    """Resume searching after complete failure"""

    # Log failure
    logger.warning(f"[{self.symbol}] No fills achieved")

    # Resume other executors to continue searching
    await self.strategy.resume_all_executors()
    logger.info("Resuming search with other symbols")

    # Continue running until opportunity found
```

#### **Rollback Manager System**
```python
@dataclass
class RollbackAttempt:
    """Track individual rollback attempts"""
    rollback_id: str
    symbol: str
    timestamp: datetime
    reason: str  # "partial_fill_timeout", "execution_error", etc.
    filled_legs: List[str]  # ["stock", "call"] or ["put", "stock"]
    rollback_cost: Optional[float] = None
    rollback_time: Optional[float] = None  # seconds to complete
    success: bool = False
    error_message: Optional[str] = None
    legs_rolled_back: List[str] = None

class RollbackManager:
    """Manages rollback attempts and enforces limits"""

    def __init__(self):
        self.rollback_attempts: List[RollbackAttempt] = []
        self.symbol_rollback_counts: Dict[str, int] = {}
        self.symbol_pause_until: Dict[str, datetime] = {}
        self.global_rollback_count = 0

    def should_stop_rollbacks(self, symbol: str) -> bool:
        """Check if we should stop attempting rollbacks"""

        # Global limit check
        if self.global_rollback_count >= MAX_ROLLBACK_ATTEMPTS:
            logger.error(f"Global rollback limit reached: {self.global_rollback_count}/{MAX_ROLLBACK_ATTEMPTS}")
            return True

        # Per-symbol limit check
        symbol_count = self.symbol_rollback_counts.get(symbol, 0)
        if symbol_count >= MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL:
            logger.error(f"Symbol rollback limit reached for {symbol}: {symbol_count}/{MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL}")
            return True

        return False

    def should_pause_symbol(self, symbol: str) -> bool:
        """Check if symbol should be paused after rollback"""
        symbol_count = self.symbol_rollback_counts.get(symbol, 0)
        return symbol_count >= (MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL - 1)

    def start_rollback(self, symbol: str, filled_legs: List, reason: str) -> str:
        """Start tracking a rollback attempt"""

        rollback_id = f"{symbol}_{int(time.time())}_{len(self.rollback_attempts)}"

        attempt = RollbackAttempt(
            rollback_id=rollback_id,
            symbol=symbol,
            timestamp=datetime.now(),
            reason=reason,
            filled_legs=[leg.leg_type for leg in filled_legs]
        )

        self.rollback_attempts.append(attempt)
        self.global_rollback_count += 1
        self.symbol_rollback_counts[symbol] = self.symbol_rollback_counts.get(symbol, 0) + 1

        # Log rollback start
        self.log_rollback_summary(attempt, "STARTED")

        return rollback_id

    async def complete_rollback(self, rollback_id: str, cost: float,
                              success: bool, error: str = None):
        """Complete rollback tracking"""

        attempt = next((a for a in self.rollback_attempts if a.rollback_id == rollback_id), None)
        if not attempt:
            logger.error(f"Rollback attempt not found: {rollback_id}")
            return

        attempt.rollback_cost = cost
        attempt.rollback_time = (datetime.now() - attempt.timestamp).total_seconds()
        attempt.success = success
        attempt.error_message = error

        # Log completion
        status = "COMPLETED" if success else "FAILED"
        self.log_rollback_summary(attempt, status)

        # Save to database/file
        await self.save_rollback_report(attempt)

    def log_rollback_summary(self, attempt: RollbackAttempt, status: str):
        """Generate formatted rollback summary log"""

        remaining_global = MAX_ROLLBACK_ATTEMPTS - self.global_rollback_count
        remaining_symbol = MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL - self.symbol_rollback_counts.get(attempt.symbol, 0)

        summary = f"""
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                         ROLLBACK SUMMARY #{self.global_rollback_count} - {status}                         ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║ Symbol: {attempt.symbol.upper():<15} │ ID: {attempt.rollback_id:<35} ║
║ Timestamp: {attempt.timestamp.strftime('%Y-%m-%d %H:%M:%S'):<25} │ Reason: {attempt.reason:<25} ║
╠═══════════════════════════════════════════════════════════════════════════════════╣
║ Filled Legs: {', '.join(attempt.filled_legs).upper():<57} ║
║ Rollback Cost: ${attempt.rollback_cost or 0.0:<8.2f} │ Time: {attempt.rollback_time or 0.0:<6.1f}s             ║
║ Success: {'✅ YES' if attempt.success else '❌ NO':<8} │ Global Attempts: {self.global_rollback_count}/{MAX_ROLLBACK_ATTEMPTS}              ║
║ Symbol Attempts: {self.symbol_rollback_counts.get(attempt.symbol, 0)}/{MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL}         │ Remaining: G={remaining_global}, S={remaining_symbol}                ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
        """

        if attempt.error_message and not attempt.success:
            summary += f"\n❌ ERROR: {attempt.error_message}\n"

        if remaining_global <= 1 or remaining_symbol <= 1:
            summary += "\n⚠️  WARNING: Approaching rollback limits!\n"

        logger.info(summary)

    def get_rollback_statistics(self) -> Dict:
        """Get rollback statistics for reporting"""
        successful = len([a for a in self.rollback_attempts if a.success])
        failed = len([a for a in self.rollback_attempts if not a.success])
        total_cost = sum(a.rollback_cost or 0 for a in self.rollback_attempts)
        avg_time = sum(a.rollback_time or 0 for a in self.rollback_attempts) / len(self.rollback_attempts) if self.rollback_attempts else 0

        return {
            "total_attempts": len(self.rollback_attempts),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(self.rollback_attempts) if self.rollback_attempts else 0,
            "total_cost": total_cost,
            "average_time": avg_time,
            "symbols_affected": len(self.symbol_rollback_counts),
            "global_remaining": MAX_ROLLBACK_ATTEMPTS - self.global_rollback_count,
        }
```

### Configuration Parameters

```python
# /modules/Arbitrage/sfr/constants.py

# Execution Strategy
EXECUTION_MODE = "parallel"  # "combo" | "parallel" | "sequential"
ENABLE_GLOBAL_LOCK = True

# Post-Execution Behavior
STOP_ON_FILL = True  # Stop all activity after successful execution
STOP_ON_PARTIAL_FILL = True  # Stop even on partial fills
MAX_EXECUTIONS_PER_SESSION = 1  # Disconnect after N successful executions
CONTINUE_SEARCH_ON_FAILURE = True  # Keep searching if execution fails

# Timing Configuration
PARALLEL_TIMEOUT = 15  # Total execution timeout (seconds)
PARTIAL_FILL_TIMEOUT = 5  # Timeout after first fill (seconds)
EXECUTOR_PAUSE_TIMEOUT = 60  # Max time to hold global lock

# Risk Management
MIN_REMAINING_PROFIT = 0.10  # Minimum $ profit to complete partial fills
MAX_SLIPPAGE_PER_LEG = 0.005  # 0.5% maximum slippage per leg
AGGRESSIVE_COMPLETION = True  # Use market orders for final leg

# Rollback Management
MAX_ROLLBACK_ATTEMPTS = 3  # Global limit across all symbols
MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL = 2  # Per-symbol limit
ROLLBACK_COOLDOWN_SECONDS = 300  # Wait 5min before retrying same symbol
STOP_ON_MAX_ROLLBACKS = True  # Exit when global limit reached
ROLLBACK_REPORT_TO_DB = True  # Save rollback reports to database

# Limit Price Buffers
STOCK_BUFFER_PCT = 0.001  # 0.1% buffer for stock purchase
CALL_BUFFER_PCT = 0.002  # 0.2% buffer for call sale
PUT_BUFFER_PCT = 0.002   # 0.2% buffer for put purchase
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `GlobalExecutionLock` singleton
2. Add executor pause/resume methods to `Strategy.py`
3. Implement parallel execution framework

### Phase 2: Execution Logic
4. Build `ParallelLegExecutor` class
5. Implement fill monitoring system
6. Add partial fill handling

### Phase 3: Risk Management
7. Implement rollback mechanisms
8. Add timeout and deadlock prevention
9. Create comprehensive logging

### Phase 4: Integration
10. Integrate with existing SFR executor
11. Add configuration switches
12. Update validation logic

## Benefits

### Immediate Benefits
- **Price Certainty**: Exact control over individual leg prices
- **Speed**: All legs hit market within milliseconds
- **No Internalization**: Direct access to individual leg markets
- **Risk Control**: Global lock prevents capital overuse

### Long-term Benefits
- **Scalability**: Framework supports multiple arbitrage strategies
- **Reliability**: Sophisticated partial fill handling
- **Transparency**: Clear visibility into execution quality
- **Profitability**: Eliminates unprofitable slippage issues

## Trade-offs

### Increased Complexity
- **More Code**: Parallel execution requires sophisticated coordination
- **More Failure Modes**: Partial fills create additional scenarios to handle
- **Testing Complexity**: Need to test all partial fill combinations

### Execution Risk
- **Partial Fill Risk**: May get stuck with incomplete positions
- **Timing Risk**: Market can move between order placement and fills
- **Rollback Cost**: May lose money unwinding partial positions

### Mitigation Strategies
- **Comprehensive Testing**: Test all scenarios in paper trading
- **Conservative Limits**: Use tight spreads and realistic buffers
- **Quick Rollback**: Minimize time in partial positions
- **Monitoring**: Real-time alerting for partial fills

## Alternatives Considered

### Alternative 1: Keep Combo Orders
**Rejected**: Does not solve the core slippage problem

### Alternative 2: Sequential Execution
**Rejected**: Expert analysis shows this introduces unacceptable market risk

### Alternative 3: Hybrid Approach
**Considered**: Use combo for liquid symbols, parallel for illiquid
**Deferred**: Adds complexity without clear benefit

## Migration Strategy

### Phase 1: Configuration Flag
```python
if EXECUTION_MODE == "parallel":
    result = await self.execute_parallel_legs(opportunity)
else:
    result = await self.execute_combo_order(opportunity)
```

### Phase 2: Paper Trading
- Test parallel execution with small sizes
- Monitor fill rates and execution quality
- Refine timeout and buffer parameters

### Phase 3: Gradual Rollout
- Start with high-liquidity symbols (SPY, QQQ)
- Monitor performance vs combo orders
- Expand to all symbols after validation

### Phase 4: Full Migration
- Make parallel execution the default
- Remove combo order fallback
- Archive legacy code

## Success Metrics

### Execution Quality
- **Fill Rate**: >95% complete fills within timeout
- **Price Improvement**: Actual prices vs expected prices
- **Rollback Rate**: <5% of executions require rollback

### Profitability
- **Slippage Reduction**: Eliminate large negative slippage events
- **Profit Consistency**: Actual profits match calculated profits
- **Win Rate**: Increase profitable execution percentage

### Operational Metrics
- **Execution Time**: Average time to complete all legs
- **System Stability**: No deadlocks or stuck executors
- **Error Rate**: <1% of executions fail due to system errors

## Related Documents

- [ADR-001: Daily Options Data Collection](ADR-001-daily-options-data-collection.md)
- [ADR-002: Fix Metric Collection Logging](ADR-002-fix-metric-collection-logging.md)
- [SFR Improvements](SFR_IMPROVEMENTS.md)

## Implementation Files

### New Files to Create
- `/modules/Arbitrage/sfr/global_execution_lock.py`
- `/modules/Arbitrage/sfr/parallel_executor.py`
- `/modules/Arbitrage/sfr/partial_fill_handler.py`
- `/modules/Arbitrage/sfr/execution_reporter.py`
- `/modules/Arbitrage/sfr/report_manager.py`
- `/modules/Arbitrage/sfr/rollback_manager.py`

### Files to Modify
- `/modules/Arbitrage/sfr/executor.py`
- `/modules/Arbitrage/Strategy.py`
- `/modules/Arbitrage/sfr/constants.py`
- `/modules/Arbitrage/sfr/opportunity_evaluator.py`

## Comprehensive Testing Requirements

### 1. Unit Tests - Individual Component Validation

#### 1.1 Global Execution Lock (`test_global_execution_lock.py`)

```python
class TestGlobalExecutionLock:
    """Comprehensive tests for singleton global execution lock"""

    def test_singleton_pattern(self):
        """Verify only one instance exists across threads"""
        lock1 = GlobalExecutionLock()
        lock2 = GlobalExecutionLock()
        assert lock1 is lock2
        assert id(lock1) == id(lock2)

    @pytest.mark.asyncio
    async def test_basic_lock_acquisition(self):
        """Test basic lock acquire/release cycle"""
        lock = await GlobalExecutionLock.get_instance()

        # Should acquire successfully
        result = await lock.acquire("SPY", "executor_1", "execution")
        assert result is True
        assert lock.is_locked() is True
        assert lock.get_current_holder().symbol == "SPY"

        # Release should work
        lock.release("SPY", "executor_1")
        assert lock.is_locked() is False
        assert lock.get_current_holder() is None

    @pytest.mark.asyncio
    async def test_concurrent_acquisition_blocking(self):
        """Test that second executor waits for lock"""
        lock = await GlobalExecutionLock.get_instance()

        # First executor gets lock
        result1 = await lock.acquire("SPY", "executor_1", "execution")
        assert result1 is True

        # Second executor should timeout
        result2 = await lock.acquire("QQQ", "executor_2", "execution", timeout=0.1)
        assert result2 is False

        # After release, second can acquire
        lock.release("SPY", "executor_1")
        result3 = await lock.acquire("QQQ", "executor_2", "execution")
        assert result3 is True

    def test_lock_holder_validation(self):
        """Test lock release validation"""
        lock = GlobalExecutionLock()

        # Cannot release unlocked lock
        with pytest.warns():
            lock.release("SPY", "executor_1")

        # Cannot release from wrong holder
        asyncio.run(lock.acquire("SPY", "executor_1"))
        with pytest.warns():
            lock.release("QQQ", "executor_2")

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test lock acquisition timeouts"""
        lock = await GlobalExecutionLock.get_instance()

        # Hold lock
        await lock.acquire("SPY", "executor_1")
        start_time = time.time()

        # Should timeout after 1 second
        result = await lock.acquire("QQQ", "executor_2", timeout=1.0)
        elapsed = time.time() - start_time

        assert result is False
        assert 0.9 <= elapsed <= 1.5  # Allow some variance

    def test_performance_metrics_tracking(self):
        """Test lock statistics and metrics"""
        lock = GlobalExecutionLock()

        # Initially no stats
        stats = lock.get_lock_stats()
        assert stats["total_locks_acquired"] == 0
        assert stats["lock_contentions"] == 0

        # After acquisition
        asyncio.run(lock.acquire("SPY", "executor_1"))
        time.sleep(0.1)  # Hold briefly
        lock.release("SPY", "executor_1")

        stats = lock.get_lock_stats()
        assert stats["total_locks_acquired"] == 1
        assert stats["total_lock_time_seconds"] > 0
        assert stats["average_lock_duration_seconds"] > 0

    def test_force_release_emergency(self):
        """Test emergency force release functionality"""
        lock = GlobalExecutionLock()

        # Acquire lock
        asyncio.run(lock.acquire("SPY", "executor_1"))
        assert lock.is_locked() is True

        # Force release
        result = asyncio.run(lock.force_release("test_emergency"))
        assert result is True
        assert lock.is_locked() is False

    def test_lock_history_tracking(self):
        """Test lock holder history"""
        lock = GlobalExecutionLock()

        # Multiple acquisitions
        for i in range(3):
            asyncio.run(lock.acquire("SPY", f"executor_{i}"))
            time.sleep(0.01)
            lock.release("SPY", f"executor_{i}")

        history = lock.get_recent_history(3)
        assert len(history) == 3
        assert history[-1]["executor_id"] == "executor_2"
```

#### 1.2 Parallel Executor (`test_parallel_executor.py`)

```python
class TestParallelLegExecutor:
    """Test parallel order execution logic"""

    @pytest.fixture
    def mock_ib_setup(self):
        """Setup mock IB environment"""
        mock_ib = MagicMock()
        mock_stock = MagicMock(conId=1, symbol="SPY", secType="STK")
        mock_call = MagicMock(conId=2, symbol="SPY", secType="OPT", right="C")
        mock_put = MagicMock(conId=3, symbol="SPY", secType="OPT", right="P")

        return {
            "ib": mock_ib,
            "contracts": {"stock": mock_stock, "call": mock_call, "put": mock_put}
        }

    @pytest.mark.asyncio
    async def test_successful_parallel_execution(self, mock_ib_setup):
        """Test complete 3-leg parallel execution success"""
        executor = ParallelLegExecutor(
            ib=mock_ib_setup["ib"],
            symbol="SPY",
            on_execution_complete=None,
            on_execution_failed=None
        )

        # Mock successful fills for all legs
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=True)
        ]

        result = await executor.execute_parallel_arbitrage(
            stock_contract=mock_ib_setup["contracts"]["stock"],
            call_contract=mock_ib_setup["contracts"]["call"],
            put_contract=mock_ib_setup["contracts"]["put"],
            stock_price=100.0,
            call_price=8.50,
            put_price=3.25,
            quantity=1
        )

        assert result.success is True
        assert result.all_legs_filled is True
        assert result.legs_filled == 3
        assert result.total_legs == 3
        assert result.total_execution_time > 0

    @pytest.mark.asyncio
    async def test_partial_fill_scenarios(self, mock_ib_setup):
        """Test handling of partial fill scenarios"""
        executor = ParallelLegExecutor(
            ib=mock_ib_setup["ib"],
            symbol="SPY"
        )

        # Test 1: Only stock fills
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=False),
            create_mock_trade("put", filled=False)
        ]

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        assert result.success is False
        assert result.legs_filled == 1
        assert result.partially_filled is True

        # Test 2: Two legs fill (stock + call)
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=True),
            create_mock_trade("put", filled=False)
        ]

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        assert result.legs_filled == 2
        assert result.partially_filled is True

    @pytest.mark.asyncio
    async def test_execution_timeout_handling(self, mock_ib_setup):
        """Test execution timeout scenarios"""
        executor = ParallelLegExecutor(
            ib=mock_ib_setup["ib"],
            symbol="SPY"
        )

        # Mock slow fills that exceed timeout
        async def slow_fill_mock(*args, **kwargs):
            await asyncio.sleep(2.0)  # Slower than timeout
            return create_mock_trade("stock", filled=True)

        mock_ib_setup["ib"].placeOrder.side_effect = slow_fill_mock

        # Set short timeout for test
        with patch.object(executor, 'EXECUTION_TIMEOUT', 0.5):
            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

        assert result.success is False
        assert "timeout" in result.error_message.lower()

    def test_price_calculation_accuracy(self):
        """Test accurate price calculations and slippage"""
        executor = ParallelLegExecutor(ib=MagicMock(), symbol="SPY")

        # Expected prices
        expected = {"stock": 100.0, "call": 8.50, "put": 3.25}

        # Actual fill prices with slippage
        actual = {"stock": 100.02, "call": 8.47, "put": 3.28}

        slippage = executor._calculate_slippage(expected, actual)

        assert slippage["stock"] == 0.02  # Bought higher
        assert slippage["call"] == -0.03  # Sold lower (bad)
        assert slippage["put"] == 0.03   # Bought higher
        assert slippage["total"] == 0.02  # Net slippage

    @pytest.mark.asyncio
    async def test_rollback_invocation(self, mock_ib_setup):
        """Test that rollback is properly invoked for partial fills"""
        executor = ParallelLegExecutor(
            ib=mock_ib_setup["ib"],
            symbol="SPY"
        )

        # Mock partial fill scenario
        mock_ib_setup["ib"].placeOrder.side_effect = [
            create_mock_trade("stock", filled=True),
            create_mock_trade("call", filled=False),
            create_mock_trade("put", filled=False)
        ]

        with patch.object(executor, '_handle_rollback') as mock_rollback:
            mock_rollback.return_value = True  # Rollback succeeds

            result = await executor.execute_parallel_arbitrage(
                **create_test_execution_params()
            )

            # Verify rollback was called
            mock_rollback.assert_called_once()
            assert result.success is False
            assert "rollback" in result.error_message.lower()

    def test_execution_result_completeness(self):
        """Test that ExecutionResult contains all required fields"""
        result = ExecutionResult(
            success=True,
            execution_id="test_123",
            symbol="SPY",
            total_execution_time=2.5,
            all_legs_filled=True,
            legs_filled=3,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=1002.5,
            total_slippage=2.5
        )

        # Verify all critical fields present
        assert hasattr(result, 'success')
        assert hasattr(result, 'execution_id')
        assert hasattr(result, 'symbol')
        assert hasattr(result, 'total_execution_time')
        assert hasattr(result, 'slippage_percentage')
        assert hasattr(result, 'stock_result')
        assert hasattr(result, 'call_result')
        assert hasattr(result, 'put_result')
```

#### 1.3 Rollback Manager (`test_rollback_manager.py`)

```python
class TestRollbackManager:
    """Test rollback attempt tracking and limits"""

    @pytest.fixture
    def rollback_manager(self):
        """Create fresh rollback manager"""
        manager = RollbackManager()
        manager.reset_counters()  # Ensure clean state
        return manager

    def test_global_rollback_limit_enforcement(self, rollback_manager):
        """Test global rollback limit (3 attempts max)"""

        # Should allow first 3 rollbacks
        for i in range(3):
            assert rollback_manager.should_stop_rollbacks("SPY") is False
            rollback_id = rollback_manager.start_rollback(
                symbol="SPY",
                filled_legs=["stock"],
                reason="test"
            )
            asyncio.run(rollback_manager.complete_rollback(rollback_id, 0.0, True))

        # 4th rollback should be blocked
        assert rollback_manager.should_stop_rollbacks("QQQ") is True

    def test_per_symbol_rollback_limit_enforcement(self, rollback_manager):
        """Test per-symbol rollback limit (2 attempts max)"""

        # SPY: 2 rollbacks allowed
        for i in range(2):
            assert rollback_manager.should_stop_rollbacks("SPY") is False
            rollback_id = rollback_manager.start_rollback(
                symbol="SPY",
                filled_legs=["stock"],
                reason="test"
            )
            asyncio.run(rollback_manager.complete_rollback(rollback_id, 0.0, True))

        # SPY: 3rd rollback blocked (per-symbol limit)
        assert rollback_manager.should_stop_rollbacks("SPY") is True

        # QQQ: Should still be allowed (different symbol)
        assert rollback_manager.should_stop_rollbacks("QQQ") is False

    def test_rollback_attempt_tracking(self, rollback_manager):
        """Test detailed rollback attempt tracking"""

        filled_legs = [MockLeg("stock", 100.0), MockLeg("call", 8.5)]

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY",
            filled_legs=filled_legs,
            reason="partial_fill_timeout"
        )

        # Verify attempt tracked
        attempt = next(
            (a for a in rollback_manager.rollback_attempts if a.rollback_id == rollback_id),
            None
        )
        assert attempt is not None
        assert attempt.symbol == "SPY"
        assert attempt.reason == "partial_fill_timeout"
        assert attempt.filled_legs == ["stock", "call"]
        assert attempt.success is False  # Not completed yet

    @pytest.mark.asyncio
    async def test_rollback_completion_tracking(self, rollback_manager):
        """Test rollback completion with success/failure tracking"""

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY",
            filled_legs=[MockLeg("stock", 100.0)],
            reason="test"
        )

        # Complete successfully
        await rollback_manager.complete_rollback(
            rollback_id=rollback_id,
            cost=2.50,
            success=True
        )

        attempt = next(
            a for a in rollback_manager.rollback_attempts
            if a.rollback_id == rollback_id
        )

        assert attempt.success is True
        assert attempt.rollback_cost == 2.50
        assert attempt.rollback_time > 0
        assert attempt.error_message is None

    @pytest.mark.asyncio
    async def test_rollback_failure_tracking(self, rollback_manager):
        """Test rollback failure tracking"""

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY",
            filled_legs=[MockLeg("stock", 100.0)],
            reason="test"
        )

        # Complete with failure
        await rollback_manager.complete_rollback(
            rollback_id=rollback_id,
            cost=0.0,
            success=False,
            error="Market order failed"
        )

        attempt = next(
            a for a in rollback_manager.rollback_attempts
            if a.rollback_id == rollback_id
        )

        assert attempt.success is False
        assert attempt.rollback_cost == 0.0
        assert attempt.error_message == "Market order failed"

    def test_symbol_pause_recommendation(self, rollback_manager):
        """Test symbol pause logic after repeated rollbacks"""

        # One rollback for SPY
        rollback_id = rollback_manager.start_rollback("SPY", [], "test")
        asyncio.run(rollback_manager.complete_rollback(rollback_id, 0.0, True))

        # Should recommend pause after 1st rollback (approaching limit)
        assert rollback_manager.should_pause_symbol("SPY") is True

        # Fresh symbol should not need pause
        assert rollback_manager.should_pause_symbol("QQQ") is False

    def test_rollback_statistics_accuracy(self, rollback_manager):
        """Test rollback statistics calculation"""

        # Create mix of successful and failed rollbacks
        for i, success in enumerate([True, True, False, True]):
            rollback_id = rollback_manager.start_rollback(f"SYM{i}", [], "test")
            cost = 2.0 if success else 0.0
            asyncio.run(rollback_manager.complete_rollback(rollback_id, cost, success))

        stats = rollback_manager.get_rollback_statistics()

        assert stats["total_attempts"] == 4
        assert stats["successful"] == 3
        assert stats["failed"] == 1
        assert stats["success_rate"] == 0.75
        assert stats["total_cost"] == 6.0
        assert stats["symbols_affected"] == 4

    def test_rollback_summary_logging(self, rollback_manager, caplog):
        """Test rollback summary log generation"""

        rollback_id = rollback_manager.start_rollback(
            symbol="SPY",
            filled_legs=[MockLeg("stock", 100.0)],
            reason="partial_fill_timeout"
        )

        # Should generate start log
        assert "ROLLBACK SUMMARY" in caplog.text
        assert "STARTED" in caplog.text
        assert "SPY" in caplog.text

        caplog.clear()

        # Complete rollback
        asyncio.run(rollback_manager.complete_rollback(rollback_id, 5.0, True))

        # Should generate completion log
        assert "COMPLETED" in caplog.text
        assert "$5.00" in caplog.text
```

#### 1.4 Execution Reporter (`test_execution_reporter.py`)

```python
class TestExecutionReporter:
    """Test execution reporting and analysis"""

    @pytest.fixture
    def sample_execution_result(self):
        """Create sample execution result for testing"""
        return ExecutionResult(
            success=True,
            execution_id="SPY_1234567890",
            symbol="SPY",
            total_execution_time=2.34,
            all_legs_filled=True,
            legs_filled=3,
            total_legs=3,
            expected_total_cost=1000.0,
            actual_total_cost=1002.5,
            total_slippage=2.5,
            slippage_percentage=0.25,
            stock_result={
                "leg_type": "stock",
                "action": "BUY",
                "target_price": 100.0,
                "avg_fill_price": 100.02,
                "slippage": 0.02
            },
            call_result={
                "leg_type": "call",
                "action": "SELL",
                "target_price": 8.50,
                "avg_fill_price": 8.47,
                "slippage": -0.03
            },
            put_result={
                "leg_type": "put",
                "action": "BUY",
                "target_price": 3.25,
                "avg_fill_price": 3.28,
                "slippage": 0.03
            }
        )

    def test_console_report_generation(self, sample_execution_result):
        """Test console report formatting and content"""
        reporter = ExecutionReporter()

        report = reporter.generate_execution_report(
            sample_execution_result,
            level=ReportLevel.DETAILED,
            format_type=ReportFormat.CONSOLE
        )

        # Verify key content present
        assert "SPY" in report
        assert "SUCCESS" in report
        assert "$2.50" in report  # Total slippage
        assert "0.25%" in report  # Slippage percentage
        assert "2.34s" in report  # Execution time
        assert "3/3" in report    # Legs filled

        # Verify leg-specific data
        assert "STOCK" in report and "BUY" in report
        assert "CALL" in report and "SELL" in report
        assert "PUT" in report and "BUY" in report

    def test_json_report_structure(self, sample_execution_result):
        """Test JSON report structure and completeness"""
        reporter = ExecutionReporter()

        json_report = reporter.generate_execution_report(
            sample_execution_result,
            format_type=ReportFormat.JSON
        )

        import json
        report_data = json.loads(json_report)

        # Verify main sections present
        assert "execution_summary" in report_data
        assert "financial_summary" in report_data
        assert "slippage_analysis" in report_data
        assert "performance_metrics" in report_data
        assert "leg_details" in report_data

        # Verify critical fields
        exec_summary = report_data["execution_summary"]
        assert exec_summary["symbol"] == "SPY"
        assert exec_summary["success"] is True
        assert exec_summary["legs_filled"] == "3/3"

    def test_html_report_generation(self, sample_execution_result):
        """Test HTML report generation"""
        reporter = ExecutionReporter()

        html_report = reporter.generate_execution_report(
            sample_execution_result,
            format_type=ReportFormat.HTML
        )

        # Basic HTML structure
        assert "<html>" in html_report
        assert "<body>" in html_report
        assert "</html>" in html_report

        # Content verification
        assert "SPY" in html_report
        assert "SUCCESS" in html_report
        assert "$2.50" in html_report

    def test_slippage_analysis_accuracy(self, sample_execution_result):
        """Test slippage analysis calculations"""
        reporter = ExecutionReporter()

        # Access the internal slippage analysis method
        slippage_analysis = reporter._analyze_slippage(sample_execution_result)

        assert slippage_analysis.total_slippage_dollars == 2.5
        assert slippage_analysis.slippage_percentage == 0.25
        assert slippage_analysis.worst_leg == "put"  # Highest absolute slippage
        assert slippage_analysis.worst_leg_slippage == 0.03
        assert slippage_analysis.avg_slippage_per_leg == pytest.approx(0.0267, abs=0.001)

    def test_performance_metrics_calculation(self, sample_execution_result):
        """Test performance metrics calculation"""
        reporter = ExecutionReporter()

        # First execution
        reporter.generate_execution_report(sample_execution_result)

        # Second execution
        sample_execution_result.total_execution_time = 3.1
        sample_execution_result.total_slippage = 1.8
        reporter.generate_execution_report(sample_execution_result)

        metrics = reporter._calculate_performance_metrics(sample_execution_result)

        assert metrics.all_legs_fill_rate == 100.0  # Both successful
        assert metrics.average_slippage_dollars == pytest.approx(2.15, abs=0.1)  # Average of 2.5 and 1.8

    def test_session_statistics_tracking(self, sample_execution_result):
        """Test session-wide statistics tracking"""
        reporter = ExecutionReporter()

        # Generate multiple reports
        for i in range(5):
            sample_execution_result.execution_id = f"SPY_{i}"
            sample_execution_result.success = (i < 4)  # 4 successes, 1 failure
            reporter.generate_execution_report(sample_execution_result)

        stats = reporter.get_session_statistics()

        assert stats["total_executions"] == 5
        assert stats["successful_executions"] == 4
        assert stats["success_rate_percent"] == 80.0
        assert stats["reports_generated"] == 5

    def test_report_export_functionality(self):
        """Test report export to file"""
        reporter = ExecutionReporter()

        # Create some session data
        reporter.session_metrics["total_executions"] = 3
        reporter.session_metrics["successful_executions"] = 2

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filename = f.name

        try:
            success = reporter.export_session_report(filename, ReportFormat.JSON)
            assert success is True

            # Verify file content
            with open(filename, 'r') as f:
                data = json.load(f)
                assert data["total_executions"] == 3
                assert data["successful_executions"] == 2
                assert data["success_rate"] == pytest.approx(66.67, abs=0.1)

        finally:
            os.unlink(filename)

    def test_report_level_filtering(self, sample_execution_result):
        """Test different report detail levels"""
        reporter = ExecutionReporter()

        # Summary level - basic info only
        summary_report = reporter.generate_execution_report(
            sample_execution_result,
            level=ReportLevel.SUMMARY
        )

        # Detailed level - includes leg breakdown
        detailed_report = reporter.generate_execution_report(
            sample_execution_result,
            level=ReportLevel.DETAILED
        )

        # Comprehensive level - includes performance metrics
        comprehensive_report = reporter.generate_execution_report(
            sample_execution_result,
            level=ReportLevel.COMPREHENSIVE
        )

        # Summary should be shortest
        assert len(summary_report) < len(detailed_report)
        assert len(detailed_report) < len(comprehensive_report)

        # Comprehensive should have performance data
        assert "Performance Metrics" in comprehensive_report or "Session" in comprehensive_report
```

### 2. Integration Tests - Component Interaction Validation

#### 2.1 End-to-End Execution Flow (`test_sfr_parallel_integration.py`)

```python
class TestParallelExecutionIntegration:
    """Test complete parallel execution flow integration"""

    @pytest.fixture
    def integration_setup(self):
        """Setup complete integration environment"""
        mock_ib = MagicMock()
        mock_order_manager = MagicMock()

        # Create opportunity evaluator with real-like data
        opportunity_evaluator = OpportunityEvaluator("SPY")

        return {
            "ib": mock_ib,
            "order_manager": mock_order_manager,
            "opportunity_evaluator": opportunity_evaluator,
            "symbol": "SPY"
        }

    @pytest.mark.asyncio
    async def test_complete_successful_execution_flow(self, integration_setup):
        """Test complete successful parallel execution flow"""

        # Create integrator
        integrator = await create_parallel_integrator(**integration_setup)

        # Create profitable opportunity
        opportunity = create_profitable_opportunity("SPY")

        # Mock successful fills
        mock_successful_fills(integration_setup["ib"])

        # Execute opportunity
        result = await integrator.execute_opportunity(opportunity)

        # Verify successful execution
        assert result["success"] is True
        assert result["method"] == "parallel"
        assert result["legs_filled"] == "3/3"
        assert "parallel_result" in result

        # Verify execution result details
        parallel_result = result["parallel_result"]
        assert parallel_result.success is True
        assert parallel_result.all_legs_filled is True
        assert parallel_result.total_execution_time > 0

    @pytest.mark.asyncio
    async def test_partial_fill_with_rollback_flow(self, integration_setup):
        """Test partial fill detection and rollback integration"""

        integrator = await create_parallel_integrator(**integration_setup)
        opportunity = create_profitable_opportunity("SPY")

        # Mock partial fill (only stock fills)
        mock_partial_fill(integration_setup["ib"], filled_legs=["stock"])

        result = await integrator.execute_opportunity(opportunity)

        # Should have attempted rollback
        assert result["success"] is False
        assert result["method"] == "parallel"
        assert "rollback" in result.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_global_lock_coordination(self, integration_setup):
        """Test global lock prevents concurrent executions"""

        integrator1 = await create_parallel_integrator(**integration_setup)
        integrator2 = await create_parallel_integrator(
            **{**integration_setup, "symbol": "QQQ"}
        )

        opportunity1 = create_profitable_opportunity("SPY")
        opportunity2 = create_profitable_opportunity("QQQ")

        # Mock slow execution for first integrator
        mock_slow_execution(integration_setup["ib"], delay=2.0)

        # Start both executions simultaneously
        task1 = asyncio.create_task(integrator1.execute_opportunity(opportunity1))
        await asyncio.sleep(0.1)  # Let first one acquire lock
        task2 = asyncio.create_task(integrator2.execute_opportunity(opportunity2))

        results = await asyncio.gather(task1, task2, return_exceptions=True)

        # One should succeed, one should be blocked
        successes = [r for r in results if isinstance(r, dict) and r.get("success")]
        blocks = [r for r in results if isinstance(r, dict) and not r.get("success")]

        assert len(successes) == 1
        assert len(blocks) == 1

    @pytest.mark.asyncio
    async def test_execution_decision_logic(self, integration_setup):
        """Test parallel vs combo execution decision logic"""

        integrator = await create_parallel_integrator(**integration_setup)

        # High profit opportunity - should choose parallel
        high_profit_opp = create_profitable_opportunity("SPY", profit=0.75)
        use_parallel, reason = await integrator.should_use_parallel_execution(high_profit_opp)
        assert use_parallel is True
        assert "favorable" in reason

        # Low profit opportunity - should avoid parallel
        low_profit_opp = create_profitable_opportunity("SPY", profit=0.10)
        use_parallel, reason = await integrator.should_use_parallel_execution(low_profit_opp)
        assert use_parallel is False
        assert "profit_too_low" in reason

    @pytest.mark.asyncio
    async def test_market_data_integration(self, integration_setup):
        """Test integration with market data feeds"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Setup market data mocks
        setup_market_data_mocks(integration_setup["ib"])

        opportunity = create_market_opportunity("SPY")

        result = await integrator.execute_opportunity(opportunity)

        # Verify market data was used in pricing
        assert result is not None
        # Additional market data specific assertions...

    @pytest.mark.asyncio
    async def test_error_propagation_and_handling(self, integration_setup):
        """Test error handling across integration components"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Mock IB connection error during execution
        integration_setup["ib"].placeOrder.side_effect = ConnectionError("IB disconnected")

        opportunity = create_profitable_opportunity("SPY")

        result = await integrator.execute_opportunity(opportunity)

        assert result["success"] is False
        assert "error" in result
        assert "IB disconnected" in result["error"]

    @pytest.mark.asyncio
    async def test_rollback_limit_enforcement_integration(self, integration_setup):
        """Test rollback limits are enforced across the system"""

        integrator = await create_parallel_integrator(**integration_setup)

        # Force partial fills repeatedly
        mock_partial_fill(integration_setup["ib"], filled_legs=["stock"])

        # Should stop after maximum rollbacks
        for i in range(5):  # Try more than the limit
            opportunity = create_profitable_opportunity("SPY")
            result = await integrator.execute_opportunity(opportunity)

            if i >= 3:  # After global limit reached
                # Should refuse execution
                assert "rollback" in result.get("error", "").lower()
                break

    def test_reporting_integration_with_execution(self, integration_setup):
        """Test execution reporter integration with parallel executor"""

        # This would test that reports are generated properly
        # and contain all expected data from the execution
        pass
```

### 3. Performance Tests - Validate Performance Requirements

#### 3.1 Execution Speed Benchmarks (`test_parallel_performance.py`)

```python
class TestParallelExecutionPerformance:
    """Performance benchmarks for parallel execution"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_execution_speed_benchmark(self):
        """Benchmark: Complete execution under 5 seconds"""

        setup = create_performance_test_setup()
        executor = ParallelLegExecutor(**setup)

        start_time = time.time()

        result = await executor.execute_parallel_arbitrage(
            **create_test_execution_params()
        )

        execution_time = time.time() - start_time

        # Performance requirements
        assert execution_time < 5.0  # Must complete under 5 seconds
        assert execution_time < 2.0  # Target: under 2 seconds for fast fills
        assert result.success is True

    @pytest.mark.performance
    def test_lock_contention_performance(self):
        """Test lock performance under high contention"""

        lock = GlobalExecutionLock()
        results = []

        async def contender(symbol, executor_id):
            start = time.time()
            success = await lock.acquire(symbol, executor_id, timeout=1.0)
            if success:
                await asyncio.sleep(0.1)  # Hold briefly
                lock.release(symbol, executor_id)
            return {"success": success, "time": time.time() - start}

        # 10 concurrent acquisition attempts
        tasks = [contender(f"SYM{i}", f"exec_{i}") for i in range(10)]
        results = asyncio.run(asyncio.gather(*tasks))

        # Performance requirements
        successful = [r for r in results if r["success"]]
        assert len(successful) == 1  # Only one should succeed

        # All attempts should complete quickly (not hang)
        max_time = max(r["time"] for r in results)
        assert max_time < 2.0

    @pytest.mark.performance
    def test_memory_usage_during_execution(self):
        """Monitor memory usage during parallel execution"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run multiple executions
        for i in range(10):
            setup = create_performance_test_setup()
            executor = ParallelLegExecutor(**setup)

            # Mock execution
            asyncio.run(executor.execute_parallel_arbitrage(**create_test_execution_params()))

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory should not increase dramatically
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_symbol_performance(self):
        """Test system performance with multiple symbols"""

        symbols = ["SPY", "QQQ", "IWM", "TSLA", "AAPL"]
        integrators = []

        # Create integrators for each symbol
        for symbol in symbols:
            setup = create_performance_test_setup(symbol=symbol)
            integrator = await create_parallel_integrator(**setup)
            integrators.append(integrator)

        start_time = time.time()

        # Try to execute all simultaneously (lock should serialize)
        tasks = []
        for integrator in integrators:
            opportunity = create_profitable_opportunity(integrator.symbol)
            task = asyncio.create_task(integrator.execute_opportunity(opportunity))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Performance requirements
        successful = len([r for r in results if isinstance(r, dict) and r.get("success")])
        assert successful == 1  # Only one should succeed due to global lock
        assert total_time < 30.0  # All should complete within 30 seconds

    @pytest.mark.performance
    def test_reporter_performance_large_dataset(self):
        """Test reporting performance with large dataset"""

        reporter = ExecutionReporter()

        # Generate large number of execution results
        start_time = time.time()

        for i in range(1000):
            result = create_sample_execution_result(symbol=f"SYM{i}")
            reporter.generate_execution_report(result, level=ReportLevel.SUMMARY)

        total_time = time.time() - start_time

        # Should handle 1000 reports in reasonable time
        assert total_time < 10.0  # Less than 10 seconds

        # Session stats should be accurate
        stats = reporter.get_session_statistics()
        assert stats["total_executions"] == 1000
```

### 4. Edge Case Tests - Handle Unusual Scenarios

#### 4.1 Network and Connection Failures (`test_parallel_edge_cases.py`)

```python
class TestParallelExecutionEdgeCases:
    """Test edge cases and failure scenarios"""

    @pytest.mark.asyncio
    async def test_ib_disconnection_during_execution(self):
        """Test IB disconnection during parallel execution"""

        setup = create_test_setup()
        executor = ParallelLegExecutor(**setup)

        # Mock IB disconnection after first order
        def disconnect_after_first(*args, **kwargs):
            if len(setup["ib"].placeOrder.call_args_list) == 0:
                return create_mock_trade("stock", filled=True)
            else:
                raise ConnectionError("IB connection lost")

        setup["ib"].placeOrder.side_effect = disconnect_after_first

        result = await executor.execute_parallel_arbitrage(**create_test_execution_params())

        assert result.success is False
        assert "connection" in result.error_message.lower()
        assert result.legs_filled == 1  # Only first leg should have filled

    @pytest.mark.asyncio
    async def test_market_closure_during_execution(self):
        """Test market closure scenarios"""

        setup = create_test_setup()
        executor = ParallelLegExecutor(**setup)

        # Mock market closure error
        setup["ib"].placeOrder.side_effect = Exception("Market closed")

        result = await executor.execute_parallel_arbitrage(**create_test_execution_params())

        assert result.success is False
        assert "market closed" in result.error_message.lower()

    def test_zero_liquidity_scenario(self):
        """Test handling of zero liquidity (no fills)"""

        setup = create_test_setup()
        executor = ParallelLegExecutor(**setup)

        # Mock orders that never fill
        setup["ib"].placeOrder.return_value = create_mock_trade("stock", filled=False)

        result = asyncio.run(executor.execute_parallel_arbitrage(**create_test_execution_params()))

        assert result.success is False
        assert result.legs_filled == 0
        assert "timeout" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_extreme_slippage_handling(self):
        """Test handling of extreme slippage scenarios"""

        setup = create_test_setup()
        executor = ParallelLegExecutor(**setup)

        # Mock fills with extreme slippage
        mock_fills_with_extreme_slippage(setup["ib"])

        result = await executor.execute_parallel_arbitrage(**create_test_execution_params())

        # Should detect excessive slippage
        if result.success:
            assert abs(result.total_slippage) < 10.0  # Reasonable slippage limit

    def test_invalid_contract_data(self):
        """Test handling of invalid contract data"""

        setup = create_test_setup()
        executor = ParallelLegExecutor(**setup)

        # Invalid contract (missing required fields)
        invalid_stock = MagicMock(conId=None, symbol=None)

        with pytest.raises(ValueError):
            asyncio.run(executor.execute_parallel_arbitrage(
                stock_contract=invalid_stock,
                call_contract=create_valid_contract("call"),
                put_contract=create_valid_contract("put"),
                stock_price=100.0,
                call_price=8.5,
                put_price=3.25
            ))

    @pytest.mark.asyncio
    async def test_system_resource_exhaustion(self):
        """Test behavior under system resource constraints"""

        # Simulate low memory condition
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value.percent = 95  # 95% memory usage

            setup = create_test_setup()
            executor = ParallelLegExecutor(**setup)

            result = await executor.execute_parallel_arbitrage(**create_test_execution_params())

            # Should handle gracefully (might warn but not crash)
            assert result is not None

    @pytest.mark.asyncio
    async def test_clock_synchronization_issues(self):
        """Test handling of system clock/timing issues"""

        setup = create_test_setup()
        executor = ParallelLegExecutor(**setup)

        # Mock system clock jumping backwards
        with patch('time.time') as mock_time:
            # Simulate clock going backwards during execution
            mock_time.side_effect = [1000.0, 999.0, 1001.0, 1002.0]

            result = await executor.execute_parallel_arbitrage(**create_test_execution_params())

            # Should handle timing anomalies gracefully
            assert result is not None
            assert result.total_execution_time >= 0

    def test_rollback_cascade_failure(self):
        """Test handling of cascading rollback failures"""

        setup = create_test_setup()
        manager = RollbackManager()

        # Force rollback failures
        with patch.object(manager, '_execute_rollback_order', side_effect=Exception("Rollback failed")):

            rollback_id = manager.start_rollback("SPY", [MockLeg("stock", 100.0)], "test")

            # Should handle rollback failure gracefully
            result = asyncio.run(manager.complete_rollback(rollback_id, 0.0, False, "Rollback failed"))

            # Should mark as failed but not crash
            attempt = next(a for a in manager.rollback_attempts if a.rollback_id == rollback_id)
            assert attempt.success is False

    @pytest.mark.asyncio
    async def test_concurrent_global_lock_stress(self):
        """Stress test global lock with many concurrent attempts"""

        lock = await GlobalExecutionLock.get_instance()

        async def hammer_lock(symbol, executor_id, iterations=100):
            successes = 0
            for i in range(iterations):
                if await lock.acquire(symbol, executor_id, timeout=0.01):
                    await asyncio.sleep(0.001)  # Very brief hold
                    lock.release(symbol, executor_id)
                    successes += 1
            return successes

        # 20 concurrent hammers
        tasks = [hammer_lock(f"SYM{i}", f"exec_{i}") for i in range(20)]
        results = await asyncio.gather(*tasks)

        total_successes = sum(results)

        # Should handle high contention without deadlock
        assert total_successes > 0  # At least some should succeed
        assert not lock.is_locked()  # Should end up unlocked

    def test_malformed_execution_data_handling(self):
        """Test handling of malformed or corrupted execution data"""

        reporter = ExecutionReporter()

        # Create malformed execution result
        malformed_result = MagicMock()
        malformed_result.success = "not_a_boolean"  # Wrong type
        malformed_result.symbol = None
        malformed_result.total_execution_time = -1  # Invalid time

        # Should handle gracefully
        report = reporter.generate_execution_report(malformed_result)

        assert report is not None
        assert len(report) > 0  # Should produce some output even with bad data
```

### 5. Test Configuration and Setup

#### 5.1 Test Configuration (`test_config.py`)

```python
# Test configuration constants
TEST_CONFIG = {
    "PARALLEL_TIMEOUT": 1.0,  # Shorter for tests
    "PARTIAL_FILL_TIMEOUT": 0.5,
    "MAX_ROLLBACK_ATTEMPTS": 2,  # Lower for tests
    "MAX_ROLLBACK_ATTEMPTS_PER_SYMBOL": 1
}

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "max_execution_time": 5.0,
    "target_execution_time": 2.0,
    "max_lock_contention_time": 2.0,
    "max_memory_increase_mb": 100,
    "max_report_generation_time": 10.0
}

# Test data generators
def create_profitable_opportunity(symbol, profit=0.75):
    """Create profitable opportunity for testing"""
    # Implementation details...

def create_test_execution_params():
    """Create standard execution parameters"""
    # Implementation details...

def mock_successful_fills(mock_ib):
    """Setup mock IB for successful fills"""
    # Implementation details...

def mock_partial_fill(mock_ib, filled_legs):
    """Setup mock IB for partial fills"""
    # Implementation details...
```

### 6. Performance and Acceptance Criteria

#### 6.1 Execution Performance Requirements
- **Complete Execution Time**: < 5 seconds (target: < 2 seconds)
- **Lock Acquisition Time**: < 100ms under normal load
- **Memory Usage**: < 100MB increase per execution session
- **Fill Rate**: > 95% complete fills within timeout
- **Rollback Rate**: < 5% of executions require rollback

#### 6.2 System Reliability Requirements
- **Error Recovery**: 100% of errors handled gracefully
- **Lock Deadlock**: 0% deadlock occurrence
- **Resource Cleanup**: 100% proper resource cleanup
- **Data Consistency**: 100% accurate execution reporting
- **Concurrent Safety**: 100% thread/async safety

#### 6.3 Integration Test Coverage
- **Component Coverage**: > 95% of integration points tested
- **Error Path Coverage**: > 90% of error scenarios tested
- **Edge Case Coverage**: > 80% of edge cases tested
- **Performance Coverage**: 100% of performance requirements validated

---

**Next Steps**: Begin implementation with Phase 1 (Core Infrastructure) focusing on the GlobalExecutionLock and basic parallel execution framework.
