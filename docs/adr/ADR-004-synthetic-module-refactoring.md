# ADR-004: Refactor Synthetic Module to Match SFR Modular Architecture

## Status
Proposed

## Date
2025-09-21

## Context

The current `modules/Arbitrage/Synthetic.py` is a monolithic file containing 1,980 lines of code that handles all aspects of synthetic options arbitrage strategies. This includes:
- Data models and structures (ExpiryOption, OpportunityScore, GlobalOpportunity)
- Scoring configurations and algorithms
- Global opportunity management
- Execution logic (SynExecutor)
- Strategy orchestration (Syn class)
- Various utility and validation functions

In contrast, the SFR (Synthetic-Free-Risk) module has been successfully refactored into a well-organized modular structure under `modules/Arbitrage/sfr/` with clear separation of concerns across 20+ focused files.

### Current Problems

1. **Maintainability**: The monolithic structure makes it difficult to:
   - Locate specific functionality
   - Make isolated changes without affecting other components
   - Understand component dependencies

2. **Testability**: Testing is challenging due to:
   - Tight coupling between components
   - Difficulty in mocking specific parts
   - Large test files that mirror the monolithic structure

3. **Code Reusability**: Similar functionality exists in both Synthetic and SFR but cannot be easily shared

4. **Performance Optimization**:
   - Difficult to optimize specific components in isolation
   - PyPy JIT optimization works better with smaller, focused modules

5. **Inconsistency**: Different architectural patterns between SFR and Synthetic modules despite similar purposes

## Decision

Refactor `modules/Arbitrage/Synthetic.py` into a modular structure matching the SFR pattern, creating a new `modules/Arbitrage/synthetic/` directory with focused, single-responsibility modules.

## Detailed Design

### 1. Directory Structure

```
modules/Arbitrage/synthetic/
├── __init__.py                    # Module exports and compatibility
├── constants.py                   # Configuration constants and thresholds
├── models.py                      # Data models and structures
├── executor.py                    # SynExecutor class
├── strategy.py                    # Main Syn class
├── global_opportunity_manager.py # GlobalOpportunityManager
├── scoring.py                     # ScoringConfig and scoring logic
├── validation.py                  # Strike and opportunity validation
├── utils.py                       # Utility functions
└── data_collector.py             # Market data collection logic
```

### 2. Module Breakdown

#### 2.1 `__init__.py`
```python
"""
Synthetic arbitrage strategy module.

This module provides a modular implementation of the Synthetic arbitrage strategy
with improved maintainability and reusability.
"""

from .executor import SynExecutor
from .models import ExpiryOption, OpportunityScore, GlobalOpportunity
from .strategy import Syn, contract_ticker
from .scoring import ScoringConfig
from .global_opportunity_manager import GlobalOpportunityManager
from .validation import StrikeValidator

__all__ = [
    "Syn",
    "SynExecutor",
    "ExpiryOption",
    "OpportunityScore",
    "GlobalOpportunity",
    "ScoringConfig",
    "GlobalOpportunityManager",
    "StrikeValidator",
    "contract_ticker",
]
```

#### 2.2 `constants.py`
- Data collection timeouts
- Quality score thresholds
- Price validation thresholds
- Profit thresholds
- Strike and moneyness validation
- Volume thresholds
- Buffer percentages
- PyPy-aware optimizations

#### 2.3 `models.py`
```python
@dataclass classes:
- ExpiryOption
- OpportunityScore
- GlobalOpportunity
- ScoringConfig (base structure only)
- MarketDataSnapshot
- ViabilityResult
```

#### 2.4 `executor.py`
- SynExecutor class (lines 563-1345 from original)
- Execution logic
- Market data handling
- Opportunity calculation
- Vectorized evaluation methods

#### 2.5 `strategy.py`
- Syn class (lines 1346-1980 from original)
- Main scanning orchestration
- Symbol processing
- Strike validation
- Contract qualification

#### 2.6 `global_opportunity_manager.py`
- GlobalOpportunityManager class (lines 175-495)
- Opportunity collection
- Scoring and ranking
- Best opportunity selection
- Cycle management

#### 2.7 `scoring.py`
- ScoringConfig class with factory methods
- Scoring algorithms
- Weight normalization
- Preset configurations (conservative, aggressive, balanced, liquidity-focused)

#### 2.8 `validation.py`
- Strike validation logic
- Opportunity viability checks
- Contract validation
- Market hours validation
- Data quality checks

#### 2.9 `utils.py`
- Helper functions
- Vectorized calculations
- Price calculations
- Time utilities
- Debugging utilities

#### 2.10 `data_collector.py`
- Market data collection
- Ticker management
- Data timeout handling
- Contract ticker dictionary management

### 3. Migration Mapping

| Original Lines | Destination Module | Components |
|---|---|---|
| 1-46 | Multiple | Imports distributed across modules |
| 47-56 | models.py | ExpiryOption |
| 59-67 | models.py | OpportunityScore |
| 70-87 | models.py | GlobalOpportunity |
| 89-173 | scoring.py | ScoringConfig |
| 175-495 | global_opportunity_manager.py | GlobalOpportunityManager |
| 496-561 | utils.py | Helper functions |
| 563-1345 | executor.py | SynExecutor |
| 1346-1980 | strategy.py | Syn class |
| Various | constants.py | Constants and thresholds |
| Various | validation.py | Validation methods |

### 4. Test Structure Updates

#### New Test Files
```
tests/
├── test_synthetic_models.py         # Test data models
├── test_synthetic_scoring.py        # Test scoring algorithms
├── test_synthetic_executor.py       # Test executor logic
├── test_synthetic_validation.py     # Test validation logic
├── test_global_opportunity_manager.py # Test opportunity management
└── (existing tests updated with new imports)
```

#### Existing Test Updates
- `test_synthetic.py` - Update imports, maintain integration tests
- `test_synthetic_integration.py` - Update imports
- `test_global_opportunity_lifecycle.py` - Update imports
- `test_global_opportunity_selection.py` - Update imports
- `test_scoring_algorithms.py` - Update imports
- `test_commands_option_comprehensive.py` - Update imports

### 5. Implementation Plan

#### Phase 1: Setup (Low Risk)
1. Create `modules/Arbitrage/synthetic/` directory
2. Create all empty module files with headers
3. Create `__init__.py` with planned exports

#### Phase 2: Migration (Medium Risk)
1. Extract and migrate constants → `constants.py`
2. Extract and migrate models → `models.py`
3. Extract and migrate scoring → `scoring.py`
4. Extract and migrate validation → `validation.py`
5. Extract and migrate utils → `utils.py`
6. Extract and migrate GlobalOpportunityManager → `global_opportunity_manager.py`
7. Extract and migrate SynExecutor → `executor.py`
8. Extract and migrate Syn class → `strategy.py`

#### Phase 3: Integration (Medium Risk)
1. Update internal imports in new modules
2. Implement `__init__.py` with proper exports
3. Create backwards compatibility wrapper in old `Synthetic.py`

#### Phase 4: Testing (Low Risk)
1. Run existing test suite to verify compatibility
2. Create new unit tests for individual modules
3. Test with both CPython and PyPy
4. Run performance benchmarks

#### Phase 5: Cleanup (Low Risk)
1. Update all dependent files to use new imports
2. Update documentation
3. Add deprecation warnings to old module
4. Update CLAUDE.md

### 6. Backwards Compatibility Strategy

Create a compatibility layer in `modules/Arbitrage/Synthetic.py`:

```python
"""
DEPRECATED: This file is maintained for backwards compatibility only.
Please use modules.Arbitrage.synthetic instead.

This compatibility layer will be removed in version 2.0.0
"""

import warnings

warnings.warn(
    "modules.Arbitrage.Synthetic is deprecated. "
    "Please update your imports to use modules.Arbitrage.synthetic instead. "
    "This compatibility layer will be removed in version 2.0.0",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all components from the new module structure
from modules.Arbitrage.synthetic import *

# Maintain module-level variables for compatibility
from modules.Arbitrage.synthetic import contract_ticker, strike_cache, CACHE_TTL

# Maintain test helper functions
from modules.Arbitrage.synthetic.utils import (
    test_global_opportunity_scoring,
    create_syn_with_config,
    get_symbol_contract_count,
    debug_contract_ticker_state
)
```

### 7. Import Path Updates

#### Before:
```python
from modules.Arbitrage.Synthetic import Syn, ScoringConfig, GlobalOpportunityManager
```

#### After:
```python
from modules.Arbitrage.synthetic import Syn, ScoringConfig, GlobalOpportunityManager
```

Note: During transition, both import paths will work due to the compatibility layer.

## Consequences

### Positive

1. **Improved Maintainability**
   - Clear separation of concerns
   - Easier to locate and modify specific functionality
   - Reduced cognitive load when working on individual components

2. **Enhanced Testability**
   - Unit tests can target specific modules
   - Easier to mock dependencies
   - Faster test execution

3. **Better Code Reusability**
   - Shared utilities between SFR and Synthetic
   - Common validation logic can be extracted
   - Consistent patterns across arbitrage strategies

4. **Performance Benefits**
   - PyPy JIT optimization works better with smaller modules
   - Easier to profile and optimize individual components
   - Potential for lazy loading of optional components

5. **Architectural Consistency**
   - Matches SFR module structure
   - Consistent patterns for future arbitrage strategies
   - Easier onboarding for new developers

6. **Development Velocity**
   - Multiple developers can work on different modules simultaneously
   - Reduced merge conflicts
   - Clearer code review process

### Negative

1. **Migration Effort**
   - Time required for refactoring (estimated 4-6 hours)
   - Risk of introducing bugs during migration
   - Need to update all dependent code

2. **Temporary Complexity**
   - Maintaining backwards compatibility layer
   - Two import paths during transition period
   - Additional testing burden

3. **Documentation Updates**
   - Need to update all documentation
   - Update developer guides
   - Update CLAUDE.md instructions

### Neutral

1. **File Count Increase**
   - From 1 file to ~10 files
   - More navigation required
   - IDE project tree becomes larger

2. **Import Complexity**
   - More import statements needed
   - Need to know which module contains what
   - Potential for circular import issues (mitigated by careful design)

## Risk Mitigation

1. **Version Control**
   - Create feature branch for refactoring
   - Commit after each successful phase
   - Enable easy rollback if needed

2. **Testing Strategy**
   - Run full test suite after each migration step
   - Create integration tests for backwards compatibility
   - Test with both CPython and PyPy

3. **Gradual Rollout**
   - Maintain backwards compatibility for 2-3 releases
   - Clear deprecation warnings with migration instructions
   - Monitor for issues in production

4. **Code Review**
   - Peer review of module structure
   - Review of migration completeness
   - Verification of test coverage

## Metrics for Success

1. **Code Quality Metrics**
   - Reduced average file size (from 1980 lines to <500 lines per file)
   - Improved code coverage (target >80% per module)
   - Reduced cyclomatic complexity per module

2. **Performance Metrics**
   - Maintain or improve execution speed
   - Reduced memory usage through better module loading
   - Better PyPy JIT compilation times

3. **Developer Experience**
   - Faster navigation to specific functionality
   - Reduced time to implement new features
   - Easier debugging and troubleshooting

## References

- SFR Module Structure: `modules/Arbitrage/sfr/`
- Original Synthetic Module: `modules/Arbitrage/Synthetic.py`
- Related ADRs:
  - ADR-003: Parallel Execution Strategy
  - SFR_IMPROVEMENTS.md

## Appendix A: File Size Comparison

| Module | Current (lines) | Proposed (lines) |
|---|---|---|
| Synthetic.py | 1,980 | ~50 (compatibility) |
| synthetic/__init__.py | - | ~30 |
| synthetic/constants.py | - | ~150 |
| synthetic/models.py | - | ~200 |
| synthetic/executor.py | - | ~450 |
| synthetic/strategy.py | - | ~400 |
| synthetic/global_opportunity_manager.py | - | ~320 |
| synthetic/scoring.py | - | ~180 |
| synthetic/validation.py | - | ~150 |
| synthetic/utils.py | - | ~120 |
| **Total** | **1,980** | **~2,050** |

## Appendix B: Testing Strategy

### Unit Test Coverage Goals

| Module | Target Coverage | Priority |
|---|---|---|
| models.py | 95% | High |
| scoring.py | 90% | High |
| validation.py | 95% | High |
| global_opportunity_manager.py | 85% | High |
| executor.py | 80% | Medium |
| strategy.py | 75% | Medium |
| utils.py | 90% | Low |
| constants.py | N/A | N/A |

### Integration Test Scenarios

1. End-to-end synthetic arbitrage scan
2. Global opportunity selection across multiple symbols
3. Scoring configuration validation
4. Backwards compatibility verification
5. Performance benchmarks (CPython vs PyPy)

## Decision

**Approved for Implementation**: This refactoring will significantly improve code maintainability, testability, and consistency across the arbitrage strategy modules. The benefits outweigh the migration effort, and the risk is manageable with proper testing and backwards compatibility.

## Next Steps

1. Create feature branch: `feature/synthetic-modular-refactor`
2. Implement Phase 1 (Setup)
3. Proceed with incremental migration
4. Update tests progressively
5. Document changes in CHANGELOG.md
6. Create migration guide for developers
