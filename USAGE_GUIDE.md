# AVBVD Trading Strategy - Usage Guide

## New Features

### 1. Configurable Backtest Years

The `BACKTEST_YEARS` variable (line 93) allows you to control which years to backtest:

1. Backtest specific year: BACKTEST_YEARS = 2020
2. Backtest list of specific years: BACKTEST_YEARS = [2020, 2021, 2022, 2023]  # Backtest these 4 years
3. Backtest most recent 10 years = BACKTEST_YEARS = NONE

#####################################################################################################

#### Default Behavior (Most Recent 10 Years)
```python
BACKTEST_YEARS = None  # Backtests 2014-2024
```

#### Single Year
```python
BACKTEST_YEARS = 2024  # Only backtest 2024
```

#### List of Specific Years
```python
BACKTEST_YEARS = [2020, 2021, 2022, 2023]  # Backtest these 4 years
```

#### Year Range (Start, End)
```python
BACKTEST_YEARS = (2000, 2024)  # Backtest all years from 2000 to 2024
```

```python
BACKTEST_YEARS = (2018, 2022)  # Backtest 2018 through 2022
```

### 2. Summary Text File

After running the backtest, a comprehensive `summary.txt` file is automatically generated in the `output` folder containing:

- **Configuration Parameters**: All strategy settings used
- **Years Backtested**: List of years processed
- **Yearly Results Table**: Quick overview of each year's performance
- **Overall Statistics**: Average return, median return, total trades, win rate, etc.
- **Best & Worst Years**: Highlighted performance extremes
- **Cumulative Performance**: Compounded returns across all years
- **Detailed Trade Statistics**: Year-by-year breakdown of all metrics

## Example Workflows

### For Presentation (Recent Performance)
```python
# Show only the most recent 10 years (default)
BACKTEST_YEARS = None
```

### For Testing a Specific Year
```python
# Focus on a single year to analyze in detail
BACKTEST_YEARS = 2024
```

### For Historical Analysis
```python
# Analyze performance across the full dataset
BACKTEST_YEARS = (2000, 2024)
```

### For Crisis Period Analysis
```python
# Study performance during financial crisis
BACKTEST_YEARS = [2007, 2008, 2009]
```

## Running the Backtest

1. **Configure** the `BACKTEST_YEARS` variable in the configuration section
2. **Run** the script: `python avbvd_trading_strategy.py`
3. **Check outputs** in the `output` folder:
   - `trades_YEAR.csv` - CSV file for each year
   - `trades_YEAR.xlsx` - Excel workbook for each year
   - `yearly_summary.csv` - Comparison across all years
   - `summary.txt` - **NEW!** Comprehensive text report
   - `charts_YEAR/` - Chart images for each year

## Output Files

### summary.txt Structure
```
======================================================================
ADAPTIVE VOLATILITY BREAKOUT WITH VOLUME DIVERGENCE (AVBVD)
BACKTESTING SUMMARY REPORT
======================================================================

CONFIGURATION PARAMETERS:
----------------------------------------------------------------------
Starting Capital:           $100.00
Position Sizing Method:     volatility_adjusted
Stop Loss:                  -8.0%
Take Profit:                15.0%
...

YEARLY RESULTS:
----------------------------------------------------------------------
Year     Return %     Trades     Win Rate     Final Value
----------------------------------------------------------------------
2014      12.50%      45         55.56%       $112.50
2015       8.20%      38         52.63%       $108.20
...

OVERALL STATISTICS:
----------------------------------------------------------------------
Total Years Processed:      11
Average Annual Return:      10.50%
Median Annual Return:       9.80%
...

CUMULATIVE PERFORMANCE (Compounded):
----------------------------------------------------------------------
Starting Capital:           $100.00
Final Value:                $285.43
Cumulative Return:          185.43%
Annualized Return:          10.25%
...
```

## Tips

- **For quick tests**: Use a single year like `BACKTEST_YEARS = 2024`
- **For presentations**: Use default (most recent 10 years)
- **For full analysis**: Use `BACKTEST_YEARS = (2000, 2024)`
- **Check summary.txt**: Quick way to see all results without opening Excel files
