"""
ADAPTIVE VOLATILITY BREAKOUT WITH VOLUME DIVERGENCE (AVBVD) TRADING STRATEGY
=============================================================================

EDUCATIONAL DISCLAIMER:
This backtesting system is for educational purposes ONLY. It demonstrates Level 3
Programming concepts where English prompts generate code. This is NOT intended for
real trading. Even the best hedge funds struggle to consistently beat markets.
Past performance does not guarantee future results. DO NOT use this for actual trading.

PROMPT REQUIREMENTS:
Create a comprehensive Python stock trading backtesting system that implements the
"Adaptive Volatility Breakout with Volume Divergence (AVBVD)" strategy for educational
assessment worth 25% of course grade.

KEY FEATURES:
1. Single-threaded trading: Only ONE position at a time
2. Multi-year backtesting: Process ALL years 2000-2024 separately
3. Four unique strategy components:
   - Dynamic Volatility Channels
   - Volume-Price Momentum Divergence
   - Gap Sentiment Analysis
   - Intraday Range Efficiency
4. Comprehensive risk management with stop-loss, take-profit, holding periods
5. Detailed outputs: CSV, Excel, charts for each year
6. Configurable parameters for strategy tuning

Author: Generated via Claude Code (AI-Assisted)
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from datetime import datetime
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from typing import Dict, List, Tuple, Optional
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Portfolio Configuration
STARTING_CAPITAL = 100.00
POSITION_SIZE_METHOD = 'volatility_adjusted'  # 'equal_weight', 'volatility_adjusted', 'all_in'
MAX_POSITION_SIZE = 1  # 100% of portfolio
TRANSACTION_COST_PERCENT = 0  # 0%
ALLOW_FRACTIONAL_SHARES = True

# Volatility Parameters
VOLATILITY_LOOKBACK = 20
VOLATILITY_THRESHOLD = 1.8
VOLATILITY_CHANNEL_WIDTH = 2.0

# Volume Parameters
VOLUME_VELOCITY_PERIOD = 10
VOLUME_DIVERGENCE_THRESHOLD = 0.25
VOLUME_ACCELERATION_THRESHOLD = 0.30

# Gap Parameters
GAP_MOMENTUM_WINDOW = 5
GAP_EXTENSION_THRESHOLD = 0.4
POSITIVE_GAP_THRESHOLD = 0.01

# Efficiency Parameters
EFFICIENCY_THRESHOLD = 0.65
EFFICIENCY_WINDOW = 3
INCREASING_EFFICIENCY_REQUIRED = True

# Risk Management
STOP_LOSS_PERCENT = -0.08
TAKE_PROFIT_PERCENT = 0.15
MIN_HOLDING_PERIOD = 5
MAX_HOLDING_PERIOD = 30

# Trading Rules
POSITION_CHANNEL_THRESHOLD = 0.25  # Buy in lower 25%, sell in upper 25%
REQUIRE_VOLUME_CONFIRMATION = True
REQUIRE_GAP_CONFIRMATION = True
REQUIRE_EFFICIENCY_CONFIRMATION = True

# Backtest Year Configuration
# Set to None to use default (most recent 10 years: 2014-2024)
# Set to a single year (e.g., 2024) to backtest only that year
# Set to a list of years (e.g., [2020, 2021, 2022]) to backtest specific years
# Set to a tuple (start, end) (e.g., (2000, 2024)) to backtest a range
BACKTEST_YEARS = None  # None = default to most recent 10 years (2014-2024)

# Output Configuration
OUTPUT_FOLDER = "output"
CHARTS_FOLDER_PREFIX = "output/charts"
MAX_CHARTS_PER_YEAR = 20

# Data Configuration
STOCKS_FOLDER = "stocks"

# =============================================================================
# STRATEGY IMPLEMENTATION
# =============================================================================

class AVBVDStrategy:
    """
    Adaptive Volatility Breakout with Volume Divergence Strategy

    Implements four unique components:
    1. Dynamic Volatility Channels - Measures expansion/contraction of volatility
    2. Volume-Price Momentum Divergence - Detects accumulation/distribution
    3. Gap Sentiment Analysis - Analyzes overnight gaps and their behavior
    4. Intraday Range Efficiency - Measures quality of price movements
    """

    def __init__(self):
        """Initialize the strategy with configuration parameters"""
        self.volatility_lookback = VOLATILITY_LOOKBACK
        self.volatility_threshold = VOLATILITY_THRESHOLD
        self.volume_velocity_period = VOLUME_VELOCITY_PERIOD
        self.gap_momentum_window = GAP_MOMENTUM_WINDOW
        self.efficiency_window = EFFICIENCY_WINDOW

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all four indicator components for the strategy

        Args:
            df: DataFrame with columns [Date, Open, High, Low, Close, Volume, Ticker]

        Returns:
            DataFrame with all calculated indicators
        """
        df = df.copy()
        df = df.sort_values('Date').reset_index(drop=True)

        # Component A: Dynamic Volatility Channels
        df = self._calculate_volatility_channels(df)

        # Component B: Volume-Price Momentum Divergence
        df = self._calculate_volume_divergence(df)

        # Component C: Gap Sentiment Analysis
        df = self._calculate_gap_sentiment(df)

        # Component D: Intraday Range Efficiency
        df = self._calculate_range_efficiency(df)

        return df

    def _calculate_volatility_channels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Component A: Dynamic Volatility Channels (30% weight)

        Calculates:
        - True Range: max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
        - Average True Range (ATR) over lookback period
        - True Range Expansion Ratio = TrueRange / ATR
        - Volatility Channels: Upper/Lower bands around price
        - Channel Position: Where price sits within the channel (0-1)
        """
        df = df.copy()

        # Calculate True Range
        df['PrevClose'] = df['Close'].shift(1)
        df['HL'] = df['High'] - df['Low']
        df['HC'] = np.abs(df['High'] - df['PrevClose'])
        df['LC'] = np.abs(df['Low'] - df['PrevClose'])
        df['TrueRange'] = df[['HL', 'HC', 'LC']].max(axis=1)

        # Calculate Average True Range (ATR)
        df['ATR'] = df['TrueRange'].rolling(window=self.volatility_lookback, min_periods=1).mean()

        # True Range Expansion Ratio (avoid division by zero)
        df['TR_Expansion_Ratio'] = np.where(
            df['ATR'] > 0,
            df['TrueRange'] / df['ATR'],
            1.0
        )

        # Volatility signals
        df['Volatility_Expanding'] = df['TR_Expansion_Ratio'] > self.volatility_threshold
        df['Volatility_Contracting'] = df['TR_Expansion_Ratio'] < (1 / self.volatility_threshold)

        # Create volatility channels
        df['Upper_Channel'] = df['Close'] + (df['ATR'] * VOLATILITY_CHANNEL_WIDTH)
        df['Lower_Channel'] = df['Close'] - (df['ATR'] * VOLATILITY_CHANNEL_WIDTH)

        # Calculate channel position (0 to 1 scale)
        channel_range = df['Upper_Channel'] - df['Lower_Channel']
        df['Channel_Position'] = np.where(
            channel_range > 0,
            (df['Close'] - df['Lower_Channel']) / channel_range,
            0.5
        )

        # Clip to 0-1 range
        df['Channel_Position'] = df['Channel_Position'].clip(0, 1)

        return df

    def _calculate_volume_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Component B: Volume-Price Momentum Divergence (30% weight)

        Calculates:
        - Volume Velocity: Percent change in volume over period
        - Price Momentum: Percent change in price over same period
        - Volume Acceleration: Rate of change of volume velocity
        - Detects ACCUMULATION: Volume up, price flat/down
        - Detects DISTRIBUTION: Volume down, price up
        """
        df = df.copy()

        # Volume Velocity (percent change over period)
        df['Volume_Shifted'] = df['Volume'].shift(self.volume_velocity_period)
        df['Volume_Velocity'] = np.where(
            df['Volume_Shifted'] > 0,
            (df['Volume'] - df['Volume_Shifted']) / df['Volume_Shifted'],
            0.0
        )

        # Price Momentum (percent change over same period)
        df['Price_Shifted'] = df['Close'].shift(self.volume_velocity_period)
        df['Price_Momentum'] = np.where(
            df['Price_Shifted'] > 0,
            (df['Close'] - df['Price_Shifted']) / df['Price_Shifted'],
            0.0
        )

        # Volume Acceleration (rate of change of volume velocity)
        df['Volume_Velocity_Prev'] = df['Volume_Velocity'].shift(1)
        df['Volume_Acceleration'] = df['Volume_Velocity'] - df['Volume_Velocity_Prev']

        # Accumulation Signal: Volume up significantly, Price flat/down
        df['Accumulation'] = (
            (df['Volume_Velocity'] > VOLUME_DIVERGENCE_THRESHOLD) &
            (df['Price_Momentum'] < VOLUME_DIVERGENCE_THRESHOLD)
        )

        # Distribution Signal: Volume down significantly, Price up
        df['Distribution'] = (
            (df['Volume_Velocity'] < -VOLUME_DIVERGENCE_THRESHOLD) &
            (df['Price_Momentum'] > VOLUME_DIVERGENCE_THRESHOLD)
        )

        return df

    def _calculate_gap_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Component C: Gap Sentiment Analysis (30% weight)

        Calculates:
        - Overnight Gap: (Open - Previous Close) / Previous Close
        - Gap Fill Ratio: How much of the gap was filled during the day
        - Gap Momentum Score: Weighted average of recent gaps
        - Gap Extension: Gap continues in same direction
        - Gap Reversal: Gap fills and reverses
        """
        df = df.copy()

        # Overnight Gap
        df['Overnight_Gap'] = np.where(
            df['PrevClose'] > 0,
            (df['Open'] - df['PrevClose']) / df['PrevClose'],
            0.0
        )

        # Gap Fill Ratio (handle division by zero)
        gap_size = df['Open'] - df['PrevClose']
        df['Gap_Fill_Ratio'] = np.where(
            np.abs(gap_size) > 0,
            (df['Close'] - df['Open']) / gap_size,
            0.0
        )

        # Gap Momentum Score: Weighted average of recent gaps (more recent = higher weight)
        weights = np.arange(1, self.gap_momentum_window + 1)
        weights = weights / weights.sum()  # Normalize

        gap_momentum_values = []
        for i in range(len(df)):
            if i < self.gap_momentum_window - 1:
                # Not enough history, use simple mean
                gap_momentum_values.append(df['Overnight_Gap'].iloc[:i+1].mean())
            else:
                # Weighted average of recent gaps
                recent_gaps = df['Overnight_Gap'].iloc[i-self.gap_momentum_window+1:i+1].values
                weighted_gap = np.sum(recent_gaps * weights)
                gap_momentum_values.append(weighted_gap)

        df['Gap_Momentum'] = gap_momentum_values

        # Gap signals
        df['Gap_Extension'] = df['Gap_Fill_Ratio'] > GAP_EXTENSION_THRESHOLD
        df['Gap_Reversal'] = df['Gap_Fill_Ratio'] < -GAP_EXTENSION_THRESHOLD
        df['Positive_Gap_Momentum'] = df['Gap_Momentum'] > POSITIVE_GAP_THRESHOLD
        df['Negative_Gap_Momentum'] = df['Gap_Momentum'] < -POSITIVE_GAP_THRESHOLD

        return df

    def _calculate_range_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Component D: Intraday Range Efficiency (30% weight)

        Calculates:
        - Daily Range: High - Low
        - Directional Move: abs(Close - Open)
        - Range Efficiency: Directional Move / Daily Range (0-1)
        - Average Efficiency over window
        - Measures "quality" of moves: high = trending, low = choppy
        """
        df = df.copy()

        # Daily Range
        df['Daily_Range'] = df['High'] - df['Low']

        # Directional Move
        df['Directional_Move'] = np.abs(df['Close'] - df['Open'])

        # Range Efficiency (0 to 1, handle division by zero)
        df['Range_Efficiency'] = np.where(
            df['Daily_Range'] > 0,
            df['Directional_Move'] / df['Daily_Range'],
            0.0
        )

        # Average Efficiency over window
        df['Avg_Efficiency'] = df['Range_Efficiency'].rolling(
            window=self.efficiency_window,
            min_periods=1
        ).mean()

        # Efficiency signals
        df['High_Efficiency'] = df['Avg_Efficiency'] > EFFICIENCY_THRESHOLD

        # Efficiency Increasing
        df['Avg_Efficiency_Prev'] = df['Avg_Efficiency'].shift(1)
        df['Efficiency_Increasing'] = df['Avg_Efficiency'] > df['Avg_Efficiency_Prev']

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate BUY and SELL signals based on the four components

        BUY SIGNALS - Require at least 3 of 5 conditions:
        1. Volatility expanding
        2. Accumulation detected
        3. Positive gap momentum WITH gap extension
        4. High efficiency AND efficiency increasing
        5. Price in lower 25% of volatility channel

        SELL SIGNALS - Require at least 3 of 5 conditions:
        1. Volatility contracting
        2. Distribution detected
        3. Negative gap momentum OR gap reversal
        4. Low efficiency OR efficiency decreasing
        5. Price in upper 25% of volatility channel
        """
        df = df.copy()

        # Buy conditions
        buy_cond_1 = df['Volatility_Expanding']
        buy_cond_2 = df['Accumulation']
        buy_cond_3 = df['Positive_Gap_Momentum'] & df['Gap_Extension']
        buy_cond_4 = df['High_Efficiency'] & df['Efficiency_Increasing']
        buy_cond_5 = df['Channel_Position'] < POSITION_CHANNEL_THRESHOLD

        # Count buy conditions met
        df['Buy_Conditions_Met'] = (
            buy_cond_1.astype(int) +
            buy_cond_2.astype(int) +
            buy_cond_3.astype(int) +
            buy_cond_4.astype(int) +
            buy_cond_5.astype(int)
        )

        # Buy signal when at least 3 conditions met
        df['Buy_Signal'] = df['Buy_Conditions_Met'] >= 3

        # Sell conditions
        sell_cond_1 = df['Volatility_Contracting']
        sell_cond_2 = df['Distribution']
        sell_cond_3 = df['Negative_Gap_Momentum'] | df['Gap_Reversal']
        sell_cond_4 = (~df['High_Efficiency']) | (~df['Efficiency_Increasing'])
        sell_cond_5 = df['Channel_Position'] > (1 - POSITION_CHANNEL_THRESHOLD)

        # Count sell conditions met
        df['Sell_Conditions_Met'] = (
            sell_cond_1.astype(int) +
            sell_cond_2.astype(int) +
            sell_cond_3.astype(int) +
            sell_cond_4.astype(int) +
            sell_cond_5.astype(int)
        )

        # Sell signal when at least 3 conditions met
        df['Sell_Signal'] = df['Sell_Conditions_Met'] >= 3

        return df


# =============================================================================
# PORTFOLIO MANAGEMENT
# =============================================================================

class PortfolioManager:
    """
    Manages portfolio cash, positions, and trade execution

    Features:
    - Single-threaded: Only ONE position at a time
    - Position sizing: equal weight, volatility adjusted, or all-in
    - Transaction costs on both buy and sell
    - Fractional shares support
    """

    def __init__(self, starting_capital: float = STARTING_CAPITAL):
        """
        Initialize portfolio with starting capital

        Args:
            starting_capital: Initial cash amount
        """
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.current_position = None  # Dict with position details or None
        self.portfolio_value = starting_capital
        self.portfolio_history = []

    def has_position(self) -> bool:
        """Check if portfolio currently holds a position"""
        return self.current_position is not None

    def calculate_position_size(self, price: float, volatility: float,
                               method: str = POSITION_SIZE_METHOD) -> float:
        """
        Calculate position size based on selected method

        Args:
            price: Current stock price
            volatility: Stock volatility (ATR)
            method: 'equal_weight', 'volatility_adjusted', or 'all_in'

        Returns:
            Dollar amount to invest
        """
        if method == 'all_in':
            return self.cash

        elif method == 'equal_weight':
            return self.cash * MAX_POSITION_SIZE

        elif method == 'volatility_adjusted':
            # Inverse volatility weighting: higher volatility = smaller position
            if volatility <= 0:
                return self.cash * MAX_POSITION_SIZE

            # Normalize volatility to position size
            # Use a reference volatility to scale (e.g., $2 ATR = full position)
            reference_volatility = 2.0
            volatility_factor = reference_volatility / max(volatility, 0.01)
            volatility_factor = np.clip(volatility_factor, 0.5, 2.0)  # Limit range

            position_size = self.cash * MAX_POSITION_SIZE * volatility_factor
            return min(position_size, self.cash)

        else:
            return self.cash * MAX_POSITION_SIZE

    def buy(self, ticker: str, date: pd.Timestamp, price: float,
            volatility: float) -> Optional[Dict]:
        """
        Execute a buy order

        Args:
            ticker: Stock ticker symbol
            date: Trade date
            price: Buy price
            volatility: Current ATR for position sizing

        Returns:
            Dict with trade details or None if buy failed
        """
        if self.has_position():
            return None  # Already holding a position

        if price <= 0:
            return None  # Invalid price

        # Calculate position size
        position_value = self.calculate_position_size(price, volatility)

        # Ensure we have enough cash
        if position_value > self.cash:
            position_value = self.cash

        # Calculate transaction cost
        transaction_cost = position_value * TRANSACTION_COST_PERCENT
        available_for_shares = position_value - transaction_cost

        if available_for_shares <= 0:
            return None  # Not enough for transaction costs

        # Calculate shares
        shares = available_for_shares / price

        if not ALLOW_FRACTIONAL_SHARES:
            shares = int(shares)
            if shares == 0:
                return None  # Not enough for even 1 share

        # Final buy amount
        buy_amount = shares * price
        total_cost = buy_amount + transaction_cost

        # Execute buy
        self.cash -= total_cost

        self.current_position = {
            'ticker': ticker,
            'buy_date': date,
            'buy_price': price,
            'shares': shares,
            'buy_amount': buy_amount,
            'buy_transaction_cost': transaction_cost,
            'holding_days': 0
        }

        return self.current_position.copy()

    def sell(self, date: pd.Timestamp, price: float,
             reason: str = 'Sell Signal') -> Optional[Dict]:
        """
        Execute a sell order for current position

        Args:
            date: Trade date
            price: Sell price
            reason: Exit reason

        Returns:
            Dict with complete trade details or None if no position
        """
        if not self.has_position():
            return None

        if price <= 0:
            return None

        position = self.current_position
        shares = position['shares']

        # Calculate sell proceeds
        sell_amount = shares * price
        transaction_cost = sell_amount * TRANSACTION_COST_PERCENT
        net_proceeds = sell_amount - transaction_cost

        # Update cash
        self.cash += net_proceeds

        # Calculate profit
        total_transaction_cost = position['buy_transaction_cost'] + transaction_cost
        profit = net_proceeds - position['buy_amount']
        profit_percent = (profit / position['buy_amount']) * 100 if position['buy_amount'] > 0 else 0

        # Create trade record
        trade = {
            'ticker': position['ticker'],
            'buy_date': position['buy_date'],
            'buy_price': position['buy_price'],
            'shares': shares,
            'buy_amount': position['buy_amount'],
            'sell_date': date,
            'sell_price': price,
            'sell_amount': sell_amount,
            'profit': profit,
            'profit_percent': profit_percent,
            'transaction_cost': total_transaction_cost,
            'exit_reason': reason,
            'holding_days': position['holding_days']
        }

        # Clear position
        self.current_position = None

        return trade

    def update_portfolio_value(self, date: pd.Timestamp, current_price: float = None):
        """
        Update portfolio value and increment holding days

        Args:
            date: Current date
            current_price: Current price of held stock (if any)
        """
        position_value = 0

        if self.has_position() and current_price is not None:
            position_value = self.current_position['shares'] * current_price
            self.current_position['holding_days'] += 1

        self.portfolio_value = self.cash + position_value

        self.portfolio_history.append({
            'date': date,
            'cash': self.cash,
            'position_value': position_value,
            'portfolio_value': self.portfolio_value
        })

    def check_risk_management(self, current_price: float) -> Optional[str]:
        """
        Check if risk management rules trigger an exit

        Args:
            current_price: Current stock price

        Returns:
            Exit reason string if exit triggered, else None
        """
        if not self.has_position():
            return None

        position = self.current_position

        # Calculate current profit percentage
        current_value = position['shares'] * current_price
        profit_percent = (current_value - position['buy_amount']) / position['buy_amount']

        # Check stop loss
        if profit_percent <= STOP_LOSS_PERCENT:
            return 'Stop Loss'

        # Check take profit
        if profit_percent >= TAKE_PROFIT_PERCENT:
            return 'Take Profit'

        # Check maximum holding period
        if position['holding_days'] >= MAX_HOLDING_PERIOD:
            return 'Max Hold'

        return None


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class BacktestEngine:
    """
    Runs backtest for a single year across all stocks

    Features:
    - Single-threaded trading: Only one position at a time
    - Processes stocks sequentially day by day
    - Applies risk management rules
    - Tracks all trades and portfolio history
    """

    def __init__(self, year: int, starting_capital: float = STARTING_CAPITAL):
        """
        Initialize backtest engine for specific year

        Args:
            year: Year to backtest (e.g., 2020)
            starting_capital: Starting cash amount
        """
        self.year = year
        self.strategy = AVBVDStrategy()
        self.portfolio = PortfolioManager(starting_capital)
        self.trades = []

    def run_backtest(self, stock_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run backtest for the year across all stocks

        Args:
            stock_data: Dict mapping ticker to DataFrame

        Returns:
            Dict with backtest results
        """
        print(f"\n{'='*70}")
        print(f"BACKTESTING YEAR {self.year}")
        print(f"{'='*70}")
        print(f"Starting Capital: ${self.portfolio.starting_capital:.2f}")

        # Prepare all stock data for this year
        prepared_data = {}
        for ticker, df in stock_data.items():
            # Filter for this year
            year_data = df[df['Date'].dt.year == self.year].copy()

            if len(year_data) < VOLATILITY_LOOKBACK + VOLUME_VELOCITY_PERIOD:
                continue  # Not enough data

            # Calculate indicators
            try:
                year_data = self.strategy.calculate_indicators(year_data)
                year_data = self.strategy.generate_signals(year_data)
                prepared_data[ticker] = year_data
            except Exception as e:
                print(f"Warning: Failed to process {ticker}: {str(e)}")
                continue

        print(f"Processing {len(prepared_data)} stocks with sufficient data...")

        # Get all unique dates across all stocks
        all_dates = set()
        for df in prepared_data.values():
            all_dates.update(df['Date'].values)
        all_dates = sorted(all_dates)

        # Process day by day
        for idx, current_date in enumerate(all_dates):
            if idx % 20 == 0:  # Progress indicator
                print(f"Processing date {idx+1}/{len(all_dates)}: {pd.Timestamp(current_date).date()}", end='\r')

            # Check if we have a position
            if self.portfolio.has_position():
                # Monitor current position
                ticker = self.portfolio.current_position['ticker']

                if ticker in prepared_data:
                    ticker_data = prepared_data[ticker]
                    today_data = ticker_data[ticker_data['Date'] == current_date]

                    if len(today_data) > 0:
                        row = today_data.iloc[0]
                        current_price = row['Close']

                        # Update portfolio value
                        self.portfolio.update_portfolio_value(current_date, current_price)

                        # Check risk management
                        exit_reason = self.portfolio.check_risk_management(current_price)

                        if exit_reason:
                            # Force exit due to risk management
                            trade = self.portfolio.sell(current_date, current_price, exit_reason)
                            if trade:
                                self.trades.append(trade)

                        elif self.portfolio.current_position['holding_days'] >= MIN_HOLDING_PERIOD:
                            # Check sell signal
                            if row['Sell_Signal']:
                                trade = self.portfolio.sell(current_date, current_price, 'Sell Signal')
                                if trade:
                                    self.trades.append(trade)

            else:
                # Look for buy opportunities
                best_signal = None
                best_score = 0

                for ticker, ticker_data in prepared_data.items():
                    today_data = ticker_data[ticker_data['Date'] == current_date]

                    if len(today_data) > 0:
                        row = today_data.iloc[0]

                        if row['Buy_Signal']:
                            # Score this opportunity
                            score = row['Buy_Conditions_Met']

                            if score > best_score:
                                best_score = score
                                best_signal = {
                                    'ticker': ticker,
                                    'date': current_date,
                                    'price': row['Close'],
                                    'atr': row['ATR']
                                }

                # Execute best buy signal
                if best_signal:
                    trade = self.portfolio.buy(
                        best_signal['ticker'],
                        best_signal['date'],
                        best_signal['price'],
                        best_signal['atr']
                    )

                # Update portfolio value
                self.portfolio.update_portfolio_value(current_date)

        print(f"\nProcessing complete.{' '*30}")

        # Close any open position at year end
        if self.portfolio.has_position():
            ticker = self.portfolio.current_position['ticker']

            if ticker in prepared_data:
                ticker_data = prepared_data[ticker]
                last_price = ticker_data.iloc[-1]['Close']
                last_date = ticker_data.iloc[-1]['Date']

                trade = self.portfolio.sell(last_date, last_price, 'Year End')
                if trade:
                    self.trades.append(trade)

        # Calculate summary statistics
        return self._calculate_summary()

    def _calculate_summary(self) -> Dict:
        """Calculate summary statistics for the backtest"""
        if len(self.trades) == 0:
            return {
                'year': self.year,
                'starting_capital': self.portfolio.starting_capital,
                'final_value': self.portfolio.portfolio_value,
                'total_return': 0,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_profit_pct': 0,
                'total_transaction_costs': 0,
                'best_trade_pct': 0,
                'worst_trade_pct': 0
            }

        trades_df = pd.DataFrame(self.trades)

        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        win_rate = (winning_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0

        total_return = self.portfolio.portfolio_value - self.portfolio.starting_capital
        total_return_pct = (total_return / self.portfolio.starting_capital) * 100

        summary = {
            'year': self.year,
            'starting_capital': self.portfolio.starting_capital,
            'final_value': self.portfolio.portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': len(trades_df),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_profit': trades_df['profit'].mean(),
            'avg_profit_pct': trades_df['profit_percent'].mean(),
            'total_transaction_costs': trades_df['transaction_cost'].sum(),
            'best_trade_pct': trades_df['profit_percent'].max(),
            'worst_trade_pct': trades_df['profit_percent'].min()
        }

        print(f"\n{'='*70}")
        print(f"YEAR {self.year} RESULTS")
        print(f"{'='*70}")
        print(f"Starting Capital:        ${summary['starting_capital']:.2f}")
        print(f"Final Portfolio Value:   ${summary['final_value']:.2f}")
        print(f"Total Return:            ${summary['total_return']:.2f} ({summary['total_return_pct']:.2f}%)")
        print(f"Total Trades:            {summary['total_trades']}")
        print(f"Winning Trades:          {summary['winning_trades']}")
        print(f"Losing Trades:           {summary['losing_trades']}")
        print(f"Win Rate:                {summary['win_rate']:.2f}%")
        print(f"Average Profit:          ${summary['avg_profit']:.2f} ({summary['avg_profit_pct']:.2f}%)")
        print(f"Transaction Costs:       ${summary['total_transaction_costs']:.2f}")
        print(f"Best Trade:              {summary['best_trade_pct']:.2f}%")
        print(f"Worst Trade:             {summary['worst_trade_pct']:.2f}%")

        return summary


# =============================================================================
# DATA LOADING
# =============================================================================

def load_stock_data() -> Dict[str, pd.DataFrame]:
    """
    Load all stock CSV files from the stocks folder

    Returns:
        Dict mapping ticker to DataFrame
    """
    print("\nLoading stock data from 'stocks/' folder...")

    stocks_path = Path(STOCKS_FOLDER)

    if not stocks_path.exists():
        print(f"ERROR: Stocks folder '{STOCKS_FOLDER}' not found!")
        sys.exit(1)

    csv_files = list(stocks_path.glob("*.csv"))

    if len(csv_files) == 0:
        print(f"ERROR: No CSV files found in '{STOCKS_FOLDER}' folder!")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files")

    stock_data = {}

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Skipping {csv_file.name} - missing required columns")
                continue

            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Get ticker from filename or Ticker column
            if 'Ticker' in df.columns:
                ticker = df['Ticker'].iloc[0]
            else:
                ticker = csv_file.stem
                df['Ticker'] = ticker

            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)

            stock_data[ticker] = df

        except Exception as e:
            print(f"Warning: Failed to load {csv_file.name}: {str(e)}")
            continue

    print(f"Successfully loaded {len(stock_data)} stocks")

    return stock_data


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

def save_trades_csv(trades: List[Dict], year: int, output_folder: str = OUTPUT_FOLDER):
    """Save trades to CSV file"""
    if len(trades) == 0:
        print(f"No trades for {year}, skipping CSV output")
        return

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    trades_df = pd.DataFrame(trades)

    # Format columns
    trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date']).dt.date
    trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date']).dt.date

    # Rename columns for output
    output_df = trades_df[[
        'ticker', 'buy_date', 'buy_price', 'shares', 'buy_amount',
        'sell_date', 'sell_price', 'sell_amount', 'profit',
        'profit_percent', 'transaction_cost', 'exit_reason'
    ]].copy()

    output_df.columns = [
        'Ticker', 'Buy Date', 'Buy Price', 'Shares', 'Buy Amount ($)',
        'Sell Date', 'Sell Price', 'Sell Amount ($)', 'Profit ($)',
        'Percentage Profit', 'Transaction Cost ($)', 'Exit Reason'
    ]

    csv_path = output_path / f"trades_{year}.csv"
    output_df.to_csv(csv_path, index=False)
    print(f"Saved trades to {csv_path}")


def save_trades_excel(trades: List[Dict], portfolio_history: List[Dict],
                     summary: Dict, year: int, output_folder: str = OUTPUT_FOLDER):
    """Save trades, summary, and portfolio history to Excel file"""
    if len(trades) == 0:
        print(f"No trades for {year}, skipping Excel output")
        return

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    excel_path = output_path / f"trades_{year}.xlsx"

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Trades
        trades_df = pd.DataFrame(trades)
        trades_df['buy_date'] = pd.to_datetime(trades_df['buy_date']).dt.date
        trades_df['sell_date'] = pd.to_datetime(trades_df['sell_date']).dt.date

        output_df = trades_df[[
            'ticker', 'buy_date', 'buy_price', 'shares', 'buy_amount',
            'sell_date', 'sell_price', 'sell_amount', 'profit',
            'profit_percent', 'transaction_cost', 'exit_reason'
        ]].copy()

        output_df.columns = [
            'Ticker', 'Buy Date', 'Buy Price', 'Shares', 'Buy Amount ($)',
            'Sell Date', 'Sell Price', 'Sell Amount ($)', 'Profit ($)',
            'Percentage Profit', 'Transaction Cost ($)', 'Exit Reason'
        ]

        output_df.to_excel(writer, sheet_name='Trades', index=False)

        # Sheet 2: Summary
        summary_data = {
            'Metric': [
                'Year',
                'Starting Capital',
                'Final Portfolio Value',
                'Total Return ($)',
                'Total Return (%)',
                'Total Trades',
                'Winning Trades',
                'Losing Trades',
                'Win Rate (%)',
                'Average Profit ($)',
                'Average Profit (%)',
                'Transaction Costs',
                'Best Trade (%)',
                'Worst Trade (%)'
            ],
            'Value': [
                summary['year'],
                f"${summary['starting_capital']:.2f}",
                f"${summary['final_value']:.2f}",
                f"${summary['total_return']:.2f}",
                f"{summary['total_return_pct']:.2f}%",
                summary['total_trades'],
                summary['winning_trades'],
                summary['losing_trades'],
                f"{summary['win_rate']:.2f}%",
                f"${summary['avg_profit']:.2f}",
                f"{summary['avg_profit_pct']:.2f}%",
                f"${summary['total_transaction_costs']:.2f}",
                f"{summary['best_trade_pct']:.2f}%",
                f"{summary['worst_trade_pct']:.2f}%"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 3: Portfolio History
        if len(portfolio_history) > 0:
            history_df = pd.DataFrame(portfolio_history)
            history_df['date'] = pd.to_datetime(history_df['date']).dt.date
            history_df.columns = ['Date', 'Cash', 'Position Value', 'Portfolio Value']
            history_df.to_excel(writer, sheet_name='Portfolio History', index=False)

    print(f"Saved Excel workbook to {excel_path}")


def generate_charts(trades: List[Dict], stock_data: Dict[str, pd.DataFrame],
                   year: int, output_folder: str = CHARTS_FOLDER_PREFIX):
    """
    Generate charts for top trades

    Args:
        trades: List of trade dictionaries
        stock_data: Dict mapping ticker to DataFrame
        year: Year being processed
        output_folder: Base folder for chart output
    """
    if len(trades) == 0:
        print(f"No trades for {year}, skipping chart generation")
        return

    # Create output folder
    charts_path = Path(output_folder) / f"charts_{year}"
    charts_path.mkdir(parents=True, exist_ok=True)

    # Sort trades by absolute profit percentage and take top N
    trades_df = pd.DataFrame(trades)
    trades_df['abs_profit_pct'] = trades_df['profit_percent'].abs()
    top_trades = trades_df.nlargest(min(MAX_CHARTS_PER_YEAR, len(trades_df)), 'abs_profit_pct')

    print(f"\nGenerating charts for top {len(top_trades)} trades of {year}...")

    for idx, trade in top_trades.iterrows():
        try:
            ticker = trade['ticker']

            if ticker not in stock_data:
                continue

            # Get stock data for the year
            df = stock_data[ticker]
            year_data = df[df['Date'].dt.year == year].copy()

            if len(year_data) == 0:
                continue

            # Calculate indicators
            strategy = AVBVDStrategy()
            year_data = strategy.calculate_indicators(year_data)
            year_data = strategy.generate_signals(year_data)

            # Create figure with 5 panels
            fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
            fig.suptitle(f"{ticker} - {year} - Profit: {trade['profit_percent']:.2f}%", fontsize=14, fontweight='bold')

            # Panel 1: Price with volatility channels
            ax1 = axes[0]
            ax1.plot(year_data['Date'], year_data['Close'], label='Close Price', color='black', linewidth=1.5)
            ax1.fill_between(year_data['Date'], year_data['Lower_Channel'], year_data['Upper_Channel'],
                            alpha=0.2, color='blue', label='Volatility Channel')

            # Mark buy/sell signals
            buy_signals = year_data[year_data['Buy_Signal']]
            sell_signals = year_data[year_data['Sell_Signal']]
            ax1.scatter(buy_signals['Date'], buy_signals['Close'], marker='^', color='green',
                       s=100, label='Buy Signal', zorder=5)
            ax1.scatter(sell_signals['Date'], sell_signals['Close'], marker='v', color='red',
                       s=100, label='Sell Signal', zorder=5)

            # Mark this specific trade
            buy_date = pd.to_datetime(trade['buy_date'])
            sell_date = pd.to_datetime(trade['sell_date'])
            ax1.axvline(buy_date, color='green', linestyle='--', linewidth=2, alpha=0.7)
            ax1.axvline(sell_date, color='red', linestyle='--', linewidth=2, alpha=0.7)

            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Panel 2: Volume with Volume Velocity
            ax2 = axes[1]
            ax2.bar(year_data['Date'], year_data['Volume'], color='lightblue', alpha=0.6, label='Volume')
            ax2.set_ylabel('Volume', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')

            ax2b = ax2.twinx()
            ax2b.plot(year_data['Date'], year_data['Volume_Velocity'], color='orange',
                     linewidth=1.5, label='Volume Velocity')
            ax2b.axhline(VOLUME_DIVERGENCE_THRESHOLD, color='green', linestyle='--', alpha=0.5)
            ax2b.axhline(-VOLUME_DIVERGENCE_THRESHOLD, color='red', linestyle='--', alpha=0.5)
            ax2b.set_ylabel('Volume Velocity', color='orange')
            ax2b.tick_params(axis='y', labelcolor='orange')

            ax2.legend(loc='upper left', fontsize=8)
            ax2b.legend(loc='upper right', fontsize=8)
            ax2.grid(True, alpha=0.3)

            # Panel 3: Gap Momentum
            ax3 = axes[2]
            ax3.plot(year_data['Date'], year_data['Gap_Momentum'], color='purple', linewidth=1.5)
            ax3.axhline(POSITIVE_GAP_THRESHOLD, color='green', linestyle='--', alpha=0.5)
            ax3.axhline(-POSITIVE_GAP_THRESHOLD, color='red', linestyle='--', alpha=0.5)
            ax3.fill_between(year_data['Date'], 0, year_data['Gap_Momentum'],
                            where=year_data['Gap_Momentum'] > 0, alpha=0.3, color='green')
            ax3.fill_between(year_data['Date'], 0, year_data['Gap_Momentum'],
                            where=year_data['Gap_Momentum'] < 0, alpha=0.3, color='red')
            ax3.set_ylabel('Gap Momentum')
            ax3.grid(True, alpha=0.3)
            ax3.legend(['Gap Momentum'], loc='upper left', fontsize=8)

            # Panel 4: Range Efficiency
            ax4 = axes[3]
            ax4.plot(year_data['Date'], year_data['Avg_Efficiency'], color='brown', linewidth=1.5)
            ax4.axhline(EFFICIENCY_THRESHOLD, color='green', linestyle='--', alpha=0.5,
                       label='Efficiency Threshold')
            ax4.fill_between(year_data['Date'], EFFICIENCY_THRESHOLD, year_data['Avg_Efficiency'],
                            where=year_data['Avg_Efficiency'] > EFFICIENCY_THRESHOLD,
                            alpha=0.3, color='green')
            ax4.set_ylabel('Avg Efficiency')
            ax4.set_ylim(0, 1)
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='upper left', fontsize=8)

            # Panel 5: Position value for this trade
            ax5 = axes[4]
            trade_data = year_data[
                (year_data['Date'] >= buy_date) &
                (year_data['Date'] <= sell_date)
            ].copy()

            if len(trade_data) > 0:
                trade_data['Position_Value'] = trade_data['Close'] * trade['shares']
                ax5.plot(trade_data['Date'], trade_data['Position_Value'], color='darkgreen', linewidth=2)
                ax5.axhline(trade['buy_amount'], color='blue', linestyle='--', alpha=0.5, label='Buy Amount')
                ax5.axhline(trade['sell_amount'], color='red', linestyle='--', alpha=0.5, label='Sell Amount')
                ax5.fill_between(trade_data['Date'], trade['buy_amount'], trade_data['Position_Value'],
                                where=trade_data['Position_Value'] > trade['buy_amount'],
                                alpha=0.3, color='green', label='Profit')
                ax5.fill_between(trade_data['Date'], trade['buy_amount'], trade_data['Position_Value'],
                                where=trade_data['Position_Value'] < trade['buy_amount'],
                                alpha=0.3, color='red', label='Loss')
                ax5.set_ylabel('Position Value ($)')
                ax5.legend(loc='upper left', fontsize=8)

            ax5.set_xlabel('Date')
            ax5.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save chart
            chart_filename = f"{year}_{ticker}_profit_{trade['profit_percent']:.1f}pct.png"
            chart_path = charts_path / chart_filename
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Warning: Failed to generate chart for {ticker}: {str(e)}")
            continue

    print(f"Charts saved to {charts_path}")


def save_summary_txt(yearly_summaries: List[Dict], years: List[int],
                    output_folder: str = OUTPUT_FOLDER):
    """
    Save a comprehensive summary text file with all backtest results

    Args:
        yearly_summaries: List of summary dictionaries for each year
        years: List of years that were backtested
        output_folder: Folder to save the summary file
    """
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    summary_file = output_path / "summary.txt"

    with open(summary_file, 'w') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("ADAPTIVE VOLATILITY BREAKOUT WITH VOLUME DIVERGENCE (AVBVD)\n")
        f.write("BACKTESTING SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("EDUCATIONAL DISCLAIMER:\n")
        f.write("This is for educational purposes ONLY. Not for real trading.\n")
        f.write("Past performance does not guarantee future results.\n")
        f.write("=" * 70 + "\n\n")

        # Configuration Summary
        f.write("CONFIGURATION PARAMETERS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Starting Capital:           ${STARTING_CAPITAL:.2f}\n")
        f.write(f"Position Sizing Method:     {POSITION_SIZE_METHOD}\n")
        f.write(f"Stop Loss:                  {STOP_LOSS_PERCENT*100:.1f}%\n")
        f.write(f"Take Profit:                {TAKE_PROFIT_PERCENT*100:.1f}%\n")
        f.write(f"Min Holding Period:         {MIN_HOLDING_PERIOD} days\n")
        f.write(f"Max Holding Period:         {MAX_HOLDING_PERIOD} days\n")
        f.write(f"Volatility Threshold:       {VOLATILITY_THRESHOLD}\n")
        f.write(f"Volume Divergence Thresh:   {VOLUME_DIVERGENCE_THRESHOLD}\n")
        f.write(f"Efficiency Threshold:       {EFFICIENCY_THRESHOLD}\n")
        f.write("=" * 70 + "\n\n")

        # Years Backtested
        f.write("YEARS BACKTESTED:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Years: {min(years)} to {max(years)}\n")
        f.write(f"Total Years: {len(years)}\n")
        f.write(f"Year List: {years}\n")
        f.write("=" * 70 + "\n\n")

        if len(yearly_summaries) == 0:
            f.write("No trades were executed during the backtest period.\n")
            return

        # Yearly Results Table
        f.write("YEARLY RESULTS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Year':<8} {'Return %':<12} {'Trades':<10} {'Win Rate':<12} {'Final Value':<15}\n")
        f.write("-" * 70 + "\n")

        for summary in yearly_summaries:
            f.write(f"{summary['year']:<8} "
                   f"{summary['total_return_pct']:>10.2f}%  "
                   f"{summary['total_trades']:<10} "
                   f"{summary['win_rate']:>10.2f}%  "
                   f"${summary['final_value']:>12.2f}\n")

        f.write("=" * 70 + "\n\n")

        # Overall Statistics
        summary_df = pd.DataFrame(yearly_summaries)

        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Years Processed:      {len(summary_df)}\n")
        f.write(f"Average Annual Return:      {summary_df['total_return_pct'].mean():.2f}%\n")
        f.write(f"Median Annual Return:       {summary_df['total_return_pct'].median():.2f}%\n")
        f.write(f"Standard Deviation:         {summary_df['total_return_pct'].std():.2f}%\n")
        f.write(f"Total Trades (All Years):   {int(summary_df['total_trades'].sum())}\n")
        f.write(f"Average Win Rate:           {summary_df['win_rate'].mean():.2f}%\n")
        f.write(f"Average Profit Per Trade:   {summary_df['avg_profit_pct'].mean():.2f}%\n")
        f.write("=" * 70 + "\n\n")

        # Best and Worst Years
        best_year = summary_df.loc[summary_df['total_return_pct'].idxmax()]
        worst_year = summary_df.loc[summary_df['total_return_pct'].idxmin()]

        f.write("BEST & WORST YEARS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best Year:    {int(best_year['year'])} "
               f"({best_year['total_return_pct']:.2f}% return, "
               f"{int(best_year['total_trades'])} trades)\n")
        f.write(f"Worst Year:   {int(worst_year['year'])} "
               f"({worst_year['total_return_pct']:.2f}% return, "
               f"{int(worst_year['total_trades'])} trades)\n")
        f.write("=" * 70 + "\n\n")

        # Cumulative Performance
        cumulative_multiplier = 1.0
        for ret in summary_df['total_return_pct']:
            cumulative_multiplier *= (1 + ret / 100)

        cumulative_return = (cumulative_multiplier - 1) * 100
        final_value = STARTING_CAPITAL * cumulative_multiplier

        f.write("CUMULATIVE PERFORMANCE (Compounded):\n")
        f.write("-" * 70 + "\n")
        f.write(f"Starting Capital:           ${STARTING_CAPITAL:.2f}\n")
        f.write(f"Final Value:                ${final_value:.2f}\n")
        f.write(f"Cumulative Return:          {cumulative_return:.2f}%\n")
        f.write(f"Annualized Return:          {((cumulative_multiplier ** (1/len(years))) - 1) * 100:.2f}%\n")
        f.write("=" * 70 + "\n\n")

        # Trade Statistics by Year
        f.write("DETAILED TRADE STATISTICS BY YEAR:\n")
        f.write("-" * 70 + "\n")

        for summary in yearly_summaries:
            f.write(f"\n{summary['year']}:\n")
            f.write(f"  Total Return:          {summary['total_return_pct']:>8.2f}%  "
                   f"(${summary['total_return']:>8.2f})\n")
            f.write(f"  Total Trades:          {summary['total_trades']:>8}\n")
            f.write(f"  Winning Trades:        {summary['winning_trades']:>8}\n")
            f.write(f"  Losing Trades:         {summary['losing_trades']:>8}\n")
            f.write(f"  Win Rate:              {summary['win_rate']:>8.2f}%\n")
            f.write(f"  Avg Profit/Trade:      {summary['avg_profit_pct']:>8.2f}%  "
                   f"(${summary['avg_profit']:>8.2f})\n")
            f.write(f"  Best Trade:            {summary['best_trade_pct']:>8.2f}%\n")
            f.write(f"  Worst Trade:           {summary['worst_trade_pct']:>8.2f}%\n")
            f.write(f"  Transaction Costs:     ${summary['total_transaction_costs']:>8.2f}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"\nSaved summary report to {summary_file}")


# =============================================================================
# MULTI-YEAR BACKTESTING
# =============================================================================

def determine_years_to_backtest(stock_data: Dict[str, pd.DataFrame]) -> List[int]:
    """
    Determine which years to backtest based on BACKTEST_YEARS configuration

    Args:
        stock_data: Dict mapping ticker to DataFrame

    Returns:
        List of years to backtest
    """
    # Get all available years from data
    all_years = set()
    for df in stock_data.values():
        all_years.update(df['Date'].dt.year.unique())

    available_years = sorted(all_years)

    if BACKTEST_YEARS is None:
        # Default: most recent 10 years (2014-2024)
        recent_10_years = [y for y in available_years if y >= 2014 and y <= 2024]
        return recent_10_years if recent_10_years else available_years[-10:]

    elif isinstance(BACKTEST_YEARS, int):
        # Single year
        if BACKTEST_YEARS in available_years:
            return [BACKTEST_YEARS]
        else:
            print(f"Warning: Year {BACKTEST_YEARS} not found in data. No years to process.")
            return []

    elif isinstance(BACKTEST_YEARS, list):
        # List of specific years
        valid_years = [y for y in BACKTEST_YEARS if y in available_years]
        if len(valid_years) < len(BACKTEST_YEARS):
            missing = set(BACKTEST_YEARS) - set(valid_years)
            print(f"Warning: Years {missing} not found in data.")
        return sorted(valid_years)

    elif isinstance(BACKTEST_YEARS, tuple) and len(BACKTEST_YEARS) == 2:
        # Range of years (start, end) inclusive
        start_year, end_year = BACKTEST_YEARS
        range_years = [y for y in available_years if start_year <= y <= end_year]
        return range_years

    else:
        print(f"Warning: Invalid BACKTEST_YEARS format. Using all available years.")
        return available_years


def run_multi_year_backtest():
    """
    Main function to run backtests across configured years

    For each year:
    - Initialize fresh portfolio
    - Run backtest
    - Save outputs (CSV, Excel, charts)
    - Record summary

    After all years:
    - Generate yearly summary comparison
    - Display overall statistics
    - Save summary.txt file
    """
    print("\n" + "="*70)
    print("ADAPTIVE VOLATILITY BREAKOUT WITH VOLUME DIVERGENCE (AVBVD)")
    print("Multi-Year Backtesting System")
    print("="*70)
    print("\nEDUCATIONAL DISCLAIMER:")
    print("This is for educational purposes ONLY. Not for real trading.")
    print("Past performance does not guarantee future results.")
    print("="*70)

    # Load all stock data
    stock_data = load_stock_data()

    # Determine which years to backtest
    years = determine_years_to_backtest(stock_data)

    if not years:
        print("\nNo years to process. Check BACKTEST_YEARS configuration.")
        return

    print(f"\nYears to backtest: {min(years)} to {max(years)}")
    print(f"Total years to process: {len(years)}")
    print(f"Years: {years}")

    # Run backtest for each year
    yearly_summaries = []

    for year in years:
        # Check if any stock has data for this year
        year_has_data = any(
            (df['Date'].dt.year == year).any()
            for df in stock_data.values()
        )

        if not year_has_data:
            print(f"\nSkipping {year} - no data available")
            continue

        # Initialize backtest engine for this year
        engine = BacktestEngine(year, STARTING_CAPITAL)

        # Run backtest
        summary = engine.run_backtest(stock_data)
        yearly_summaries.append(summary)

        # Save outputs
        save_trades_csv(engine.trades, year)
        save_trades_excel(engine.trades, engine.portfolio.portfolio_history, summary, year)
        generate_charts(engine.trades, stock_data, year)

    # Generate yearly summary comparison
    if len(yearly_summaries) > 0:
        summary_df = pd.DataFrame(yearly_summaries)

        summary_df = summary_df[[
            'year', 'starting_capital', 'final_value', 'total_return',
            'total_return_pct', 'total_trades', 'win_rate',
            'avg_profit_pct', 'best_trade_pct', 'worst_trade_pct'
        ]].copy()

        summary_df.columns = [
            'Year', 'Starting Capital', 'Final Portfolio Value', 'Total Return ($)',
            'Total Return (%)', 'Total Trades', 'Win Rate (%)',
            'Avg Profit Per Trade (%)', 'Best Trade (%)', 'Worst Trade (%)'
        ]

        output_path = Path(OUTPUT_FOLDER)
        output_path.mkdir(exist_ok=True)
        summary_path = output_path / "yearly_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*70}")
        print(f"Saved yearly summary to {summary_path}")

        # Display overall statistics
        print(f"\n{'='*70}")
        print("OVERALL STATISTICS (ALL YEARS)")
        print(f"{'='*70}")

        best_year = summary_df.loc[summary_df['Total Return (%)'].idxmax()]
        worst_year = summary_df.loc[summary_df['Total Return (%)'].idxmin()]

        print(f"Total Years Processed:   {len(summary_df)}")
        print(f"Average Annual Return:   {summary_df['Total Return (%)'].mean():.2f}%")
        print(f"Median Annual Return:    {summary_df['Total Return (%)'].median():.2f}%")
        print(f"Best Year:               {int(best_year['Year'])} ({best_year['Total Return (%)']:.2f}%)")
        print(f"Worst Year:              {int(worst_year['Year'])} ({worst_year['Total Return (%)']:.2f}%)")
        print(f"Total Trades (All Years): {int(summary_df['Total Trades'].sum())}")
        print(f"Average Win Rate:        {summary_df['Win Rate (%)'].mean():.2f}%")

        # Calculate cumulative return if compounded
        cumulative_multiplier = 1.0
        for ret in summary_df['Total Return (%)']:
            cumulative_multiplier *= (1 + ret / 100)

        cumulative_return = (cumulative_multiplier - 1) * 100
        final_value = STARTING_CAPITAL * cumulative_multiplier

        print(f"\nIf compounded across all years:")
        print(f"Starting Capital:        ${STARTING_CAPITAL:.2f}")
        print(f"Final Value:             ${final_value:.2f}")
        print(f"Cumulative Return:       {cumulative_return:.2f}%")

        print(f"\n{'='*70}")
        print("BACKTESTING COMPLETE")
        print(f"{'='*70}\n")

        # Save summary.txt file
        save_summary_txt(yearly_summaries, years)

    else:
        print("\nNo data was processed. Check your stocks folder and data files.")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        run_multi_year_backtest()
    except KeyboardInterrupt:
        print("\n\nBacktest interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
