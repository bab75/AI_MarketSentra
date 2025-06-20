import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class BacktestingEngine:
    """Comprehensive backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital=10000, transaction_cost=0.001):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital (float): Starting capital for backtesting
            transaction_cost (float): Transaction cost as percentage (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def run_backtest(self, data: pd.DataFrame, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run backtest for a single strategy
        
        Args:
            data (DataFrame): Stock data with OHLCV columns
            strategy_config (dict): Strategy configuration
            
        Returns:
            dict: Backtest results
        """
        try:
            strategy_name = strategy_config['name']
            strategy_type = strategy_config['type']
            
            # Generate trading signals based on strategy type
            if strategy_type == 'buy_and_hold':
                signals = self._buy_and_hold_strategy(data)
            elif strategy_type == 'moving_average':
                signals = self._moving_average_strategy(data, strategy_config['params'])
            elif strategy_type == 'rsi':
                signals = self._rsi_strategy(data, strategy_config['params'])
            elif strategy_type == 'bollinger_bands':
                signals = self._bollinger_bands_strategy(data, strategy_config['params'])
            elif strategy_type == 'macd':
                signals = self._macd_strategy(data, strategy_config['params'])
            elif strategy_type == 'momentum':
                signals = self._momentum_strategy(data, strategy_config['params'])
            elif strategy_type == 'stochastic':
                signals = self._stochastic_strategy(data, strategy_config['params'])
            elif strategy_type == 'williams_r':
                signals = self._williams_r_strategy(data, strategy_config['params'])
            elif strategy_type == 'atr_reversion':
                signals = self._atr_reversion_strategy(data, strategy_config['params'])
            elif strategy_type == 'triple_ma':
                signals = self._triple_ma_strategy(data, strategy_config['params'])
            elif strategy_type == 'volume_breakout':
                signals = self._volume_breakout_strategy(data, strategy_config['params'])
            elif strategy_type == 'support_resistance':
                signals = self._support_resistance_strategy(data, strategy_config['params'])
            elif strategy_type == 'channel_breakout':
                signals = self._channel_breakout_strategy(data, strategy_config['params'])
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            # Calculate portfolio performance
            portfolio_results = self._calculate_portfolio_performance(data, signals)
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(portfolio_results, data, signals)
            
            return {
                'strategy_name': strategy_name,
                'signals': signals,
                'portfolio': portfolio_results,
                'metrics': metrics,
                'trades': self._extract_trades(signals, data)
            }
            
        except Exception as e:
            st.error(f"Error running backtest for {strategy_config.get('name', 'Unknown')}: {str(e)}")
            return {
                'strategy_name': strategy_config.get('name', 'Unknown'),
                'signals': pd.DataFrame(),
                'portfolio': pd.DataFrame(),
                'metrics': {},
                'trades': pd.DataFrame()
            }
    
    def compare_strategies(self, data: pd.DataFrame, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple strategies side-by-side
        
        Args:
            data (DataFrame): Stock data
            strategies (list): List of strategy configurations
            
        Returns:
            dict: Comparison results
        """
        results = {}
        comparison_data = []
        
        for strategy_config in strategies:
            result = self.run_backtest(data, strategy_config)
            if result:
                results[result['strategy_name']] = result
                comparison_data.append({
                    'Strategy': result['strategy_name'],
                    'Total Return (%)': result['metrics']['total_return_pct'],
                    'Annual Return (%)': result['metrics']['annual_return_pct'],
                    'Sharpe Ratio': result['metrics']['sharpe_ratio'],
                    'Max Drawdown (%)': result['metrics']['max_drawdown_pct'],
                    'Win Rate (%)': result['metrics']['win_rate'],
                    'Total Trades': result['metrics']['total_trades'],
                    'Avg Trade Return (%)': result['metrics']['avg_trade_return'],
                    'Volatility (%)': result['metrics']['volatility_pct']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        return {
            'individual_results': results,
            'comparison_table': comparison_df,
            'best_strategy': self._find_best_strategy(comparison_df)
        }
    
    def _buy_and_hold_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Buy and hold strategy - buy at start, hold until end"""
        signals = pd.DataFrame(index=data.index)
        signals['Position'] = 1  # Always long
        signals['Signal'] = 0
        signals.iloc[0, signals.columns.get_loc('Signal')] = 1  # Buy signal at start
        # Add sell signal at the very end to realize gains
        signals.iloc[-1, signals.columns.get_loc('Signal')] = -1  # Sell signal at end
        return signals
    
    def _moving_average_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Moving average crossover strategy"""
        short_window = params.get('short_window', 20)
        long_window = params.get('long_window', 50)
        
        signals = pd.DataFrame(index=data.index)
        signals['Short_MA'] = data['Close'].rolling(window=short_window).mean()
        signals['Long_MA'] = data['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        signals['Position'] = 0
        signals['Position'][short_window:] = np.where(
            signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1, 0
        )
        
        # Generate trading signals (1 for buy, -1 for sell, 0 for hold)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _rsi_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """RSI-based strategy"""
        period = params.get('period', 14)
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)
        
        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.DataFrame(index=data.index)
        signals['RSI'] = rsi
        signals['Position'] = 0
        
        # Buy when RSI < oversold, sell when RSI > overbought
        signals.loc[signals['RSI'] < oversold, 'Position'] = 1
        signals.loc[signals['RSI'] > overbought, 'Position'] = 0
        
        # Forward fill positions
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _bollinger_bands_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Bollinger Bands strategy"""
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2)
        
        signals = pd.DataFrame(index=data.index)
        signals['MA'] = data['Close'].rolling(window=period).mean()
        signals['STD'] = data['Close'].rolling(window=period).std()
        signals['Upper_Band'] = signals['MA'] + (signals['STD'] * std_dev)
        signals['Lower_Band'] = signals['MA'] - (signals['STD'] * std_dev)
        
        signals['Position'] = 0
        
        # Buy when price touches lower band, sell when price touches upper band
        signals.loc[data['Close'] <= signals['Lower_Band'], 'Position'] = 1
        signals.loc[data['Close'] >= signals['Upper_Band'], 'Position'] = 0
        
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _macd_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """MACD strategy"""
        fast = params.get('fast', 12)
        slow = params.get('slow', 26)
        signal_period = params.get('signal', 9)
        
        signals = pd.DataFrame(index=data.index)
        
        # Calculate MACD
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        signals['MACD'] = ema_fast - ema_slow
        signals['Signal_Line'] = signals['MACD'].ewm(span=signal_period).mean()
        signals['Histogram'] = signals['MACD'] - signals['Signal_Line']
        
        # Generate position signals
        signals['Position'] = 0
        signals.loc[signals['MACD'] > signals['Signal_Line'], 'Position'] = 1
        
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _momentum_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Momentum strategy"""
        lookback = params.get('lookback', 20)
        threshold = params.get('threshold', 0.02)  # 2% momentum threshold
        
        signals = pd.DataFrame(index=data.index)
        signals['Momentum'] = data['Close'].pct_change(lookback)
        
        signals['Position'] = 0
        signals.loc[signals['Momentum'] > threshold, 'Position'] = 1
        signals.loc[signals['Momentum'] < -threshold, 'Position'] = 0
        
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _calculate_portfolio_performance(self, data: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate portfolio performance based on signals"""
        portfolio = pd.DataFrame(index=data.index)
        portfolio['Price'] = data['Close']
        portfolio['Position'] = signals['Position']
        portfolio['Holdings'] = 0.0
        portfolio['Cash'] = float(self.initial_capital)
        portfolio['Total'] = float(self.initial_capital)
        portfolio['Returns'] = 0.0
        
        # Initialize first position if there's a buy signal at start
        if signals['Signal'].iloc[0] > 0:
            shares_to_buy = (portfolio['Cash'].iloc[0] * (1 - self.transaction_cost)) // portfolio['Price'].iloc[0]
            cost = shares_to_buy * portfolio['Price'].iloc[0] * (1 + self.transaction_cost)
            portfolio.iloc[0, portfolio.columns.get_loc('Holdings')] = shares_to_buy
            portfolio.iloc[0, portfolio.columns.get_loc('Cash')] = self.initial_capital - cost
            portfolio.iloc[0, portfolio.columns.get_loc('Total')] = portfolio['Cash'].iloc[0] + portfolio['Holdings'].iloc[0] * portfolio['Price'].iloc[0]
        
        for i in range(1, len(portfolio)):
            # Check for position changes
            if signals['Signal'].iloc[i] != 0:
                if signals['Signal'].iloc[i] > 0:  # Buy signal
                    # Calculate shares to buy with available cash
                    shares_to_buy = (portfolio['Cash'].iloc[i-1] * (1 - self.transaction_cost)) // portfolio['Price'].iloc[i]
                    cost = shares_to_buy * portfolio['Price'].iloc[i] * (1 + self.transaction_cost)
                    
                    portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = portfolio['Holdings'].iloc[i-1] + shares_to_buy
                    portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio['Cash'].iloc[i-1] - cost
                    
                elif signals['Signal'].iloc[i] < 0:  # Sell signal
                    # Sell all holdings
                    proceeds = portfolio['Holdings'].iloc[i-1] * portfolio['Price'].iloc[i] * (1 - self.transaction_cost)
                    
                    portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = 0
                    portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio['Cash'].iloc[i-1] + proceeds
            else:
                # No signal, carry forward previous values
                portfolio.iloc[i, portfolio.columns.get_loc('Holdings')] = portfolio['Holdings'].iloc[i-1]
                portfolio.iloc[i, portfolio.columns.get_loc('Cash')] = portfolio['Cash'].iloc[i-1]
            
            # Calculate total portfolio value
            portfolio.iloc[i, portfolio.columns.get_loc('Total')] = (
                portfolio['Cash'].iloc[i] + portfolio['Holdings'].iloc[i] * portfolio['Price'].iloc[i]
            )
            
            # Calculate returns
            portfolio.iloc[i, portfolio.columns.get_loc('Returns')] = (
                portfolio['Total'].iloc[i] / portfolio['Total'].iloc[i-1] - 1
            )
        
        return portfolio
    
    def _calculate_performance_metrics(self, portfolio: pd.DataFrame, data: pd.DataFrame, signals: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        total_return = (portfolio['Total'].iloc[-1] / self.initial_capital) - 1
        
        # Calculate annual return
        days = len(portfolio)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # Calculate Sharpe ratio
        daily_returns = portfolio['Returns'].dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
        
        # Calculate maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate volatility
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Calculate trade-related metrics
        if signals is not None and 'Signal' in signals.columns:
            trade_signals = signals['Signal']
            trades = len(trade_signals[trade_signals != 0]) // 2
        else:
            trades = 0
        
        # Calculate win rate (simplified)
        positive_returns = daily_returns[daily_returns > 0]
        win_rate = len(positive_returns) / len(daily_returns) * 100 if len(daily_returns) > 0 else 0
        
        return {
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'volatility_pct': volatility * 100,
            'win_rate': win_rate,
            'total_trades': trades,
            'avg_trade_return': daily_returns.mean() * 100 if not daily_returns.empty else 0,
            'best_day': daily_returns.max() * 100 if not daily_returns.empty else 0,
            'worst_day': daily_returns.min() * 100 if not daily_returns.empty else 0
        }
    
    def _extract_trades(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Extract individual trades from signals"""
        trades = []
        in_position = False
        entry_date = None
        entry_price = None
        
        for date, row in signals.iterrows():
            if row['Signal'] > 0 and not in_position:  # Buy signal
                in_position = True
                entry_date = date
                entry_price = data.loc[date, 'Close']
                
            elif row['Signal'] < 0 and in_position:  # Sell signal
                exit_date = date
                exit_price = data.loc[date, 'Close']
                
                trade_return = (exit_price - entry_price) / entry_price
                try:
                    holding_days = (exit_date - entry_date).days
                except:
                    holding_days = 0
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': exit_date,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Return_%': trade_return * 100,
                    'Holding_Days': holding_days
                })
                
                in_position = False
        
        return pd.DataFrame(trades)
    
    def _find_best_strategy(self, comparison_df: pd.DataFrame) -> Dict[str, Any]:
        """Find the best performing strategy based on multiple criteria"""
        if comparison_df.empty:
            return {}
        
        # Calculate composite score (weighted average of key metrics)
        weights = {
            'Total Return (%)': 0.3,
            'Sharpe Ratio': 0.3,
            'Max Drawdown (%)': -0.2,  # Negative weight (lower is better)
            'Win Rate (%)': 0.2
        }
        
        comparison_df['Score'] = 0
        for metric, weight in weights.items():
            if metric in comparison_df.columns:
                normalized = (comparison_df[metric] - comparison_df[metric].min()) / (comparison_df[metric].max() - comparison_df[metric].min())
                comparison_df['Score'] += normalized * weight
        
        best_idx = comparison_df['Score'].idxmax()
        best_strategy = comparison_df.loc[best_idx]
        
        return {
            'name': best_strategy['Strategy'],
            'score': best_strategy['Score'],
            'metrics': best_strategy.to_dict()
        }
    
    def create_comparison_charts(self, comparison_results: Dict[str, Any]) -> Dict[str, go.Figure]:
        """Create comprehensive comparison visualizations"""
        charts = {}
        
        # 1. Portfolio Performance Comparison
        fig_performance = go.Figure()
        
        for strategy_name, result in comparison_results['individual_results'].items():
            portfolio = result['portfolio']
            fig_performance.add_trace(go.Scatter(
                x=portfolio.index,
                y=portfolio['Total'],
                mode='lines',
                name=strategy_name,
                hovertemplate=f'{strategy_name}<br>Value: $%{{y:,.2f}}<br>Date: %{{x}}<extra></extra>'
            ))
        
        fig_performance.update_layout(
            title='Portfolio Value Comparison Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            showlegend=True
        )
        
        charts['performance'] = fig_performance
        
        # 2. Metrics Comparison Bar Chart
        comparison_df = comparison_results['comparison_table']
        
        if not comparison_df.empty:
            fig_metrics = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Total Return
            fig_metrics.add_trace(
                go.Bar(x=comparison_df['Strategy'], y=comparison_df['Total Return (%)'], name='Total Return'),
                row=1, col=1
            )
            
            # Sharpe Ratio
            fig_metrics.add_trace(
                go.Bar(x=comparison_df['Strategy'], y=comparison_df['Sharpe Ratio'], name='Sharpe Ratio'),
                row=1, col=2
            )
            
            # Max Drawdown
            fig_metrics.add_trace(
                go.Bar(x=comparison_df['Strategy'], y=comparison_df['Max Drawdown (%)'], name='Max Drawdown'),
                row=2, col=1
            )
            
            # Win Rate
            fig_metrics.add_trace(
                go.Bar(x=comparison_df['Strategy'], y=comparison_df['Win Rate (%)'], name='Win Rate'),
                row=2, col=2
            )
            
            fig_metrics.update_layout(
                title='Strategy Performance Metrics Comparison',
                showlegend=False,
                height=600
            )
            
            charts['metrics'] = fig_metrics
        
        return charts

    def get_predefined_strategies(self) -> List[Dict[str, Any]]:
        """Get list of predefined trading strategies"""
        return [
            {
                'name': 'Buy and Hold',
                'type': 'buy_and_hold',
                'description': 'Simple buy and hold strategy - buy at start and hold until end',
                'params': {}
            },
            {
                'name': 'Moving Average Crossover (20/50)',
                'type': 'moving_average',
                'description': 'Buy when 20-day MA crosses above 50-day MA, sell when it crosses below',
                'params': {'short_window': 20, 'long_window': 50}
            },
            {
                'name': 'Moving Average Crossover (10/30)',
                'type': 'moving_average',
                'description': 'Buy when 10-day MA crosses above 30-day MA, sell when it crosses below',
                'params': {'short_window': 10, 'long_window': 30}
            },
            {
                'name': 'RSI Mean Reversion',
                'type': 'rsi',
                'description': 'Buy when RSI < 30 (oversold), sell when RSI > 70 (overbought)',
                'params': {'period': 14, 'oversold': 30, 'overbought': 70}
            },
            {
                'name': 'RSI Conservative',
                'type': 'rsi',
                'description': 'Buy when RSI < 25, sell when RSI > 75 (more conservative)',
                'params': {'period': 14, 'oversold': 25, 'overbought': 75}
            },
            {
                'name': 'Bollinger Bands Mean Reversion',
                'type': 'bollinger_bands',
                'description': 'Buy at lower band, sell at upper band',
                'params': {'period': 20, 'std_dev': 2}
            },
            {
                'name': 'MACD Trend Following',
                'type': 'macd',
                'description': 'Buy when MACD crosses above signal line, sell when it crosses below',
                'params': {'fast': 12, 'slow': 26, 'signal': 9}
            },
            {
                'name': 'Momentum Breakout',
                'type': 'momentum',
                'description': 'Buy on positive momentum, sell on negative momentum',
                'params': {'lookback': 20, 'threshold': 0.02}
            },
            {
                'name': 'Stochastic Oscillator',
                'type': 'stochastic',
                'description': 'Buy when Stochastic oversold, sell when overbought',
                'params': {'k_period': 14, 'd_period': 3, 'oversold': 20, 'overbought': 80}
            },
            {
                'name': 'Williams %R Strategy',
                'type': 'williams_r',
                'description': 'Buy when Williams %R oversold, sell when overbought',
                'params': {'period': 14, 'oversold': -80, 'overbought': -20}
            },
            {
                'name': 'Mean Reversion with ATR',
                'type': 'atr_reversion',
                'description': 'Mean reversion strategy using Average True Range for volatility',
                'params': {'period': 14, 'atr_multiplier': 2}
            },
            {
                'name': 'Trend Following Triple MA',
                'type': 'triple_ma',
                'description': 'Buy when price above all 3 MAs, sell when below',
                'params': {'short': 10, 'medium': 20, 'long': 50}
            },
            {
                'name': 'Volume Breakout',
                'type': 'volume_breakout',
                'description': 'Buy on high volume price breakouts',
                'params': {'volume_period': 20, 'price_period': 20, 'volume_multiplier': 1.5}
            },
            {
                'name': 'Support & Resistance',
                'type': 'support_resistance',
                'description': 'Buy at support levels, sell at resistance levels',
                'params': {'lookback': 50, 'threshold': 0.02}
            },
            {
                'name': 'Channel Breakout',
                'type': 'channel_breakout',
                'description': 'Buy when price breaks above recent highs',
                'params': {'period': 20, 'breakout_threshold': 0.01}
            }
        ]
    
    def _stochastic_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Stochastic Oscillator strategy"""
        k_period = params.get('k_period', 14)
        d_period = params.get('d_period', 3)
        oversold = params.get('oversold', 20)
        overbought = params.get('overbought', 80)
        
        signals = pd.DataFrame(index=data.index)
        
        # Calculate Stochastic
        lowest_low = data['Low'].rolling(window=k_period).min()
        highest_high = data['High'].rolling(window=k_period).max()
        signals['Stoch_K'] = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
        signals['Stoch_D'] = signals['Stoch_K'].rolling(window=d_period).mean()
        
        signals['Position'] = 0
        signals.loc[(signals['Stoch_K'] < oversold) & (signals['Stoch_D'] < oversold), 'Position'] = 1
        signals.loc[(signals['Stoch_K'] > overbought) & (signals['Stoch_D'] > overbought), 'Position'] = 0
        
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _williams_r_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Williams %R strategy"""
        period = params.get('period', 14)
        oversold = params.get('oversold', -80)
        overbought = params.get('overbought', -20)
        
        signals = pd.DataFrame(index=data.index)
        
        # Calculate Williams %R
        highest_high = data['High'].rolling(window=period).max()
        lowest_low = data['Low'].rolling(window=period).min()
        signals['Williams_R'] = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))
        
        signals['Position'] = 0
        signals.loc[signals['Williams_R'] < oversold, 'Position'] = 1
        signals.loc[signals['Williams_R'] > overbought, 'Position'] = 0
        
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _atr_reversion_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """ATR-based mean reversion strategy"""
        period = params.get('period', 14)
        atr_multiplier = params.get('atr_multiplier', 2)
        
        signals = pd.DataFrame(index=data.index)
        
        # Calculate ATR
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        signals['ATR'] = true_range.rolling(window=period).mean()
        
        # Calculate mean and bands
        signals['SMA'] = data['Close'].rolling(window=period).mean()
        signals['Upper_Band'] = signals['SMA'] + (signals['ATR'] * atr_multiplier)
        signals['Lower_Band'] = signals['SMA'] - (signals['ATR'] * atr_multiplier)
        
        signals['Position'] = 0
        signals.loc[data['Close'] <= signals['Lower_Band'], 'Position'] = 1
        signals.loc[data['Close'] >= signals['Upper_Band'], 'Position'] = 0
        
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _triple_ma_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Triple Moving Average strategy"""
        short = params.get('short', 10)
        medium = params.get('medium', 20)
        long = params.get('long', 50)
        
        signals = pd.DataFrame(index=data.index)
        signals['MA_Short'] = data['Close'].rolling(window=short).mean()
        signals['MA_Medium'] = data['Close'].rolling(window=medium).mean()
        signals['MA_Long'] = data['Close'].rolling(window=long).mean()
        
        # Buy when price above all MAs and MAs in ascending order
        buy_condition = ((data['Close'] > signals['MA_Short']) & 
                        (signals['MA_Short'] > signals['MA_Medium']) & 
                        (signals['MA_Medium'] > signals['MA_Long']))
        
        signals['Position'] = 0
        signals.loc[buy_condition, 'Position'] = 1
        
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _volume_breakout_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Volume breakout strategy"""
        volume_period = params.get('volume_period', 20)
        price_period = params.get('price_period', 20)
        volume_multiplier = params.get('volume_multiplier', 1.5)
        
        signals = pd.DataFrame(index=data.index)
        
        # Calculate volume and price indicators
        signals['Volume_MA'] = data['Volume'].rolling(window=volume_period).mean()
        signals['Price_High'] = data['High'].rolling(window=price_period).max()
        
        # High volume breakout conditions
        high_volume = data['Volume'] > (signals['Volume_MA'] * volume_multiplier)
        price_breakout_up = data['Close'] > signals['Price_High'].shift(1)
        
        signals['Position'] = 0
        signals.loc[high_volume & price_breakout_up, 'Position'] = 1
        
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _support_resistance_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Support and resistance strategy"""
        lookback = params.get('lookback', 50)
        threshold = params.get('threshold', 0.02)
        
        signals = pd.DataFrame(index=data.index)
        
        # Calculate support and resistance levels
        signals['Support'] = data['Low'].rolling(window=lookback).min()
        signals['Resistance'] = data['High'].rolling(window=lookback).max()
        
        # Buy near support
        near_support = (data['Close'] - signals['Support']) / signals['Support'] <= threshold
        
        signals['Position'] = 0
        signals.loc[near_support, 'Position'] = 1
        
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
    
    def _channel_breakout_strategy(self, data: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Channel breakout strategy"""
        period = params.get('period', 20)
        breakout_threshold = params.get('breakout_threshold', 0.01)
        
        signals = pd.DataFrame(index=data.index)
        
        # Calculate channel
        signals['Upper_Channel'] = data['High'].rolling(window=period).max()
        
        # Breakout conditions
        upper_breakout = data['Close'] > signals['Upper_Channel'].shift(1) * (1 + breakout_threshold)
        
        signals['Position'] = 0
        signals.loc[upper_breakout, 'Position'] = 1
        
        signals['Position'] = signals['Position'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        signals['Signal'] = signals['Position'].diff()
        
        return signals
