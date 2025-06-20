import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Comprehensive technical indicators calculator and visualizer"""
    
    def __init__(self):
        pass
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given data"""
        df = data.copy()
        
        # Trend Indicators
        df = self.add_moving_averages(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_ichimoku_cloud(df)
        df = self.add_parabolic_sar(df)
        
        # Momentum Indicators
        df = self.add_rsi(df)
        df = self.add_stochastic_oscillator(df)
        df = self.add_williams_r(df)
        df = self.add_cci(df)
        df = self.add_momentum(df)
        df = self.add_roc(df)
        
        # Volatility Indicators
        df = self.add_atr(df)
        df = self.add_volatility(df)
        df = self.add_keltner_channels(df)
        
        # Volume Indicators
        df = self.add_obv(df)
        df = self.add_vwap(df)
        df = self.add_volume_sma(df)
        df = self.add_money_flow_index(df)
        
        # Support and Resistance
        df = self.add_pivot_points(df)
        df = self.add_fibonacci_levels(df)
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame, periods=[5, 10, 20, 50, 100, 200]) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages"""
        for period in periods:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()
        return df
    
    def add_macd(self, df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
        """Add MACD indicator"""
        ema_fast = df['Close'].ewm(span=fast).mean()
        ema_slow = df['Close'].ewm(span=slow).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=signal).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df
    
    def add_rsi(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """Add Relative Strength Index"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, period=20, std_dev=2) -> pd.DataFrame:
        """Add Bollinger Bands"""
        sma = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        df['BB_Upper'] = sma + (std * std_dev)
        df['BB_Middle'] = sma
        df['BB_Lower'] = sma - (std * std_dev)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        return df
    
    def add_stochastic_oscillator(self, df: pd.DataFrame, k_period=14, d_period=3) -> pd.DataFrame:
        """Add Stochastic Oscillator"""
        lowest_low = df['Low'].rolling(window=k_period).min()
        highest_high = df['High'].rolling(window=k_period).max()
        df['Stoch_K'] = 100 * ((df['Close'] - lowest_low) / (highest_high - lowest_low))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        return df
    
    def add_williams_r(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """Add Williams %R"""
        highest_high = df['High'].rolling(window=period).max()
        lowest_low = df['Low'].rolling(window=period).min()
        df['Williams_R'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
        return df
    
    def add_cci(self, df: pd.DataFrame, period=20) -> pd.DataFrame:
        """Add Commodity Channel Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        return df
    
    def add_atr(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """Add Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=period).mean()
        return df
    
    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume"""
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        return df
    
    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return df
    
    def add_ichimoku_cloud(self, df: pd.DataFrame, tenkan_period=9, kijun_period=26, senkou_span_b_period=52) -> pd.DataFrame:
        """Add Ichimoku Cloud indicators"""
        # Tenkan-sen (Conversion Line)
        tenkan_high = df['High'].rolling(window=tenkan_period).max()
        tenkan_low = df['Low'].rolling(window=tenkan_period).min()
        df['Ichimoku_Tenkan'] = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = df['High'].rolling(window=kijun_period).max()
        kijun_low = df['Low'].rolling(window=kijun_period).min()
        df['Ichimoku_Kijun'] = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        df['Ichimoku_Senkou_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = df['High'].rolling(window=senkou_span_b_period).max()
        senkou_b_low = df['Low'].rolling(window=senkou_span_b_period).min()
        df['Ichimoku_Senkou_B'] = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span)
        df['Ichimoku_Chikou'] = df['Close'].shift(-kijun_period)
        
        return df
    
    def add_parabolic_sar(self, df: pd.DataFrame, af_start=0.02, af_increment=0.02, af_max=0.2) -> pd.DataFrame:
        """Add Parabolic SAR"""
        psar = df['Close'].copy()
        psarbull = [None] * len(df)
        psarbear = [None] * len(df)
        
        bull = True
        af = af_start
        ep = df['Low'].iloc[0] if not bull else df['High'].iloc[0]
        hp = df['High'].iloc[0]
        lp = df['Low'].iloc[0]
        
        for i in range(2, len(df)):
            if bull:
                psar.iloc[i] = psar.iloc[i-1] + af * (hp - psar.iloc[i-1])
            else:
                psar.iloc[i] = psar.iloc[i-1] + af * (lp - psar.iloc[i-1])
            
            reverse = False
            
            if bull:
                if df['Low'].iloc[i] <= psar.iloc[i]:
                    bull = False
                    reverse = True
                    psar.iloc[i] = hp
                    lp = df['Low'].iloc[i]
                    af = af_start
            else:
                if df['High'].iloc[i] >= psar.iloc[i]:
                    bull = True
                    reverse = True
                    psar.iloc[i] = lp
                    hp = df['High'].iloc[i]
                    af = af_start
            
            if not reverse:
                if bull:
                    if df['High'].iloc[i] > hp:
                        hp = df['High'].iloc[i]
                        af = min(af + af_increment, af_max)
                    if df['Low'].iloc[i-1] < psar.iloc[i]:
                        psar.iloc[i] = df['Low'].iloc[i-1]
                    if df['Low'].iloc[i-2] < psar.iloc[i]:
                        psar.iloc[i] = df['Low'].iloc[i-2]
                else:
                    if df['Low'].iloc[i] < lp:
                        lp = df['Low'].iloc[i]
                        af = min(af + af_increment, af_max)
                    if df['High'].iloc[i-1] > psar.iloc[i]:
                        psar.iloc[i] = df['High'].iloc[i-1]
                    if df['High'].iloc[i-2] > psar.iloc[i]:
                        psar.iloc[i] = df['High'].iloc[i-2]
            
            if bull:
                psarbull[i] = psar.iloc[i]
            else:
                psarbear[i] = psar.iloc[i]
        
        df['PSAR'] = psar
        df['PSAR_Bull'] = psarbull
        df['PSAR_Bear'] = psarbear
        return df
    
    def add_momentum(self, df: pd.DataFrame, period=10) -> pd.DataFrame:
        """Add Momentum indicator"""
        df['Momentum'] = df['Close'] / df['Close'].shift(period) * 100
        return df
    
    def add_roc(self, df: pd.DataFrame, period=12) -> pd.DataFrame:
        """Add Rate of Change"""
        df['ROC'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        return df
    
    def add_volatility(self, df: pd.DataFrame, period=20) -> pd.DataFrame:
        """Add Volatility indicators"""
        df['Volatility'] = df['Close'].pct_change().rolling(window=period).std() * np.sqrt(252) * 100
        return df
    
    def add_keltner_channels(self, df: pd.DataFrame, period=20, multiplier=2) -> pd.DataFrame:
        """Add Keltner Channels"""
        df = self.add_atr(df, period)
        ema = df['Close'].ewm(span=period).mean()
        df['Keltner_Upper'] = ema + (multiplier * df['ATR'])
        df['Keltner_Middle'] = ema
        df['Keltner_Lower'] = ema - (multiplier * df['ATR'])
        return df
    
    def add_volume_sma(self, df: pd.DataFrame, period=20) -> pd.DataFrame:
        """Add Volume Simple Moving Average"""
        df['Volume_SMA'] = df['Volume'].rolling(window=period).mean()
        return df
    
    def add_money_flow_index(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """Add Money Flow Index"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        positive_flow = [0] + positive_flow
        negative_flow = [0] + negative_flow
        
        positive_flow_sum = pd.Series(positive_flow).rolling(window=period).sum()
        negative_flow_sum = pd.Series(negative_flow).rolling(window=period).sum()
        
        money_flow_ratio = positive_flow_sum / negative_flow_sum
        df['MFI'] = 100 - (100 / (1 + money_flow_ratio))
        return df
    
    def add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Pivot Points"""
        df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
        df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
        df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
        df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
        df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
        return df
    
    def add_fibonacci_levels(self, df: pd.DataFrame, period=50) -> pd.DataFrame:
        """Add Fibonacci Retracement Levels"""
        rolling_max = df['High'].rolling(window=period).max()
        rolling_min = df['Low'].rolling(window=period).min()
        diff = rolling_max - rolling_min
        
        df['Fib_0'] = rolling_max
        df['Fib_236'] = rolling_max - 0.236 * diff
        df['Fib_382'] = rolling_max - 0.382 * diff
        df['Fib_500'] = rolling_max - 0.500 * diff
        df['Fib_618'] = rolling_max - 0.618 * diff
        df['Fib_100'] = rolling_min
        return df
    
    def create_technical_analysis_chart(self, data: pd.DataFrame, indicators: List[str], title: str = "Technical Analysis") -> go.Figure:
        """Create comprehensive technical analysis chart"""
        # Calculate indicators if not present
        if not any(indicator in data.columns for indicator in indicators):
            data = self.calculate_all_indicators(data)
        
        # Create subplots
        subplot_titles = ['Price Action']
        subplot_specs = [{"secondary_y": True}]
        
        # Add subplots for oscillators
        oscillator_indicators = ['RSI', 'Stoch_K', 'Stoch_D', 'Williams_R', 'CCI', 'MFI']
        volume_indicators = ['Volume', 'OBV', 'Volume_SMA']
        
        if any(ind in indicators for ind in oscillator_indicators):
            subplot_titles.append('Oscillators')
            subplot_specs.append({"secondary_y": False})
        
        if any(ind in indicators for ind in volume_indicators):
            subplot_titles.append('Volume')
            subplot_specs.append({"secondary_y": False})
        
        if 'MACD' in indicators:
            subplot_titles.append('MACD')
            subplot_specs.append({"secondary_y": False})
        
        rows = len(subplot_titles)
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True}] if i == 0 else [{"secondary_y": False}] for i in range(rows)],
            row_heights=[0.6] + [0.4/(rows-1)]*(rows-1) if rows > 1 else [1.0]
        )
        
        # Main price chart (candlestick)
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Add price indicators to main chart
        colors = ['blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_idx = 0
        
        # Moving Averages
        ma_indicators = [ind for ind in indicators if 'SMA_' in ind or 'EMA_' in ind]
        for ma in ma_indicators:
            if ma in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[ma],
                        mode='lines',
                        name=ma,
                        line=dict(color=colors[color_idx % len(colors)], width=1),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
                color_idx += 1
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']) and 'BB' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='lightblue', width=1),
                    opacity=0.5
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='lightblue', width=1),
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.2)',
                    opacity=0.5
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_Middle'],
                    mode='lines',
                    name='BB Middle',
                    line=dict(color='blue', width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Ichimoku Cloud
        if all(col in data.columns for col in ['Ichimoku_Senkou_A', 'Ichimoku_Senkou_B']) and 'Ichimoku' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Ichimoku_Senkou_A'],
                    mode='lines',
                    name='Ichimoku Senkou A',
                    line=dict(color='green', width=1),
                    opacity=0.3
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Ichimoku_Senkou_B'],
                    mode='lines',
                    name='Ichimoku Senkou B',
                    line=dict(color='red', width=1),
                    fill='tonexty',
                    fillcolor='rgba(144, 238, 144, 0.2)',
                    opacity=0.3
                ),
                row=1, col=1
            )
        
        # Parabolic SAR
        if 'PSAR' in indicators and 'PSAR' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['PSAR'],
                    mode='markers',
                    name='Parabolic SAR',
                    marker=dict(color='purple', size=2),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # VWAP
        if 'VWAP' in indicators and 'VWAP' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['VWAP'],
                    mode='lines',
                    name='VWAP',
                    line=dict(color='orange', width=2),
                    opacity=0.8
                ),
                row=1, col=1
            )
        
        # Add volume to secondary y-axis of main chart
        if 'Volume' in indicators:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='lightgray',
                    opacity=0.3,
                    yaxis='y2'
                ),
                row=1, col=1, secondary_y=True
            )
        
        # Oscillators subplot
        current_row = 2
        if any(ind in indicators for ind in oscillator_indicators):
            if 'RSI' in indicators and 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=current_row, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
            
            if 'Stoch_K' in indicators and 'Stoch_K' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Stoch_K'],
                        mode='lines',
                        name='Stoch %K',
                        line=dict(color='blue', width=1)
                    ),
                    row=current_row, col=1
                )
                if 'Stoch_D' in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['Stoch_D'],
                            mode='lines',
                            name='Stoch %D',
                            line=dict(color='red', width=1)
                        ),
                        row=current_row, col=1
                    )
            
            if 'Williams_R' in indicators and 'Williams_R' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Williams_R'],
                        mode='lines',
                        name='Williams %R',
                        line=dict(color='orange', width=1)
                    ),
                    row=current_row, col=1
                )
            
            if 'CCI' in indicators and 'CCI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['CCI'],
                        mode='lines',
                        name='CCI',
                        line=dict(color='green', width=1)
                    ),
                    row=current_row, col=1
                )
                fig.add_hline(y=100, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=-100, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
            
            if 'MFI' in indicators and 'MFI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MFI'],
                        mode='lines',
                        name='MFI',
                        line=dict(color='brown', width=1)
                    ),
                    row=current_row, col=1
                )
                fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
            
            current_row += 1
        
        # Volume subplot
        if any(ind in indicators for ind in volume_indicators) and current_row <= rows:
            if 'Volume' in indicators:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color='lightblue',
                        opacity=0.7
                    ),
                    row=current_row, col=1
                )
            
            if 'OBV' in indicators and 'OBV' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['OBV'],
                        mode='lines',
                        name='OBV',
                        line=dict(color='purple', width=2),
                        yaxis='y2'
                    ),
                    row=current_row, col=1
                )
            
            if 'Volume_SMA' in indicators and 'Volume_SMA' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Volume_SMA'],
                        mode='lines',
                        name='Volume SMA',
                        line=dict(color='red', width=2)
                    ),
                    row=current_row, col=1
                )
            
            current_row += 1
        
        # MACD subplot
        if 'MACD' in indicators and current_row <= rows:
            if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACD_Signal'],
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color='red', width=2)
                    ),
                    row=current_row, col=1
                )
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color='gray',
                        opacity=0.6
                    ),
                    row=current_row, col=1
                )
                fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=current_row, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            showlegend=True,
            height=200 * rows + 100,
            hovermode='x unified'
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if rows > 1:
            fig.update_yaxes(title_text="Volume", secondary_y=True, row=1, col=1)
        
        current_row = 2
        if any(ind in indicators for ind in oscillator_indicators) and current_row <= rows:
            fig.update_yaxes(title_text="Oscillators", row=current_row, col=1)
            current_row += 1
        
        if any(ind in indicators for ind in volume_indicators) and current_row <= rows:
            fig.update_yaxes(title_text="Volume", row=current_row, col=1)
            current_row += 1
        
        if 'MACD' in indicators and current_row <= rows:
            fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        
        return fig
    
    def get_indicator_categories(self) -> Dict[str, List[str]]:
        """Get indicators organized by categories"""
        return {
            'Trend Indicators': [
                'SMA_20', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50', 'EMA_200',
                'BB', 'MACD', 'Ichimoku', 'PSAR', 'VWAP'
            ],
            'Momentum Indicators': [
                'RSI', 'Stoch_K', 'Williams_R', 'CCI', 'Momentum', 'ROC', 'MFI'
            ],
            'Volatility Indicators': [
                'ATR', 'Volatility', 'Keltner_Upper', 'Keltner_Lower'
            ],
            'Volume Indicators': [
                'Volume', 'OBV', 'Volume_SMA'
            ],
            'Support/Resistance': [
                'Pivot', 'R1', 'R2', 'S1', 'S2', 'Fib_236', 'Fib_382', 'Fib_618'
            ]
        }
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on technical indicators"""
        df = data.copy()
        
        # Calculate indicators if not present
        if 'RSI' not in df.columns:
            df = self.calculate_all_indicators(df)
        
        # Initialize signal columns
        df['Buy_Signal'] = 0
        df['Sell_Signal'] = 0
        df['Signal_Strength'] = 0
        
        signals = []
        
        # RSI signals
        if 'RSI' in df.columns:
            rsi_oversold = df['RSI'] < 30
            rsi_overbought = df['RSI'] > 70
            signals.append(('RSI Oversold', rsi_oversold, 1))
            signals.append(('RSI Overbought', rsi_overbought, -1))
        
        # MACD signals
        if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
            macd_bullish = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
            macd_bearish = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
            signals.append(('MACD Bullish Cross', macd_bullish, 1))
            signals.append(('MACD Bearish Cross', macd_bearish, -1))
        
        # Bollinger Bands signals
        if all(col in df.columns for col in ['BB_Upper', 'BB_Lower']):
            bb_oversold = df['Close'] <= df['BB_Lower']
            bb_overbought = df['Close'] >= df['BB_Upper']
            signals.append(('BB Oversold', bb_oversold, 1))
            signals.append(('BB Overbought', bb_overbought, -1))
        
        # Stochastic signals
        if all(col in df.columns for col in ['Stoch_K', 'Stoch_D']):
            stoch_oversold = (df['Stoch_K'] < 20) & (df['Stoch_D'] < 20)
            stoch_overbought = (df['Stoch_K'] > 80) & (df['Stoch_D'] > 80)
            signals.append(('Stochastic Oversold', stoch_oversold, 1))
            signals.append(('Stochastic Overbought', stoch_overbought, -1))
        
        # Aggregate signals
        for signal_name, condition, direction in signals:
            df.loc[condition, 'Buy_Signal' if direction > 0 else 'Sell_Signal'] += abs(direction)
            df.loc[condition, 'Signal_Strength'] += direction
        
        return df
    
    def get_indicator_summary(self, data: pd.DataFrame) -> Dict[str, str]:
        """Get summary of current indicator readings"""
        if 'RSI' not in data.columns:
            data = self.calculate_all_indicators(data)
        
        latest = data.iloc[-1]
        summary = {}
        
        # RSI analysis
        if 'RSI' in data.columns:
            rsi_val = latest['RSI']
            if rsi_val < 30:
                summary['RSI'] = f"Oversold ({rsi_val:.1f})"
            elif rsi_val > 70:
                summary['RSI'] = f"Overbought ({rsi_val:.1f})"
            else:
                summary['RSI'] = f"Neutral ({rsi_val:.1f})"
        
        # MACD analysis
        if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
            macd_val = latest['MACD']
            signal_val = latest['MACD_Signal']
            if macd_val > signal_val:
                summary['MACD'] = "Bullish"
            else:
                summary['MACD'] = "Bearish"
        
        # Bollinger Bands analysis
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Position']):
            bb_pos = latest['BB_Position']
            if bb_pos < 0.2:
                summary['Bollinger Bands'] = "Near Lower Band"
            elif bb_pos > 0.8:
                summary['Bollinger Bands'] = "Near Upper Band"
            else:
                summary['Bollinger Bands'] = "Within Bands"
        
        # Moving Average analysis
        if all(col in data.columns for col in ['SMA_20', 'SMA_50']):
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            price = latest['Close']
            
            if price > sma20 > sma50:
                summary['Trend'] = "Strong Uptrend"
            elif price > sma20 and sma20 < sma50:
                summary['Trend'] = "Weak Uptrend"
            elif price < sma20 < sma50:
                summary['Trend'] = "Strong Downtrend"
            else:
                summary['Trend'] = "Weak Downtrend"
        
        return summary