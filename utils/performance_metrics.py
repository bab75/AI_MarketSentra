import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import streamlit as st

class PerformanceMetrics:
    """Calculate and analyze performance metrics for financial data and models"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # Default risk-free rate (2%)
        self.trading_days = 252  # Trading days per year
    
    def calculate_all_metrics(self, data, price_col='Close'):
        """
        Calculate comprehensive performance metrics
        
        Args:
            data (DataFrame): Stock data
            price_col (str): Price column name
            
        Returns:
            dict: All calculated metrics
        """
        try:
            metrics = {}
            
            # Basic price metrics
            metrics.update(self.calculate_price_metrics(data, price_col))
            
            # Return metrics
            metrics.update(self.calculate_return_metrics(data, price_col))
            
            # Risk metrics
            metrics.update(self.calculate_risk_metrics(data, price_col))
            
            # Technical metrics
            metrics.update(self.calculate_technical_metrics(data))
            
            # Portfolio metrics
            metrics.update(self.calculate_portfolio_metrics(data, price_col))
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def calculate_price_metrics(self, data, price_col='Close'):
        """Calculate basic price-related metrics"""
        try:
            prices = data[price_col].dropna()
            
            metrics = {
                'current_price': prices.iloc[-1],
                'price_change': prices.iloc[-1] - prices.iloc[-2] if len(prices) > 1 else 0,
                'price_change_pct': ((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100) if len(prices) > 1 else 0,
                'min_price': prices.min(),
                'max_price': prices.max(),
                'avg_price': prices.mean(),
                'median_price': prices.median(),
                'price_std': prices.std(),
                'price_range': prices.max() - prices.min(),
                'price_range_pct': ((prices.max() - prices.min()) / prices.min() * 100) if prices.min() > 0 else 0
            }
            
            # Recent performance (last 30 days)
            if len(prices) >= 30:
                recent_prices = prices.tail(30)
                metrics.update({
                    'price_change_30d': recent_prices.iloc[-1] - recent_prices.iloc[0],
                    'price_change_30d_pct': ((recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100),
                    'avg_price_30d': recent_prices.mean(),
                    'volatility_30d': recent_prices.std()
                })
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating price metrics: {str(e)}")
            return {}
    
    def calculate_return_metrics(self, data, price_col='Close'):
        """Calculate return-based metrics"""
        try:
            prices = data[price_col].dropna()
            
            if len(prices) < 2:
                return {}
            
            # Calculate returns
            daily_returns = prices.pct_change().dropna()
            
            # Cumulative returns
            cumulative_returns = (1 + daily_returns).cumprod() - 1
            total_return = cumulative_returns.iloc[-1]
            
            # Annualized metrics
            n_periods = len(daily_returns)
            years = n_periods / self.trading_days
            
            annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
            annualized_volatility = daily_returns.std() * np.sqrt(self.trading_days)
            
            metrics = {
                'total_return': total_return * 100,
                'annualized_return': annualized_return * 100,
                'annualized_volatility': annualized_volatility * 100,
                'daily_return_mean': daily_returns.mean() * 100,
                'daily_return_std': daily_returns.std() * 100,
                'positive_days': len(daily_returns[daily_returns > 0]),
                'negative_days': len(daily_returns[daily_returns < 0]),
                'win_rate': len(daily_returns[daily_returns > 0]) / len(daily_returns) * 100,
                'best_day': daily_returns.max() * 100,
                'worst_day': daily_returns.min() * 100,
                'avg_positive_return': daily_returns[daily_returns > 0].mean() * 100 if len(daily_returns[daily_returns > 0]) > 0 else 0,
                'avg_negative_return': daily_returns[daily_returns < 0].mean() * 100 if len(daily_returns[daily_returns < 0]) > 0 else 0
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating return metrics: {str(e)}")
            return {}
    
    def calculate_risk_metrics(self, data, price_col='Close'):
        """Calculate risk-based metrics"""
        try:
            prices = data[price_col].dropna()
            
            if len(prices) < 2:
                return {}
            
            daily_returns = prices.pct_change().dropna()
            
            # Value at Risk (VaR)
            var_95 = np.percentile(daily_returns, 5) * 100
            var_99 = np.percentile(daily_returns, 1) * 100
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = daily_returns[daily_returns <= np.percentile(daily_returns, 5)].mean() * 100
            cvar_99 = daily_returns[daily_returns <= np.percentile(daily_returns, 1)].mean() * 100
            
            # Maximum Drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Drawdown duration
            drawdown_periods = []
            in_drawdown = False
            drawdown_start = None
            
            for i, dd in enumerate(drawdown):
                if dd < 0 and not in_drawdown:
                    in_drawdown = True
                    drawdown_start = i
                elif dd >= 0 and in_drawdown:
                    in_drawdown = False
                    if drawdown_start is not None:
                        drawdown_periods.append(i - drawdown_start)
            
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
            
            # Sharpe Ratio
            excess_returns = daily_returns - (self.risk_free_rate / self.trading_days)
            sharpe_ratio = excess_returns.mean() / daily_returns.std() * np.sqrt(self.trading_days) if daily_returns.std() > 0 else 0
            
            # Sortino Ratio (downside deviation)
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(self.trading_days) if downside_deviation > 0 else 0
            
            # Calmar Ratio
            annualized_return = daily_returns.mean() * self.trading_days
            calmar_ratio = annualized_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0
            
            # Beta (if benchmark data available)
            beta = self.calculate_beta(daily_returns)
            
            # Skewness and Kurtosis
            skewness = stats.skew(daily_returns)
            kurtosis = stats.kurtosis(daily_returns)
            
            metrics = {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'max_drawdown': max_drawdown,
                'avg_drawdown_duration': avg_drawdown_duration,
                'max_drawdown_duration': max_drawdown_duration,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'beta': beta,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'volatility': daily_returns.std() * np.sqrt(self.trading_days) * 100
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def calculate_beta(self, returns, benchmark_returns=None):
        """Calculate beta relative to benchmark"""
        try:
            if benchmark_returns is None:
                # If no benchmark provided, assume market return of 8% annually
                # This is a simplified calculation
                market_return = 0.08 / self.trading_days
                benchmark_returns = pd.Series([market_return] * len(returns))
            
            if len(returns) != len(benchmark_returns):
                return None
            
            covariance = np.cov(returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1.0
            return beta
            
        except Exception:
            return 1.0  # Default beta
    
    def calculate_technical_metrics(self, data):
        """Calculate technical analysis metrics"""
        try:
            metrics = {}
            
            # RSI analysis
            if 'RSI' in data.columns:
                rsi = data['RSI'].dropna()
                if len(rsi) > 0:
                    metrics.update({
                        'current_rsi': rsi.iloc[-1],
                        'avg_rsi': rsi.mean(),
                        'rsi_overbought_days': len(rsi[rsi > 70]),
                        'rsi_oversold_days': len(rsi[rsi < 30]),
                        'rsi_neutral_days': len(rsi[(rsi >= 30) & (rsi <= 70)])
                    })
            
            # MACD analysis
            if all(col in data.columns for col in ['MACD', 'MACD_Signal']):
                macd = data['MACD'].dropna()
                signal = data['MACD_Signal'].dropna()
                
                if len(macd) > 0 and len(signal) > 0:
                    macd_diff = macd - signal
                    metrics.update({
                        'current_macd': macd.iloc[-1],
                        'current_macd_signal': signal.iloc[-1],
                        'macd_bullish_days': len(macd_diff[macd_diff > 0]),
                        'macd_bearish_days': len(macd_diff[macd_diff < 0])
                    })
            
            # Moving Average analysis
            if 'Close' in data.columns:
                close = data['Close'].dropna()
                
                for ma_period in [20, 50]:
                    ma_col = f'MA_{ma_period}'
                    if ma_col in data.columns:
                        ma = data[ma_col].dropna()
                        if len(ma) > 0 and len(close) > 0:
                            # Align indices
                            common_idx = close.index.intersection(ma.index)
                            if len(common_idx) > 0:
                                close_aligned = close.loc[common_idx]
                                ma_aligned = ma.loc[common_idx]
                                
                                above_ma = close_aligned > ma_aligned
                                metrics[f'days_above_ma_{ma_period}'] = above_ma.sum()
                                metrics[f'days_below_ma_{ma_period}'] = (~above_ma).sum()
            
            # Bollinger Bands analysis
            if all(col in data.columns for col in ['Close', 'BB_Upper', 'BB_Lower']):
                close = data['Close'].dropna()
                bb_upper = data['BB_Upper'].dropna()
                bb_lower = data['BB_Lower'].dropna()
                
                # Align indices
                common_idx = close.index.intersection(bb_upper.index).intersection(bb_lower.index)
                if len(common_idx) > 0:
                    close_aligned = close.loc[common_idx]
                    bb_upper_aligned = bb_upper.loc[common_idx]
                    bb_lower_aligned = bb_lower.loc[common_idx]
                    
                    above_upper = close_aligned > bb_upper_aligned
                    below_lower = close_aligned < bb_lower_aligned
                    
                    metrics.update({
                        'bb_squeeze_days': above_upper.sum(),
                        'bb_oversold_days': below_lower.sum(),
                        'bb_normal_days': len(common_idx) - above_upper.sum() - below_lower.sum()
                    })
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating technical metrics: {str(e)}")
            return {}
    
    def calculate_portfolio_metrics(self, data, price_col='Close', initial_investment=10000):
        """Calculate portfolio performance metrics"""
        try:
            prices = data[price_col].dropna()
            
            if len(prices) < 2:
                return {}
            
            # Calculate portfolio value assuming buy and hold
            initial_price = prices.iloc[0]
            current_price = prices.iloc[-1]
            shares = initial_investment / initial_price
            current_value = shares * current_price
            
            # Portfolio returns
            portfolio_return = (current_value - initial_investment) / initial_investment * 100
            
            # Time-weighted metrics
            days_invested = len(prices)
            years_invested = days_invested / self.trading_days
            
            annualized_portfolio_return = ((current_value / initial_investment) ** (1/years_invested) - 1) * 100 if years_invested > 0 else 0
            
            metrics = {
                'initial_investment': initial_investment,
                'current_portfolio_value': current_value,
                'portfolio_return': portfolio_return,
                'annualized_portfolio_return': annualized_portfolio_return,
                'shares_owned': shares,
                'days_invested': days_invested,
                'years_invested': years_invested,
                'profit_loss': current_value - initial_investment
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating portfolio metrics: {str(e)}")
            return {}
    
    def calculate_model_performance_metrics(self, actual, predicted, model_name="Model"):
        """
        Calculate performance metrics for ML model predictions
        
        Args:
            actual (array): Actual values
            predicted (array): Predicted values
            model_name (str): Name of the model
            
        Returns:
            dict: Model performance metrics
        """
        try:
            actual = np.array(actual)
            predicted = np.array(predicted)
            
            # Basic metrics
            mse = mean_squared_error(actual, predicted)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)
            
            # Mean Absolute Percentage Error (MAPE)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Direction accuracy (for price prediction)
            if len(actual) > 1:
                actual_direction = np.diff(actual) > 0
                predicted_direction = np.diff(predicted) > 0
                direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
            else:
                direction_accuracy = 0
            
            # Prediction accuracy within tolerance
            tolerance_levels = [0.01, 0.05, 0.1]  # 1%, 5%, 10%
            tolerance_accuracies = {}
            
            for tolerance in tolerance_levels:
                within_tolerance = np.abs((predicted - actual) / actual) <= tolerance
                accuracy = np.mean(within_tolerance) * 100
                tolerance_accuracies[f'accuracy_{int(tolerance*100)}pct'] = accuracy
            
            # Bias and variance
            bias = np.mean(predicted - actual)
            variance = np.var(predicted - actual)
            
            # Correlation
            correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0
            
            metrics = {
                'model_name': model_name,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'mape': mape,
                'direction_accuracy': direction_accuracy,
                'bias': bias,
                'variance': variance,
                'correlation': correlation,
                **tolerance_accuracies
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating model performance metrics: {str(e)}")
            return {}
    
    def create_performance_chart(self, data, price_col='Close'):
        """
        Create comprehensive performance visualization
        
        Args:
            data (DataFrame): Stock data
            price_col (str): Price column name
            
        Returns:
            plotly.graph_objects.Figure: Performance chart
        """
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Price Performance', 'Returns Distribution',
                    'Cumulative Returns', 'Rolling Volatility',
                    'Drawdown', 'Risk-Return Scatter'
                ),
                specs=[
                    [{"secondary_y": False}, {"type": "histogram"}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"type": "scatter"}]
                ]
            )
            
            prices = data[price_col].dropna()
            daily_returns = prices.pct_change().dropna()
            
            # Price performance
            fig.add_trace(
                go.Scatter(x=prices.index, y=prices, name='Price', line=dict(color='blue')),
                row=1, col=1
            )
            
            # Returns distribution
            fig.add_trace(
                go.Histogram(x=daily_returns * 100, name='Daily Returns (%)', nbinsx=50),
                row=1, col=2
            )
            
            # Cumulative returns
            cumulative_returns = (1 + daily_returns).cumprod() - 1
            fig.add_trace(
                go.Scatter(x=cumulative_returns.index, y=cumulative_returns * 100, 
                          name='Cumulative Returns (%)', line=dict(color='green')),
                row=2, col=1
            )
            
            # Rolling volatility
            rolling_vol = daily_returns.rolling(window=30).std() * np.sqrt(self.trading_days) * 100
            fig.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol, 
                          name='30-Day Rolling Volatility (%)', line=dict(color='red')),
                row=2, col=2
            )
            
            # Drawdown
            cumulative_returns_for_dd = (1 + daily_returns).cumprod()
            running_max = cumulative_returns_for_dd.expanding().max()
            drawdown = (cumulative_returns_for_dd - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown, 
                          name='Drawdown (%)', fill='tozeroy', line=dict(color='red')),
                row=3, col=1
            )
            
            # Risk-Return scatter (yearly data if available)
            if len(daily_returns) >= self.trading_days:
                yearly_returns = []
                yearly_volatilities = []
                
                for year in range(len(daily_returns) // self.trading_days):
                    start_idx = year * self.trading_days
                    end_idx = (year + 1) * self.trading_days
                    year_data = daily_returns.iloc[start_idx:end_idx]
                    
                    if len(year_data) > 0:
                        yearly_return = year_data.mean() * self.trading_days * 100
                        yearly_vol = year_data.std() * np.sqrt(self.trading_days) * 100
                        yearly_returns.append(yearly_return)
                        yearly_volatilities.append(yearly_vol)
                
                if yearly_returns and yearly_volatilities:
                    fig.add_trace(
                        go.Scatter(x=yearly_volatilities, y=yearly_returns, 
                                  mode='markers', name='Yearly Risk-Return',
                                  marker=dict(size=10, color='purple')),
                        row=3, col=2
                    )
            
            fig.update_layout(
                height=900,
                showlegend=True,
                title_text="Comprehensive Performance Analysis"
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating performance chart: {str(e)}")
            return None
    
    def create_risk_dashboard(self, data, price_col='Close'):
        """
        Create risk analysis dashboard
        
        Args:
            data (DataFrame): Stock data
            price_col (str): Price column name
            
        Returns:
            plotly.graph_objects.Figure: Risk dashboard
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Value at Risk (VaR)', 'Maximum Drawdown Analysis',
                    'Risk Metrics Timeline', 'Return vs Risk Comparison'
                )
            )
            
            prices = data[price_col].dropna()
            daily_returns = prices.pct_change().dropna()
            
            # VaR analysis
            var_levels = [1, 5, 10, 25]
            var_values = [np.percentile(daily_returns, level) * 100 for level in var_levels]
            
            fig.add_trace(
                go.Bar(x=[f'{level}%' for level in var_levels], y=var_values,
                      name='VaR', marker_color='red'),
                row=1, col=1
            )
            
            # Maximum Drawdown
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max * 100
            
            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown, 
                          name='Drawdown (%)', fill='tozeroy', line=dict(color='red')),
                row=1, col=2
            )
            
            # Rolling risk metrics
            window = 60  # 60-day rolling window
            rolling_vol = daily_returns.rolling(window=window).std() * np.sqrt(self.trading_days) * 100
            rolling_sharpe = (daily_returns.rolling(window=window).mean() / 
                            daily_returns.rolling(window=window).std() * np.sqrt(self.trading_days))
            
            fig.add_trace(
                go.Scatter(x=rolling_vol.index, y=rolling_vol,
                          name='Rolling Volatility (%)', line=dict(color='blue')),
                row=2, col=1
            )
            
            # Add secondary y-axis for Sharpe ratio
            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe,
                          name='Rolling Sharpe Ratio', line=dict(color='green', dash='dash'),
                          yaxis='y4'),
                row=2, col=1
            )
            
            # Risk-Return comparison with benchmarks
            annual_return = daily_returns.mean() * self.trading_days * 100
            annual_vol = daily_returns.std() * np.sqrt(self.trading_days) * 100
            
            # Benchmark comparisons (hypothetical)
            benchmarks = {
                'Stock': (annual_return, annual_vol),
                'Conservative Portfolio': (6, 8),
                'Moderate Portfolio': (8, 12),
                'Aggressive Portfolio': (12, 18),
                'Market Index': (10, 15)
            }
            
            returns_list = [ret for ret, vol in benchmarks.values()]
            vols_list = [vol for ret, vol in benchmarks.values()]
            names_list = list(benchmarks.keys())
            
            fig.add_trace(
                go.Scatter(x=vols_list, y=returns_list,
                          mode='markers+text', text=names_list,
                          textposition='top center',
                          marker=dict(size=12, color=['red', 'blue', 'green', 'orange', 'purple']),
                          name='Risk-Return Comparison'),
                row=2, col=2
            )
            
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Risk Analysis Dashboard"
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Confidence Level", row=1, col=1)
            fig.update_yaxes(title_text="VaR (%)", row=1, col=1)
            
            fig.update_xaxes(title_text="Date", row=1, col=2)
            fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
            
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            
            fig.update_xaxes(title_text="Volatility (%)", row=2, col=2)
            fig.update_yaxes(title_text="Annual Return (%)", row=2, col=2)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating risk dashboard: {str(e)}")
            return None
    
    def generate_performance_report(self, data, price_col='Close'):
        """
        Generate comprehensive performance report
        
        Args:
            data (DataFrame): Stock data
            price_col (str): Price column name
            
        Returns:
            dict: Comprehensive performance report
        """
        try:
            # Calculate all metrics
            metrics = self.calculate_all_metrics(data, price_col)
            
            # Generate insights and recommendations
            insights = self.generate_insights(metrics)
            
            report = {
                'summary': {
                    'report_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'data_period': f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}",
                    'total_days': len(data),
                    'current_price': metrics.get('current_price', 0)
                },
                'performance_metrics': metrics,
                'insights': insights,
                'risk_assessment': self.assess_risk_level(metrics),
                'recommendations': self.generate_recommendations(metrics)
            }
            
            return report
            
        except Exception as e:
            st.error(f"Error generating performance report: {str(e)}")
            return {}
    
    def generate_insights(self, metrics):
        """Generate insights from calculated metrics"""
        insights = []
        
        try:
            # Price performance insights
            if metrics.get('total_return', 0) > 0:
                insights.append(f"Positive total return of {metrics['total_return']:.2f}%")
            else:
                insights.append(f"Negative total return of {metrics['total_return']:.2f}%")
            
            # Volatility insights
            volatility = metrics.get('volatility', 0)
            if volatility > 30:
                insights.append(f"High volatility at {volatility:.2f}% indicates significant price swings")
            elif volatility < 15:
                insights.append(f"Low volatility at {volatility:.2f}% suggests stable price movement")
            else:
                insights.append(f"Moderate volatility at {volatility:.2f}%")
            
            # Sharpe ratio insights
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe > 2:
                insights.append("Excellent risk-adjusted returns (Sharpe ratio > 2)")
            elif sharpe > 1:
                insights.append("Good risk-adjusted returns (Sharpe ratio > 1)")
            elif sharpe > 0:
                insights.append("Positive but modest risk-adjusted returns")
            else:
                insights.append("Poor risk-adjusted returns (negative Sharpe ratio)")
            
            # Drawdown insights
            max_dd = abs(metrics.get('max_drawdown', 0))
            if max_dd > 30:
                insights.append(f"Significant maximum drawdown of {max_dd:.2f}% indicates high risk")
            elif max_dd > 15:
                insights.append(f"Moderate maximum drawdown of {max_dd:.2f}%")
            else:
                insights.append(f"Low maximum drawdown of {max_dd:.2f}% suggests good downside protection")
            
            # Win rate insights
            win_rate = metrics.get('win_rate', 0)
            if win_rate > 60:
                insights.append(f"High win rate of {win_rate:.1f}% indicates consistent positive performance")
            elif win_rate > 40:
                insights.append(f"Balanced win rate of {win_rate:.1f}%")
            else:
                insights.append(f"Low win rate of {win_rate:.1f}% suggests frequent negative days")
            
        except Exception as e:
            insights.append(f"Error generating insights: {str(e)}")
        
        return insights
    
    def assess_risk_level(self, metrics):
        """Assess overall risk level based on metrics"""
        try:
            risk_score = 0
            risk_factors = []
            
            # Volatility factor
            volatility = metrics.get('volatility', 0)
            if volatility > 30:
                risk_score += 3
                risk_factors.append("High volatility")
            elif volatility > 20:
                risk_score += 2
                risk_factors.append("Moderate volatility")
            else:
                risk_score += 1
                risk_factors.append("Low volatility")
            
            # Maximum drawdown factor
            max_dd = abs(metrics.get('max_drawdown', 0))
            if max_dd > 30:
                risk_score += 3
                risk_factors.append("High maximum drawdown")
            elif max_dd > 15:
                risk_score += 2
                risk_factors.append("Moderate maximum drawdown")
            else:
                risk_score += 1
                risk_factors.append("Low maximum drawdown")
            
            # Sharpe ratio factor (inverted)
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe < 0:
                risk_score += 3
                risk_factors.append("Negative risk-adjusted returns")
            elif sharpe < 1:
                risk_score += 2
                risk_factors.append("Poor risk-adjusted returns")
            else:
                risk_score += 1
                risk_factors.append("Good risk-adjusted returns")
            
            # Determine risk level
            if risk_score <= 4:
                risk_level = "Low"
            elif risk_score <= 7:
                risk_level = "Moderate"
            else:
                risk_level = "High"
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'max_score': 9
            }
            
        except Exception as e:
            return {
                'risk_level': "Unknown",
                'risk_score': 0,
                'risk_factors': [f"Error assessing risk: {str(e)}"],
                'max_score': 9
            }
    
    def generate_recommendations(self, metrics):
        """Generate investment recommendations based on metrics"""
        recommendations = []
        
        try:
            # Based on Sharpe ratio
            sharpe = metrics.get('sharpe_ratio', 0)
            if sharpe < 0:
                recommendations.append("Consider reducing position size due to negative risk-adjusted returns")
            elif sharpe < 0.5:
                recommendations.append("Monitor closely - risk-adjusted returns are below average")
            elif sharpe > 2:
                recommendations.append("Strong risk-adjusted performance - consider maintaining or increasing position")
            
            # Based on volatility
            volatility = metrics.get('volatility', 0)
            if volatility > 30:
                recommendations.append("High volatility suggests using position sizing and stop-loss strategies")
            elif volatility < 10:
                recommendations.append("Low volatility may indicate potential for higher position sizes")
            
            # Based on maximum drawdown
            max_dd = abs(metrics.get('max_drawdown', 0))
            if max_dd > 25:
                recommendations.append("High drawdown risk - consider implementing strict risk management")
            
            # Based on win rate
            win_rate = metrics.get('win_rate', 0)
            if win_rate < 40:
                recommendations.append("Low win rate suggests need for better entry/exit timing")
            elif win_rate > 70:
                recommendations.append("High win rate indicates good timing - maintain current strategy")
            
            # Based on recent performance
            if 'price_change_30d_pct' in metrics:
                recent_perf = metrics['price_change_30d_pct']
                if recent_perf > 10:
                    recommendations.append("Strong recent performance - consider taking partial profits")
                elif recent_perf < -10:
                    recommendations.append("Recent underperformance - evaluate fundamental factors")
            
            if not recommendations:
                recommendations.append("Performance metrics are within normal ranges - maintain current strategy")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {str(e)}")
        
        return recommendations
