import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

def create_candlestick_chart(data, title="Stock Price Chart"):
    """
    Create an interactive candlestick chart with volume
    
    Args:
        data (DataFrame): Stock data with OHLCV columns
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Candlestick chart
    """
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price",
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Volume chart
        if 'Volume' in data.columns:
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(data['Close'], data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
        
        # Add moving averages if available
        if 'MA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_20'],
                    mode='lines',
                    name='MA 20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        if 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MA_50'],
                    mode='lines',
                    name='MA 50',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=True,
            hovermode='x unified',
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="gray",
                font_size=12,
                font_family="Arial"
            )
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating candlestick chart: {str(e)}")
        return None

def create_pl_table(data, title="Profit & Loss Analysis"):
    """
    Create an interactive P&L visualization
    
    Args:
        data (DataFrame): Data with P&L calculations
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: P&L chart
    """
    try:
        # Extract timeframe from title for subplot titles
        timeframe = "Daily"
        if "Weekly" in title:
            timeframe = "Weekly"
        elif "Monthly" in title:
            timeframe = "Monthly"
        elif "Yearly" in title:
            timeframe = "Yearly"
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(f'{timeframe} P&L', 'Cumulative P&L', 'P&L Distribution', 'Win/Loss Ratio'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # P&L with correct timeframe
        colors = ['green' if val > 0 else 'red' for val in data['PL_Value']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['PL_Value'],
                name=f'{timeframe} P&L',
                marker_color=colors,
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Cumulative P&L
        cumulative_pl = data['PL_Value'].cumsum()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=cumulative_pl,
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        # P&L Distribution
        fig.add_trace(
            go.Histogram(
                x=data['PL_Value'],
                name='P&L Distribution',
                nbinsx=30,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Win/Loss Ratio Pie Chart
        positive_days = len(data[data['PL_Value'] > 0])
        negative_days = len(data[data['PL_Value'] < 0])
        neutral_days = len(data[data['PL_Value'] == 0])
        
        fig.add_trace(
            go.Pie(
                labels=['Positive Days', 'Negative Days', 'Neutral Days'],
                values=[positive_days, negative_days, neutral_days],
                name="Win/Loss Ratio",
                marker_colors=['green', 'red', 'gray']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=700,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating P&L chart: {str(e)}")
        return None

def create_comparison_chart(comparison_data, title="Year-over-Year Monthly Comparison"):
    """
    Create a heatmap for year-over-year comparison
    
    Args:
        comparison_data (DataFrame): Year-month comparison data
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Heatmap chart
    """
    try:
        fig = go.Figure(data=go.Heatmap(
            z=comparison_data.values,
            x=comparison_data.columns,
            y=comparison_data.index,
            colorscale='RdYlGn',
            colorbar=dict(title="Monthly Return (%)"),
            hoverongaps=False,
            text=comparison_data.values.round(2),
            texttemplate="%{text}%",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Month",
            yaxis_title="Year",
            height=500,
            width=800
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating comparison chart: {str(e)}")
        return None

def create_anomaly_chart(data, anomaly_data, title="Anomaly Detection Results"):
    """
    Create an anomaly detection visualization
    
    Args:
        data (DataFrame): Original stock data
        anomaly_data (DataFrame): Data with anomaly flags
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Anomaly chart
    """
    try:
        fig = go.Figure()
        
        # Normal data points
        normal_data = anomaly_data[anomaly_data['Anomaly'] == False]
        fig.add_trace(
            go.Scatter(
                x=normal_data.index,
                y=normal_data['Close'],
                mode='lines',
                name='Normal',
                line=dict(color='blue', width=1),
                opacity=0.7
            )
        )
        
        # Anomaly data points
        anomaly_points = anomaly_data[anomaly_data['Anomaly'] == True]
        if len(anomaly_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points.index,
                    y=anomaly_points['Close'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='x',
                        line=dict(width=2, color='darkred')
                    )
                )
            )
        
        # Add anomaly scores if available
        if 'Anomaly_Score' in anomaly_data.columns and not anomaly_data['Anomaly_Score'].isna().all():
            # Create secondary y-axis for anomaly scores
            fig2 = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add all traces to the subplot
            for trace in fig.data:
                fig2.add_trace(trace, secondary_y=False)
            
            # Add anomaly scores
            fig2.add_trace(
                go.Scatter(
                    x=anomaly_data.index,
                    y=anomaly_data['Anomaly_Score'],
                    mode='lines',
                    name='Anomaly Score',
                    line=dict(color='orange', width=1, dash='dash'),
                    opacity=0.5
                ),
                secondary_y=True
            )
            
            fig2.update_xaxes(title_text="Date")
            fig2.update_yaxes(title_text="Price ($)", secondary_y=False)
            fig2.update_yaxes(title_text="Anomaly Score", secondary_y=True)
            
            fig2.update_layout(
                title=title,
                height=500,
                hovermode='x unified'
            )
            
            return fig2
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating anomaly chart: {str(e)}")
        return None

def create_technical_indicators_chart(data, title="Technical Indicators"):
    """
    Create a comprehensive technical indicators chart
    
    Args:
        data (DataFrame): Stock data with technical indicators
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Technical indicators chart
    """
    try:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=('Price & Moving Averages', 'RSI', 'MACD', 'Bollinger Bands'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price and Moving Averages
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='blue')),
            row=1, col=1
        )
        
        if 'MA_20' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MA_20'], mode='lines', name='MA 20', line=dict(color='red')),
                row=1, col=1
            )
        
        if 'MA_50' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='MA 50', line=dict(color='green')),
                row=1, col=1
            )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI', line=dict(color='purple')),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD', line=dict(color='blue')),
                row=3, col=1
            )
            if 'MACD_Signal' in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal', line=dict(color='red')),
                    row=3, col=1
                )
            if 'MACD_Histogram' in data.columns:
                fig.add_trace(
                    go.Bar(x=data.index, y=data['MACD_Histogram'], name='Histogram', opacity=0.6),
                    row=3, col=1
                )
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(
                go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='blue')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Upper'], mode='lines', name='BB Upper', line=dict(color='red', dash='dash')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Middle'], mode='lines', name='BB Middle', line=dict(color='orange')),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['BB_Lower'], mode='lines', name='BB Lower', line=dict(color='green', dash='dash')),
                row=4, col=1
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating technical indicators chart: {str(e)}")
        return None

def create_model_performance_chart(performances, title="Model Performance Comparison"):
    """
    Create a model performance comparison chart
    
    Args:
        performances (dict): Dictionary of model performances
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Performance comparison chart
    """
    try:
        models = list(performances.keys())
        accuracies = [performances[model].get('accuracy', 0) for model in models]
        rmse_values = [performances[model].get('rmse', 0) for model in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Accuracy Comparison', 'RMSE Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(
                x=models,
                y=accuracies,
                name='Accuracy (%)',
                marker_color='blue',
                text=accuracies,
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # RMSE comparison
        fig.add_trace(
            go.Bar(
                x=models,
                y=rmse_values,
                name='RMSE',
                marker_color='red',
                text=rmse_values,
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating model performance chart: {str(e)}")
        return None

def create_prediction_chart(actual_data, predictions, title="Prediction Results"):
    """
    Create a chart showing actual vs predicted values
    
    Args:
        actual_data (DataFrame): Actual stock data
        predictions (array): Predicted values
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Prediction chart
    """
    try:
        fig = go.Figure()
        
        # Actual data
        fig.add_trace(
            go.Scatter(
                x=actual_data.index,
                y=actual_data['Close'],
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            )
        )
        
        # Predicted data
        if len(predictions) > 0:
            # Create index for predictions (assuming they're for the last N days)
            pred_index = actual_data.index[-len(predictions):]
            
            fig.add_trace(
                go.Scatter(
                    x=pred_index,
                    y=predictions,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash')
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating prediction chart: {str(e)}")
        return None

def create_volatility_chart(data, title="Volatility Analysis"):
    """
    Create a volatility analysis chart
    
    Args:
        data (DataFrame): Stock data with volatility
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Volatility chart
    """
    try:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volatility')
        )
        
        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Volatility chart
        if 'Volatility' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Volatility'],
                    mode='lines',
                    name='Volatility',
                    line=dict(color='red'),
                    fill='tozeroy',
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            height=600,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating volatility chart: {str(e)}")
        return None

def create_correlation_heatmap(data, title="Feature Correlation Heatmap"):
    """
    Create a correlation heatmap for features
    
    Args:
        data (DataFrame): Stock data
        title (str): Chart title
        
    Returns:
        plotly.graph_objects.Figure: Correlation heatmap
    """
    try:
        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            width=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return None
