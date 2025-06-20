import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import os

# Import utility modules
from utils.data_loader import DataLoader
from utils.data_processor import DataProcessor
from utils.ml_models.minimal_models import MinimalModelManager as ModelManager
from utils.visualizations import create_comparison_chart, create_pl_table, create_anomaly_chart
from utils.config_manager import ConfigManager
from utils.performance_metrics import PerformanceMetrics
from utils.backtesting_engine import BacktestingEngine
from utils.technical_indicators import TechnicalIndicators

# Page configuration
st.set_page_config(
    page_title="Advanced Financial Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = "YFinance"
if 'symbol' not in st.session_state:
    st.session_state.symbol = ""
if 'period' not in st.session_state:
    st.session_state.period = "1y"

# Initialize utility classes
data_loader = DataLoader()
data_processor = DataProcessor()
model_manager = ModelManager()
config_manager = ConfigManager()
performance_metrics = PerformanceMetrics()
backtesting_engine = BacktestingEngine()
technical_indicators = TechnicalIndicators()

def main():
    st.title("📈 Advanced Financial Analysis Platform")
    st.markdown("---")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Data Input & Analysis", "ML Model Configuration", "Performance Dashboard"]
        )
    
    if page == "Data Input & Analysis":
        data_input_page()
    elif page == "ML Model Configuration":
        model_config_page()
    else:
        performance_dashboard_page()

def data_input_page():
    # Data source selection
    st.header("📊 Data Input & Analysis")
    
    data_source = st.radio(
        "Select Data Source",
        ["YFinance", "File Upload"],
        key="data_source_radio"
    )
    
    st.session_state.data_source = data_source
    
    col1, col2 = st.columns([2, 1])
    
    if data_source == "YFinance":
        with col1:
            display_yfinance_interface()
    else:
        with col1:
            display_file_upload_interface()
    
    with col2:
        if st.button("🔄 Clear All Data", type="secondary"):
            clear_data()
            st.rerun()
    
    # Display data analysis if data is available
    if st.session_state.data is not None:
        display_data_analysis()

def display_yfinance_interface():
    st.subheader("YFinance Data Retrieval")
    
    # Stock symbol input
    symbol_input = st.text_input(
        "Enter Stock Symbol",
        value=st.session_state.symbol,
        placeholder="e.g., AAPL, MSFT, GOOGL"
    )
    symbol = symbol_input.upper() if symbol_input else ""
    
    # Period selection
    col1, col2 = st.columns(2)
    
    with col1:
        period_type = st.selectbox(
            "Period Type",
            ["Predefined", "Custom Range"]
        )
    
    selected_period = None
    with col2:
        if period_type == "Predefined":
            selected_period = st.selectbox(
                "Select Period",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                index=5  # Default to 1y
            )
        else:
            st.write("Custom Date Range")
    
    if period_type == "Custom Range":
        col3, col4 = st.columns(2)
        with col3:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        with col4:
            end_date = st.date_input("End Date", datetime.now())
        period = None
    else:
        start_date = None
        end_date = None
        period = selected_period
    
    if st.button("📥 Download Data", type="primary"):
        if symbol:
            with st.spinner("Downloading data from YFinance..."):
                try:
                    period_param = period if 'period' in locals() and period else None
                    data = data_loader.load_yfinance_data(symbol, period_param, start_date, end_date)
                    if data is not None and not data.empty:
                        st.session_state.data = data
                        st.session_state.symbol = symbol
                        st.session_state.period = period_param if period_param else f"{start_date} to {end_date}"
                        
                        # Process data
                        st.session_state.processed_data = data_processor.process_stock_data(data)
                        
                        st.success(f"✅ Data downloaded successfully for {symbol}")
                        
                        # Display data info
                        display_data_info(data, symbol)
                        st.rerun()
                    else:
                        st.error("❌ No data found for the specified symbol and period")
                except Exception as e:
                    st.error(f"❌ Error downloading data: {str(e)}")
        else:
            st.warning("⚠️ Please enter a stock symbol")

def display_file_upload_interface():
    st.subheader("File Upload")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )
    
    if uploaded_file is not None:
        if st.button("📤 Process File", type="primary"):
            with st.spinner("Processing uploaded file..."):
                try:
                    data = data_loader.load_file_data(uploaded_file)
                    if data is not None and not data.empty:
                        st.session_state.data = data
                        st.session_state.processed_data = data_processor.process_stock_data(data)
                        st.success("✅ File processed successfully")
                        
                        # Display data info
                        display_data_info(data, "Uploaded File")
                        st.rerun()
                    else:
                        st.error("❌ Unable to process the uploaded file")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")

def help_info(text, key=None):
    """Create an info icon with expandable help text"""
    with st.expander("ℹ️ Help", expanded=False):
        st.info(text)

def display_data_info(data, source):
    """Display information about the loaded data"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Overview")
        help_info("Shows basic information about your loaded stock data including number of records, date range, and available columns.", "data_overview_help")
        
        st.write(f"**Data Source:** {source}")
        st.write(f"**Total Records:** {len(data):,}")
        st.write(f"**Date Range:** {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
        st.write(f"**Columns:** {', '.join(data.columns.tolist())}")
        
        # Show current price with date
        current_price = data['Close'].iloc[-1]
        current_date = data.index[-1].strftime('%b %d, %Y')
        st.write(f"**Current Price:** ${current_price:.2f} (as of {current_date})")
    
    with col2:
        st.subheader("🔍 Data Quality")
        help_info("Data quality check shows if there are any missing values in your dataset. Missing data can affect analysis accuracy.", "data_quality_help")
        
        missing_values = data.isnull().sum()
        if missing_values.sum() > 0:
            st.write("**Missing Values:**")
            for column, missing_count in missing_values.items():
                if missing_count > 0:
                    st.write(f"  - {column}: {missing_count}")
        else:
            st.write("✅ No missing values detected")

def display_data_analysis():
    """Display comprehensive data analysis"""
    st.header("📊 Data Analysis & Insights")
    
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        
        # Display data range information at the top
        try:
            start_ts = pd.Timestamp(data.index.min())
            end_ts = pd.Timestamp(data.index.max())
            total_years = (end_ts - start_ts).days / 365.25
            start_date_full = start_ts.strftime('%b %Y')
            end_date_full = end_ts.strftime('%b %Y')
            
            with st.expander("📊 Data Range Information", expanded=True):
                st.info(f"📅 **Historical data available:** {start_date_full} to {end_date_full} ({total_years:.1f}+ years)")
                st.info(f"📊 **Selected period:** {start_date_full} to {end_date_full}")
        except Exception:
            with st.expander("📊 Data Range Information", expanded=True):
                st.info(f"📅 **Data loaded successfully:** {len(data)} records")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📈 Raw Data", 
            "📊 Technical Analysis",
            "💰 P&L Analysis", 
            "🔮 ML Predictions", 
            "📅 Year Comparison", 
            "🚨 Anomaly Detection",
            "📊 Backtesting"
        ])
        
        with tab1:
            display_raw_data_tab(data)
        
        with tab2:
            display_technical_analysis_tab(data)
        
        with tab3:
            display_pl_analysis_tab(data)
        
        with tab4:
            display_ml_predictions_tab(data)
        
        with tab5:
            display_year_comparison_tab(data)
        
        with tab6:
            display_anomaly_detection_tab(data)
        
        with tab7:
            display_backtesting_tab(data)

def display_raw_data_tab(data):
    """Display raw data with expandable sections"""
    st.subheader("📈 Stock Data Overview")
    help_info("This chart shows the raw stock price data including open, high, low, close prices and trading volume. Candlestick patterns help identify market trends and potential reversal points.", "raw_data_help")
    
    if data is None or len(data) < 2:
        st.warning("⚠️ Insufficient data for analysis. Need at least 2 data points.")
        return
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['Close'].iloc[-1]
        current_date = data.index[-1].strftime('%b %d, %Y')
        st.metric("Current Price", f"${current_price:.2f}")
        st.caption(f"as of {current_date}")
    
    with col2:
        if len(data) >= 2:
            change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
            change_pct = (change/data['Close'].iloc[-2]*100)
            st.metric("Daily Change", f"${change:.2f}", f"{change_pct:.2f}%")
        else:
            st.metric("Daily Change", "N/A")
    
    with col3:
        st.metric("Period High", f"${data['High'].max():.2f}")
    with col4:
        st.metric("Period Low", f"${data['Low'].min():.2f}")
    
    # Interactive chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))
    
    fig.update_layout(
        title="Stock Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Expandable data view
    with st.expander("📋 View Raw Data"):
        st.dataframe(data.tail(100), use_container_width=True)

def display_pl_analysis_tab(data):
    """Display profit and loss analysis with different time periods"""
    st.subheader("💰 Profit & Loss Analysis")
    help_info("Profit & Loss analysis shows price changes and percentage returns across different time periods. This helps identify profitable trading periods and overall investment performance at various timeframes.", "pl_analysis_help")
    
    if data is None or len(data) < 2:
        st.warning("⚠️ Insufficient data for P&L analysis. Need at least 2 data points.")
        return
    
    # Create sub-tabs for different time periods
    pl_tab1, pl_tab2, pl_tab3, pl_tab4 = st.tabs([
        "📅 Daily P&L", 
        "📊 Weekly P&L", 
        "📈 Monthly P&L", 
        "📆 Yearly P&L"
    ])
    
    with pl_tab1:
        display_daily_pl(data)
    
    with pl_tab2:
        display_weekly_pl(data)
    
    with pl_tab3:
        display_monthly_pl(data)
    
    with pl_tab4:
        display_yearly_pl(data)

def display_daily_pl(data):
    """Display daily P&L analysis"""
    st.subheader("📅 Daily Profit & Loss Analysis")
    
    try:
        # Calculate daily P&L
        pl_data = data_processor.calculate_pl(data)
        
        if pl_data is None or len(pl_data) == 0:
            st.warning("⚠️ Insufficient data for daily P&L analysis.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_pl = pl_data['PL_Value'].sum()
        positive_days = len(pl_data[pl_data['PL_Value'] > 0])
        negative_days = len(pl_data[pl_data['PL_Value'] < 0])
        win_rate = positive_days / len(pl_data) * 100 if len(pl_data) > 0 else 0
        
        with col1:
            st.metric("Total Daily P&L", f"${total_pl:.2f}")
        with col2:
            st.metric("Daily Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Positive Days", positive_days)
        with col4:
            st.metric("Negative Days", negative_days)
        
        # Daily P&L Chart
        fig = create_pl_table(pl_data, "Daily Profit & Loss Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("📊 Daily P&L Details"):
            st.dataframe(pl_data.tail(50), use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in daily P&L analysis: {str(e)}")

def display_weekly_pl(data):
    """Display weekly P&L analysis"""
    st.subheader("📊 Weekly Profit & Loss Analysis")
    
    try:
        # Create weekly aggregation
        weekly_data = data_processor.create_weekly_aggregation(data)
        
        if weekly_data is None or len(weekly_data) < 2:
            st.warning("⚠️ Insufficient data for weekly P&L analysis. Need at least 2 weeks of data.")
            return
        
        # Calculate weekly P&L
        weekly_pl = data_processor.calculate_pl(weekly_data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_pl = weekly_pl['PL_Value'].sum()
        positive_weeks = len(weekly_pl[weekly_pl['PL_Value'] > 0])
        negative_weeks = len(weekly_pl[weekly_pl['PL_Value'] < 0])
        win_rate = positive_weeks / len(weekly_pl) * 100 if len(weekly_pl) > 0 else 0
        
        with col1:
            st.metric("Total Weekly P&L", f"${total_pl:.2f}")
        with col2:
            st.metric("Weekly Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Positive Weeks", positive_weeks)
        with col4:
            st.metric("Negative Weeks", negative_weeks)
        
        # Weekly P&L Chart
        fig = create_pl_table(weekly_pl, "Weekly Profit & Loss Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("📊 Weekly P&L Details"):
            st.dataframe(weekly_pl, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in weekly P&L analysis: {str(e)}")

def display_monthly_pl(data):
    """Display monthly P&L analysis"""
    st.subheader("📈 Monthly Profit & Loss Analysis")
    
    try:
        # Create monthly aggregation
        monthly_data = data_processor.create_monthly_aggregation(data)
        
        if monthly_data is None or len(monthly_data) < 2:
            st.warning("⚠️ Insufficient data for monthly P&L analysis. Need at least 2 months of data.")
            return
        
        # Calculate monthly P&L
        monthly_pl = data_processor.calculate_pl(monthly_data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_pl = monthly_pl['PL_Value'].sum()
        positive_months = len(monthly_pl[monthly_pl['PL_Value'] > 0])
        negative_months = len(monthly_pl[monthly_pl['PL_Value'] < 0])
        win_rate = positive_months / len(monthly_pl) * 100 if len(monthly_pl) > 0 else 0
        
        with col1:
            st.metric("Total Monthly P&L", f"${total_pl:.2f}")
        with col2:
            st.metric("Monthly Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Positive Months", positive_months)
        with col4:
            st.metric("Negative Months", negative_months)
        
        # Monthly P&L Chart
        fig = create_pl_table(monthly_pl, "Monthly Profit & Loss Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("📊 Monthly P&L Details"):
            st.dataframe(monthly_pl, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in monthly P&L analysis: {str(e)}")

def display_yearly_pl(data):
    """Display yearly P&L analysis"""
    st.subheader("📆 Yearly Profit & Loss Analysis")
    
    try:
        # Create yearly aggregation
        yearly_data = data_processor.create_yearly_aggregation(data)
        
        if yearly_data is None or len(yearly_data) < 2:
            st.warning("⚠️ Insufficient data for yearly P&L analysis. Need at least 2 years of data.")
            return
        
        # Calculate yearly P&L
        yearly_pl = data_processor.calculate_pl(yearly_data)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_pl = yearly_pl['PL_Value'].sum()
        positive_years = len(yearly_pl[yearly_pl['PL_Value'] > 0])
        negative_years = len(yearly_pl[yearly_pl['PL_Value'] < 0])
        win_rate = positive_years / len(yearly_pl) * 100 if len(yearly_pl) > 0 else 0
        
        with col1:
            st.metric("Total Yearly P&L", f"${total_pl:.2f}")
        with col2:
            st.metric("Yearly Win Rate", f"{win_rate:.1f}%")
        with col3:
            st.metric("Positive Years", positive_years)
        with col4:
            st.metric("Negative Years", negative_years)
        
        # Yearly P&L Chart
        fig = create_pl_table(yearly_pl, "Yearly Profit & Loss Analysis")
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("📊 Yearly P&L Details"):
            st.dataframe(yearly_pl, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error in yearly P&L analysis: {str(e)}")

def display_ml_predictions_tab(data):
    """Display ML predictions and model results"""
    st.subheader("🔮 Machine Learning Predictions")
    help_info("Machine Learning predictions use historical data patterns to forecast future stock prices. Different models have varying strengths - some excel at short-term predictions while others are better for long-term trends. Results should be used as guidance, not absolute investment advice.", "ml_predictions_help")
    
    # Model selection
    model_categories = model_manager.get_available_models()
    
    col1, col2 = st.columns(2)
    with col1:
        selected_category = st.selectbox("Select Model Category", list(model_categories.keys()))
    with col2:
        selected_model = st.selectbox("Select Model", model_categories[selected_category])
    
    if st.button("🚀 Train & Predict", type="primary"):
        with st.spinner(f"Training {selected_model} model..."):
            try:
                # Train model and make predictions
                model_results = model_manager.train_and_predict(
                    data, selected_category, selected_model
                )
                
                if model_results:
                    st.success("✅ Model trained successfully!")
                    
                    # Display predictions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Predicted Next Price", f"${model_results['next_price']:.2f}")
                        st.metric("Prediction Confidence", f"{model_results['confidence']:.1f}%")
                    
                    with col2:
                        st.metric("Model Accuracy", f"{model_results['accuracy']:.2f}%")
                        st.metric("RMSE", f"{model_results['rmse']:.4f}")
                    
                    # Prediction chart
                    if 'predictions' in model_results:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data.index[-len(model_results['predictions']):],
                            y=data['Close'].iloc[-len(model_results['predictions']):],
                            mode='lines',
                            name='Actual',
                            line=dict(color='blue')
                        ))
                        fig.add_trace(go.Scatter(
                            x=data.index[-len(model_results['predictions']):],
                            y=model_results['predictions'],
                            mode='lines',
                            name='Predicted',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_model} Predictions vs Actual",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error training model: {str(e)}")

def display_year_comparison_tab(data):
    """Display year-over-year comparison"""
    st.subheader("📅 Year-over-Year Comparison")
    help_info("Year-over-year comparison shows how the stock performed in each month across different years. This heatmap reveals seasonal patterns and helps identify the best and worst performing months historically. Darker colors indicate higher returns.", "year_comparison_help")
    
    # Create year-over-year comparison
    comparison_data = data_processor.create_year_comparison(data)
    
    if comparison_data is not None:
        # Create heatmap
        fig = create_comparison_chart(comparison_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Year summary table
        with st.expander("📊 Year Summary Statistics"):
            help_info("Yearly statistics show annual performance metrics including total return, volatility, best/worst months, and Sharpe ratio for risk-adjusted returns.", "yearly_stats_help")
            yearly_stats = data_processor.calculate_yearly_stats(data)
            st.dataframe(yearly_stats, use_container_width=True)
    else:
        st.info("ℹ️ Insufficient data for year-over-year comparison. Need at least 2 years of data.")

def display_anomaly_detection_tab(data):
    """Display anomaly detection results"""
    st.subheader("🚨 Anomaly Detection")
    help_info("Anomaly detection identifies unusual price movements that deviate significantly from normal patterns. These could indicate important market events, news impact, or trading opportunities. Different methods have varying sensitivity levels.", "anomaly_detection_help")
    
    # Anomaly detection controls
    col1, col2 = st.columns(2)
    
    with col1:
        anomaly_method = st.selectbox(
            "Detection Method",
            ["Isolation Forest", "Statistical", "DBSCAN", "One-Class SVM"],
            help="Choose detection algorithm: Statistical uses standard deviation, Isolation Forest uses machine learning"
        )
    
    with col2:
        sensitivity = st.slider("Sensitivity", 0.01, 0.5, 0.1, 0.01, help="Lower values detect fewer, more significant anomalies")
    
    if st.button("🔍 Detect Anomalies", type="primary"):
        with st.spinner("Detecting anomalies..."):
            try:
                anomalies = data_processor.detect_anomalies(data, anomaly_method, sensitivity)
                
                if anomalies is not None:
                    anomaly_count = len(anomalies[anomalies['Anomaly'] == True])
                    st.metric("Anomalies Detected", anomaly_count)
                    
                    # Anomaly chart
                    fig = create_anomaly_chart(data, anomalies)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Anomalies table
                    if anomaly_count > 0:
                        with st.expander("🚨 Anomaly Details"):
                            anomaly_details = anomalies[anomalies['Anomaly'] == True]
                            st.dataframe(anomaly_details, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error detecting anomalies: {str(e)}")

def model_config_page():
    """ML Model Configuration Page"""
    st.header("⚙️ ML Model Configuration")
    
    # Model categories
    categories = model_manager.get_available_models()
    
    for category, models in categories.items():
        with st.expander(f"🔧 {category} Models"):
            st.write(f"Available models: {', '.join(models)}")
            
            # Configuration for each model
            for model in models:
                st.subheader(f"{model} Configuration")
                config = config_manager.get_model_config(category, model)
                
                # Display current configuration
                if config:
                    st.json(config)
                else:
                    st.info("Using default configuration")
                
                # Allow parameter modification
                if st.button(f"Configure {model}", key=f"config_{model}"):
                    st.write("Configuration interface would be implemented here")

def performance_dashboard_page():
    """Performance Dashboard Page"""
    st.header("📊 Performance Dashboard")
    
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        
        # Performance metrics
        metrics = performance_metrics.calculate_all_metrics(data)
        
        # Display metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
        with col2:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2%}")
        with col3:
            st.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
        with col4:
            st.metric("Total Return", f"{metrics.get('total_return', 0):.2%}")
        
        # Performance charts
        fig = performance_metrics.create_performance_chart(data)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("ℹ️ Please load data first to view performance metrics")

def display_technical_analysis_tab(data):
    """Display comprehensive technical analysis with interactive indicators"""
    st.subheader("📊 Technical Analysis & Trading Signals")
    help_info("Technical analysis uses mathematical calculations based on price and volume to identify trading opportunities. Combine multiple indicators for more reliable signals and better market timing.", "technical_analysis_help")
    
    if data is None or len(data) < 50:
        st.warning("⚠️ Insufficient data for technical analysis. Need at least 50 days of historical data.")
        return
    
    # Indicator selection interface
    st.subheader("🎯 Select Technical Indicators")
    
    # Get indicator categories
    indicator_categories = technical_indicators.get_indicator_categories()
    
    # Create columns for indicator selection
    cols = st.columns(len(indicator_categories))
    selected_indicators = []
    
    for i, (category, indicators) in enumerate(indicator_categories.items()):
        with cols[i]:
            st.write(f"**{category}**")
            category_selected = []
            
            for indicator in indicators:
                if st.checkbox(indicator, key=f"indicator_{indicator}"):
                    category_selected.append(indicator)
                    selected_indicators.append(indicator)
            
            # Quick select all for category
            if st.button(f"Select All {category}", key=f"select_all_{i}"):
                st.rerun()
    
    # Quick preset buttons
    st.subheader("🔧 Quick Presets")
    preset_cols = st.columns(4)
    
    with preset_cols[0]:
        if st.button("📈 Trend Analysis", type="secondary"):
            selected_indicators = ['SMA_20', 'SMA_50', 'EMA_20', 'BB', 'MACD', 'VWAP']
    
    with preset_cols[1]:
        if st.button("⚡ Momentum Signals", type="secondary"):
            selected_indicators = ['RSI', 'Stoch_K', 'Williams_R', 'CCI', 'MFI']
    
    with preset_cols[2]:
        if st.button("📊 Volume Analysis", type="secondary"):
            selected_indicators = ['Volume', 'OBV', 'Volume_SMA', 'VWAP']
    
    with preset_cols[3]:
        if st.button("🎯 Complete Setup", type="secondary"):
            selected_indicators = ['SMA_20', 'SMA_50', 'BB', 'RSI', 'MACD', 'Volume', 'VWAP']
    
    if selected_indicators:
        # Calculate indicators
        with st.spinner("Calculating technical indicators..."):
            try:
                # Calculate all indicators
                analyzed_data = technical_indicators.calculate_all_indicators(data)
                
                # Generate trading signals
                signal_data = technical_indicators.generate_trading_signals(analyzed_data)
                
                # Display main technical analysis chart
                st.subheader("📈 Interactive Technical Chart")
                
                # Chart configuration
                col1, col2 = st.columns(2)
                with col1:
                    timeframe = st.selectbox("Timeframe", ["All Data", "Last 6 Months", "Last 3 Months", "Last Month"])
                with col2:
                    chart_height = st.slider("Chart Height", 400, 1200, 800, 100)
                
                # Filter data based on timeframe
                if timeframe == "Last 6 Months":
                    chart_data = analyzed_data.tail(126)
                elif timeframe == "Last 3 Months":
                    chart_data = analyzed_data.tail(63)
                elif timeframe == "Last Month":
                    chart_data = analyzed_data.tail(21)
                else:
                    chart_data = analyzed_data
                
                # Create and display technical analysis chart
                fig = technical_indicators.create_technical_analysis_chart(
                    chart_data, 
                    selected_indicators, 
                    f"Technical Analysis - {st.session_state.symbol if st.session_state.symbol else 'Stock'}"
                )
                fig.update_layout(height=chart_height)
                st.plotly_chart(fig, use_container_width=True)
                
                # Current indicator readings
                st.subheader("📊 Current Market Analysis")
                
                # Get indicator summary
                indicator_summary = technical_indicators.get_indicator_summary(analyzed_data)
                
                # Display summary in columns
                summary_cols = st.columns(len(indicator_summary))
                for i, (indicator, reading) in enumerate(indicator_summary.items()):
                    with summary_cols[i % len(summary_cols)]:
                        if "Bullish" in reading or "Uptrend" in reading or "Oversold" in reading:
                            st.success(f"**{indicator}**\n{reading}")
                        elif "Bearish" in reading or "Downtrend" in reading or "Overbought" in reading:
                            st.error(f"**{indicator}**\n{reading}")
                        else:
                            st.info(f"**{indicator}**\n{reading}")
                
                # Trading signals analysis
                st.subheader("🎯 Trading Signals & Opportunities")
                
                # Get recent signals
                recent_signals = signal_data.tail(20)
                
                # Count signals
                total_buy_signals = recent_signals['Buy_Signal'].sum()
                total_sell_signals = recent_signals['Sell_Signal'].sum()
                signal_strength = recent_signals['Signal_Strength'].iloc[-1] if not recent_signals.empty else 0
                
                # Display signal summary
                signal_cols = st.columns(4)
                
                with signal_cols[0]:
                    st.metric("Buy Signals (20 days)", int(total_buy_signals))
                
                with signal_cols[1]:
                    st.metric("Sell Signals (20 days)", int(total_sell_signals))
                
                with signal_cols[2]:
                    st.metric("Current Signal Strength", f"{signal_strength:.1f}")
                
                with signal_cols[3]:
                    if signal_strength > 2:
                        st.success("**Strong Buy**")
                    elif signal_strength > 0:
                        st.info("**Weak Buy**")
                    elif signal_strength < -2:
                        st.error("**Strong Sell**")
                    elif signal_strength < 0:
                        st.warning("**Weak Sell**")
                    else:
                        st.info("**Neutral**")
                
                # Detailed indicator values
                with st.expander("📋 Detailed Indicator Values"):
                    latest_values = analyzed_data.tail(1)
                    
                    # Create indicator details table
                    indicator_details = []
                    for indicator in selected_indicators:
                        matching_columns = [col for col in latest_values.columns if indicator.replace('_', '') in col.replace('_', '') or indicator in col]
                        
                        for col in matching_columns:
                            if col in latest_values.columns:
                                value = latest_values[col].iloc[0]
                                if pd.notna(value):
                                    indicator_details.append({
                                        'Indicator': col,
                                        'Current Value': f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                                    })
                    
                    if indicator_details:
                        indicator_df = pd.DataFrame(indicator_details)
                        st.dataframe(indicator_df, use_container_width=True)
                
                # Pattern recognition
                st.subheader("🔍 Pattern Recognition")
                
                pattern_cols = st.columns(3)
                
                with pattern_cols[0]:
                    # Support and Resistance
                    if 'Pivot' in analyzed_data.columns:
                        current_price = analyzed_data['Close'].iloc[-1]
                        pivot = analyzed_data['Pivot'].iloc[-1]
                        r1 = analyzed_data['R1'].iloc[-1] if 'R1' in analyzed_data.columns else None
                        s1 = analyzed_data['S1'].iloc[-1] if 'S1' in analyzed_data.columns else None
                        
                        st.info(f"**Support/Resistance**\nPrice: ${current_price:.2f}\nPivot: ${pivot:.2f}")
                        if r1: st.write(f"Resistance 1: ${r1:.2f}")
                        if s1: st.write(f"Support 1: ${s1:.2f}")
                
                with pattern_cols[1]:
                    # Trend Analysis
                    if all(col in analyzed_data.columns for col in ['SMA_20', 'SMA_50']):
                        sma20 = analyzed_data['SMA_20'].iloc[-1]
                        sma50 = analyzed_data['SMA_50'].iloc[-1]
                        current_price = analyzed_data['Close'].iloc[-1]
                        
                        if current_price > sma20 > sma50:
                            st.success("**Strong Uptrend**\nPrice > SMA20 > SMA50")
                        elif current_price > sma20:
                            st.info("**Weak Uptrend**\nPrice > SMA20")
                        elif current_price < sma20 < sma50:
                            st.error("**Strong Downtrend**\nPrice < SMA20 < SMA50")
                        else:
                            st.warning("**Weak Downtrend**\nPrice < SMA20")
                
                with pattern_cols[2]:
                    # Volatility Analysis
                    if 'ATR' in analyzed_data.columns:
                        atr = analyzed_data['ATR'].iloc[-1]
                        atr_pct = (atr / analyzed_data['Close'].iloc[-1]) * 100
                        
                        if atr_pct > 3:
                            st.warning(f"**High Volatility**\nATR: {atr_pct:.1f}%")
                        elif atr_pct < 1:
                            st.info(f"**Low Volatility**\nATR: {atr_pct:.1f}%")
                        else:
                            st.success(f"**Normal Volatility**\nATR: {atr_pct:.1f}%")
                
                # Export functionality
                st.subheader("💾 Export Analysis")
                
                export_cols = st.columns(2)
                
                with export_cols[0]:
                    if st.button("📊 Export Indicator Data"):
                        # Create downloadable CSV
                        export_data = analyzed_data[['Open', 'High', 'Low', 'Close', 'Volume'] + 
                                                  [col for col in analyzed_data.columns if any(ind in col for ind in selected_indicators)]]
                        csv = export_data.to_csv()
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"technical_analysis_{st.session_state.symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                
                with export_cols[1]:
                    if st.button("📈 Export Chart Image"):
                        st.info("Chart export functionality can be accessed using the Plotly chart toolbar above.")
                
            except Exception as e:
                st.error(f"❌ Error calculating technical indicators: {str(e)}")
                st.info("Please ensure you have sufficient data and try selecting fewer indicators.")
    
    else:
        st.info("ℹ️ Please select technical indicators to display the analysis chart.")
        
        # Show sample indicators preview
        st.subheader("📋 Available Technical Indicators")
        
        indicator_info = {
            "Trend Indicators": {
                "Moving Averages (SMA/EMA)": "Smooth price data to identify trend direction",
                "Bollinger Bands": "Volatility bands showing overbought/oversold conditions",
                "MACD": "Moving Average Convergence Divergence for trend changes",
                "Ichimoku Cloud": "Comprehensive trend and momentum system",
                "VWAP": "Volume Weighted Average Price for institutional trading levels"
            },
            "Momentum Indicators": {
                "RSI": "Relative Strength Index (14-period) for overbought/oversold levels",
                "Stochastic": "Momentum oscillator comparing closing price to price range",
                "Williams %R": "Momentum indicator similar to Stochastic",
                "CCI": "Commodity Channel Index for cyclical trends",
                "Money Flow Index": "Volume-weighted RSI for buying/selling pressure"
            },
            "Volume Indicators": {
                "On-Balance Volume": "Cumulative volume indicator for trend confirmation",
                "Volume SMA": "Moving average of volume for volume trend analysis"
            }
        }
        
        for category, indicators in indicator_info.items():
            with st.expander(f"📊 {category}"):
                for name, description in indicators.items():
                    st.write(f"**{name}**: {description}")

def display_backtesting_tab(data):
    """Display backtesting results and strategy comparison"""
    st.subheader("📊 Trading Strategy Backtesting")
    help_info("Backtesting validates trading strategies using historical data to see how they would have performed. Compare multiple strategies side-by-side to identify the most effective approach for your investment goals.", "backtesting_help")
    
    if data is None or len(data) < 30:
        st.warning("⚠️ Insufficient data for backtesting. Need at least 30 days of historical data.")
        return
    
    # Backtesting configuration
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capital = st.number_input("Initial Capital ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
        transaction_cost = st.slider("Transaction Cost (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
    
    with col2:
        backtesting_period = st.selectbox(
            "Backtesting Period",
            ["Full Period", "Last 2 Years", "Last 1 Year", "Last 6 Months"],
            help="Choose the time period for backtesting analysis"
        )
        
        # Adjust data based on selected period
        backtest_data = data.copy()
        if backtesting_period == "Last 2 Years":
            backtest_data = data.tail(504)  # ~2 years of trading days
        elif backtesting_period == "Last 1 Year":
            backtest_data = data.tail(252)  # ~1 year of trading days
        elif backtesting_period == "Last 6 Months":
            backtest_data = data.tail(126)  # ~6 months of trading days
    
    # Initialize backtesting engine with user parameters
    backtesting_engine_custom = BacktestingEngine(initial_capital=initial_capital, transaction_cost=transaction_cost)
    
    # Strategy selection
    st.subheader("🎯 Strategy Selection")
    predefined_strategies = backtesting_engine_custom.get_predefined_strategies()
    
    selected_strategies = []
    
    # Create columns for strategy selection
    cols = st.columns(2)
    
    for i, strategy in enumerate(predefined_strategies):
        col_idx = i % 2
        with cols[col_idx]:
            if st.checkbox(f"**{strategy['name']}**", key=f"strategy_{i}"):
                st.caption(strategy['description'])
                selected_strategies.append(strategy)
    
    if selected_strategies:
        if st.button("🚀 Run Backtest Analysis", type="primary"):
            with st.spinner("Running backtesting analysis..."):
                try:
                    # Run strategy comparison
                    comparison_results = backtesting_engine_custom.compare_strategies(backtest_data, selected_strategies)
                    
                    if comparison_results and comparison_results['comparison_table'] is not None:
                        # Display results
                        st.subheader("📈 Strategy Performance Comparison")
                        
                        # Performance summary table
                        comparison_df = comparison_results['comparison_table']
                        st.dataframe(comparison_df.round(2), use_container_width=True)
                        
                        # Best strategy highlight
                        if comparison_results['best_strategy']:
                            best_strategy = comparison_results['best_strategy']
                            st.success(f"🏆 **Best Performing Strategy:** {best_strategy['name']}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Return", f"{best_strategy['metrics'].get('Total Return (%)', 0):.2f}%")
                            with col2:
                                st.metric("Sharpe Ratio", f"{best_strategy['metrics'].get('Sharpe Ratio', 0):.2f}")
                            with col3:
                                st.metric("Max Drawdown", f"{best_strategy['metrics'].get('Max Drawdown (%)', 0):.2f}%")
                        
                        # Create and display comparison charts
                        charts = backtesting_engine_custom.create_comparison_charts(comparison_results)
                        
                        if 'performance' in charts:
                            st.subheader("📊 Portfolio Value Comparison")
                            st.plotly_chart(charts['performance'], use_container_width=True)
                        
                        if 'metrics' in charts:
                            st.subheader("📈 Performance Metrics Comparison")
                            st.plotly_chart(charts['metrics'], use_container_width=True)
                        
                        # Individual strategy details
                        st.subheader("🔍 Individual Strategy Analysis")
                        
                        for strategy_name, result in comparison_results['individual_results'].items():
                            with st.expander(f"📊 {strategy_name} Details"):
                                
                                # Strategy metrics
                                metrics = result['metrics']
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%")
                                with col2:
                                    st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1f}%")
                                with col3:
                                    st.metric("Total Trades", f"{metrics.get('total_trades', 0)}")
                                with col4:
                                    st.metric("Avg Trade Return", f"{metrics.get('avg_trade_return', 0):.2f}%")
                                
                                # Trade details
                                if not result['trades'].empty:
                                    st.subheader("🔄 Trading History")
                                    trades_df = result['trades'].round(2)
                                    st.dataframe(trades_df, use_container_width=True)
                                    
                                    # Trade statistics
                                    profitable_trades = len(trades_df[trades_df['Return_%'] > 0])
                                    total_trades = len(trades_df)
                                    if total_trades > 0:
                                        win_rate = profitable_trades / total_trades * 100
                                        avg_holding_days = trades_df['Holding_Days'].mean()
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.info(f"📊 Win Rate: {win_rate:.1f}% ({profitable_trades}/{total_trades} trades)")
                                        with col2:
                                            st.info(f"⏱️ Average Holding Period: {avg_holding_days:.1f} days")
                        
                        # Risk Analysis
                        st.subheader("⚠️ Risk Analysis Summary")
                        
                        risk_comparison = []
                        for strategy_name, result in comparison_results['individual_results'].items():
                            metrics = result['metrics']
                            risk_comparison.append({
                                'Strategy': strategy_name,
                                'Max Drawdown (%)': metrics.get('max_drawdown_pct', 0),
                                'Volatility (%)': metrics.get('volatility_pct', 0),
                                'Best Day (%)': metrics.get('best_day', 0),
                                'Worst Day (%)': metrics.get('worst_day', 0)
                            })
                        
                        risk_df = pd.DataFrame(risk_comparison)
                        st.dataframe(risk_df.round(2), use_container_width=True)
                        
                        # Investment recommendations
                        st.subheader("💡 Investment Insights")
                        
                        best_return_strategy = comparison_df.loc[comparison_df['Total Return (%)'].idxmax(), 'Strategy']
                        best_sharpe_strategy = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax(), 'Strategy']
                        lowest_drawdown_strategy = comparison_df.loc[comparison_df['Max Drawdown (%)'].idxmin(), 'Strategy']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.info(f"🎯 **Highest Return:** {best_return_strategy}")
                        with col2:
                            st.info(f"📈 **Best Risk-Adjusted:** {best_sharpe_strategy}")
                        with col3:
                            st.info(f"🛡️ **Lowest Risk:** {lowest_drawdown_strategy}")
                    
                    else:
                        st.error("❌ No valid backtest results generated. Please check your data and strategy selection.")
                        
                except Exception as e:
                    st.error(f"❌ Error running backtesting analysis: {str(e)}")
    
    else:
        st.info("ℹ️ Please select at least one strategy to run backtesting analysis.")

def clear_data():
    """Clear all session state data"""
    st.session_state.data = None
    st.session_state.processed_data = None
    st.session_state.symbol = ""
    st.session_state.period = "1y"

if __name__ == "__main__":
    main()
