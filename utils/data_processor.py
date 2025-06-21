import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import streamlit as st

class DataProcessor:
    """Handle data processing and analysis"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def process_stock_data(self, data):
        """
        Process raw stock data and add calculated fields
        
        Args:
            data (pandas.DataFrame): Raw stock data
            
        Returns:
            pandas.DataFrame: Processed data with additional columns
        """
        try:
            processed_data = data.copy()
            
            # Calculate daily returns
            processed_data['Daily_Return'] = processed_data['Close'].pct_change()
            
            # Calculate moving averages
            processed_data['MA_5'] = processed_data['Close'].rolling(window=5).mean()
            processed_data['MA_10'] = processed_data['Close'].rolling(window=10).mean()
            processed_data['MA_20'] = processed_data['Close'].rolling(window=20).mean()
            processed_data['MA_50'] = processed_data['Close'].rolling(window=50).mean()
            
            # Calculate volatility (rolling standard deviation)
            processed_data['Volatility'] = processed_data['Daily_Return'].rolling(window=20).std()
            
            # Calculate RSI
            processed_data['RSI'] = self.calculate_rsi(processed_data['Close'])
            
            # Calculate MACD
            macd_data = self.calculate_macd(processed_data['Close'])
            processed_data['MACD'] = macd_data['MACD']
            processed_data['MACD_Signal'] = macd_data['Signal']
            processed_data['MACD_Histogram'] = macd_data['Histogram']
            
            # Calculate Bollinger Bands
            bb_data = self.calculate_bollinger_bands(processed_data['Close'])
            processed_data['BB_Upper'] = bb_data['Upper']
            processed_data['BB_Middle'] = bb_data['Middle']
            processed_data['BB_Lower'] = bb_data['Lower']
            
            return processed_data
            
        except Exception as e:
            st.error(f"Error processing stock data: {str(e)}")
            return data
    
    def calculate_pl(self, data):
        """
        Calculate profit and loss for each day
        
        Args:
            data (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Data with P&L calculations
        """
        try:
            pl_data = data.copy()
            
            # Calculate daily P&L (Close - Open)
            pl_data['PL_Value'] = pl_data['Close'] - pl_data['Open']
            
            # Calculate percentage P&L
            pl_data['PL_Percentage'] = (pl_data['PL_Value'] / pl_data['Open']) * 100
            
            # Add anomaly flag (initially all False)
            pl_data['Anomaly_Flag'] = False
            
            # Identify potential anomalies based on extreme price movements
            pl_threshold = pl_data['PL_Percentage'].std() * 2.5
            pl_data['Anomaly_Flag'] = abs(pl_data['PL_Percentage']) > pl_threshold
            
            # Add day of week and month for pattern analysis
            pl_data['Day_of_Week'] = pl_data.index.dayofweek
            pl_data['Month'] = pl_data.index.month
            pl_data['Year'] = pl_data.index.year
            
            return pl_data
            
        except Exception as e:
            st.error(f"Error calculating P&L: {str(e)}")
            return data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return {
            'MACD': macd,
            'Signal': signal_line,
            'Histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'Upper': upper,
            'Middle': middle,
            'Lower': lower
        }
    
    def create_year_comparison(self, data):
        """
        Create year-over-year monthly comparison data
        
        Args:
            data (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Year-month comparison matrix
        """
        try:
            # Add year and month columns
            comparison_data = data.copy()
            comparison_data['Year'] = comparison_data.index.year
            comparison_data['Month'] = comparison_data.index.month
            
            # Calculate monthly returns for each year
            monthly_returns = comparison_data.groupby(['Year', 'Month']).agg({
                'Open': 'first',
                'Close': 'last'
            })
            
            monthly_returns['Monthly_Return'] = (
                (monthly_returns['Close'] - monthly_returns['Open']) / monthly_returns['Open']
            ) * 100
            
            # Pivot to create year-month matrix
            pivot_data = monthly_returns['Monthly_Return'].reset_index().pivot(
                index='Year', columns='Month', values='Monthly_Return'
            )
            
            # Add month names
            month_names = {
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
                5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            }
            
            pivot_data.columns = [month_names.get(col, col) for col in pivot_data.columns]
            
            return pivot_data
            
        except Exception as e:
            st.error(f"Error creating year comparison: {str(e)}")
            return None
    
    def calculate_yearly_stats(self, data):
        """Calculate yearly statistics"""
        try:
            yearly_stats = data.groupby(data.index.year).agg({
                'Open': 'first',
                'Close': 'last',
                'High': 'max',
                'Low': 'min',
                'Volume': 'mean'
            })
            
            yearly_stats['Annual_Return'] = (
                (yearly_stats['Close'] - yearly_stats['Open']) / yearly_stats['Open']
            ) * 100
            
            yearly_stats['Volatility'] = data.groupby(data.index.year)['Close'].pct_change().std() * np.sqrt(252) * 100
            
            return yearly_stats.round(2)
            
        except Exception as e:
            st.error(f"Error calculating yearly stats: {str(e)}")
            return pd.DataFrame()
    
    def detect_anomalies(self, data, method='isolation_forest', sensitivity=0.1):
        """
        Detect anomalies in stock data
        
        Args:
            data (pandas.DataFrame): Stock data
            method (str): Detection method
            sensitivity (float): Sensitivity parameter
            
        Returns:
            pandas.DataFrame: Data with anomaly flags
        """
        try:
            anomaly_data = data.copy()
            
            # Prepare features for anomaly detection
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_data = anomaly_data[features].dropna()
            
            if len(feature_data) < 10:
                st.warning("Insufficient data for anomaly detection")
                return None
            
            # Scale features
            scaled_features = self.scaler.fit_transform(feature_data)
            
            # Apply selected anomaly detection method
            if method.lower() == 'isolation forest':
                detector = IsolationForest(contamination=sensitivity, random_state=42)
            elif method.lower() == 'one-class svm':
                detector = OneClassSVM(nu=sensitivity)
            elif method.lower() == 'dbscan':
                detector = DBSCAN(eps=sensitivity * 10, min_samples=5)
            else:  # Statistical method
                return self.statistical_anomaly_detection(feature_data, sensitivity)
            
            # Detect anomalies
            if method.lower() == 'dbscan':
                labels = detector.fit_predict(scaled_features)
                anomalies = labels == -1
            else:
                anomalies = detector.fit_predict(scaled_features) == -1
            
            # Add anomaly flags to original data
            anomaly_data.loc[feature_data.index, 'Anomaly'] = anomalies
            anomaly_data['Anomaly'] = anomaly_data['Anomaly'].fillna(False)
            
            return anomaly_data
            
        except Exception as e:
            st.error(f"Error detecting anomalies: {str(e)}")
            return None
    
    def statistical_anomaly_detection(self, data, sensitivity):
        """Statistical anomaly detection using z-score"""
        anomaly_data = data.copy()
        anomaly_data['Anomaly'] = False
        
        for column in ['Open', 'High', 'Low', 'Close']:
            if column in data.columns:
                z_scores = np.abs(stats.zscore(data[column].dropna()))
                threshold = stats.norm.ppf(1 - sensitivity)
                anomalies = z_scores > threshold
                anomaly_data.loc[anomalies, 'Anomaly'] = True
        
        return anomaly_data
    
    def create_weekly_aggregation(self, data):
        """
        Create weekly aggregated data for performance optimization
        
        Args:
            data (pandas.DataFrame): Daily stock data
            
        Returns:
            pandas.DataFrame: Weekly aggregated data
        """
        try:
            if data is None or data.empty:
                return None
            
            # Resample to weekly data (W = week ending on Sunday)
            weekly_data = data.resample('W').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Add weekly technical indicators
            weekly_data['Weekly_Return'] = weekly_data['Close'].pct_change() * 100
            weekly_data['Weekly_Volatility'] = weekly_data['Close'].rolling(4).std()
            weekly_data['Week_High_Low_Ratio'] = weekly_data['High'] / weekly_data['Low']
            
            return weekly_data
            
        except Exception as e:
            st.error(f"Error creating weekly aggregation: {str(e)}")
            return None
    
    def create_monthly_aggregation(self, data):
        """
        Create monthly aggregated data using actual last date for each month
        
        Args:
            data (pandas.DataFrame): Daily stock data
            
        Returns:
            pandas.DataFrame: Monthly aggregated data
        """
        try:
            if data is None or data.empty:
                return None
            
            # Group by year and month to handle monthly aggregation
            monthly_groups = data.groupby([data.index.year, data.index.month])
            monthly_data = []
            for (year, month), group in monthly_groups:
                month_end_date = group.index.max()  # Actual last date for this month
                month_summary = {
                    'Date': month_end_date,
                    'Open': group['Open'].iloc[0],
                    'High': group['High'].max(),
                    'Low': group['Low'].min(),
                    'Close': group['Close'].iloc[-1],
                    'Volume': group['Volume'].sum()
                }
                monthly_data.append(month_summary)
            
            monthly_data = pd.DataFrame(monthly_data).set_index('Date')
            
            # Add monthly technical indicators
            monthly_data['Monthly_Return'] = monthly_data['Close'].pct_change() * 100
            monthly_data['Monthly_Volatility'] = monthly_data['Close'].rolling(12).std()
            monthly_data['Month_High_Low_Ratio'] = monthly_data['High'] / monthly_data['Low']
            
            return monthly_data
            
        except Exception as e:
            st.error(f"Error creating monthly aggregation: {str(e)}")
            return None
    
    def create_yearly_aggregation(self, data):
        """
        Create yearly aggregated data using actual last date for each year
        
        Args:
            data (pandas.DataFrame): Daily stock data
            
        Returns:
            pandas.DataFrame: Yearly aggregated data
        """
        try:
            if data is None or data.empty:
                return None
            
            # Group by year and use actual last date of data for each year
            yearly_groups = data.groupby(data.index.year)
            yearly_data = []
            for year, group in yearly_groups:
                year_end_date = group.index.max()  # Actual last date for this year
                year_summary = {
                    'Date': year_end_date,
                    'Open': group['Open'].iloc[0],
                    'High': group['High'].max(),
                    'Low': group['Low'].min(),
                    'Close': group['Close'].iloc[-1],
                    'Volume': group['Volume'].sum()
                }
                yearly_data.append(year_summary)
            
            yearly_data = pd.DataFrame(yearly_data).set_index('Date')
            
            # Add yearly technical indicators
            yearly_data['Yearly_Return'] = yearly_data['Close'].pct_change() * 100
            yearly_data['Yearly_Volatility'] = yearly_data['Close'].rolling(3).std()
            yearly_data['Year_High_Low_Ratio'] = yearly_data['High'] / yearly_data['Low']
            
            return yearly_data
            
        except Exception as e:
            st.error(f"Error creating yearly aggregation: {str(e)}")
            return None
    
    def prepare_ml_features(self, data, lookback_period=30):
        """
        Prepare features for machine learning models
        
        Args:
            data (pandas.DataFrame): Stock data
            lookback_period (int): Number of days to look back for features
            
        Returns:
            tuple: (features, target) for ML training
        """
        try:
            processed_data = self.process_stock_data(data)
            
            # Technical indicators as features
            feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'Daily_Return', 'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Upper', 'BB_Middle', 'BB_Lower'
            ]
            
            # Create lagged features
            features_df = pd.DataFrame(index=processed_data.index)
            
            for col in feature_columns:
                if col in processed_data.columns:
                    for lag in range(1, lookback_period + 1):
                        features_df[f'{col}_lag_{lag}'] = processed_data[col].shift(lag)
            
            # Target: next day's closing price
            target = processed_data['Close'].shift(-1)
            
            # Remove rows with NaN values
            valid_idx = features_df.dropna().index.intersection(target.dropna().index)
            
            return features_df.loc[valid_idx], target.loc[valid_idx]
            
        except Exception as e:
            st.error(f"Error preparing ML features: {str(e)}")
            return None, None
