import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import io

class DataLoader:
    """Handle data loading from various sources"""
    
    def __init__(self):
        pass
    
    def load_yfinance_data(self, symbol, period=None, start_date=None, end_date=None):
        """
        Load stock data from Yahoo Finance
        
        Args:
            symbol (str): Stock symbol
            period (str): Period for data (1d, 5d, 1mo, etc.)
            start_date (date): Start date for custom range
            end_date (date): End date for custom range
            
        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if period:
                data = ticker.history(period=period)
            else:
                data = ticker.history(start=start_date, end=end_date + pd.Timedelta(days=1)) # Add 1 day to end_date to make it inclusive (Yahoo Finance end is exclusive)
            
            if data.empty:
                return None
            
            # Ensure proper column names
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.columns = [col.title() for col in data.columns]
            
            # Fill missing columns with NaN if not present
            for col in expected_columns:
                if col not in data.columns:
                    data[col] = pd.NA
            
            # Remove timezone info if present
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            return data[expected_columns]
            
        except Exception as e:
            st.error(f"Error loading data from Yahoo Finance: {str(e)}")
            return None
    
    def load_file_data(self, uploaded_file):
        """
        Load stock data from uploaded CSV or Excel file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            pandas.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Determine file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel files.")
                return None
            
            # Try to identify date column and set as index
            date_columns = ['Date', 'date', 'DATE', 'Datetime', 'datetime', 'DATETIME']
            date_col = None
            
            for col in date_columns:
                if col in data.columns:
                    date_col = col
                    break
            
            if date_col:
                data[date_col] = pd.to_datetime(data[date_col])
                data.set_index(date_col, inplace=True)
            else:
                # Try to convert first column to datetime if it looks like a date
                first_col = data.columns[0]
                try:
                    data[first_col] = pd.to_datetime(data[first_col])
                    data.set_index(first_col, inplace=True)
                except:
                    pass
            
            # Standardize column names (case insensitive matching)
            column_mapping = {}
            expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            for expected_col in expected_columns:
                for col in data.columns:
                    if col.lower() == expected_col.lower():
                        column_mapping[col] = expected_col
                        break
            
            # Rename columns
            data.rename(columns=column_mapping, inplace=True)
            
            # Ensure all expected columns exist
            for col in expected_columns:
                if col not in data.columns:
                    data[col] = pd.NA
            
            # Convert to numeric
            for col in expected_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            return data[expected_columns].dropna(how='all')
            
        except Exception as e:
            st.error(f"Error loading file data: {str(e)}")
            return None
    
    def validate_data(self, data):
        """
        Validate that the loaded data has the required structure
        
        Args:
            data (pandas.DataFrame): Data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if data is None or data.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.warning(f"Missing required columns: {missing_columns}")
            return False
        
        # Check if data has proper datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            st.warning("Data should have a datetime index")
            return False
        
        # Check for sufficient data points
        if len(data) < 10:
            st.warning("Insufficient data points. Need at least 10 data points.")
            return False
        
        return True
    
    def get_data_info(self, data):
        """
        Get information about the loaded data
        
        Args:
            data (pandas.DataFrame): Data to analyze
            
        Returns:
            dict: Information about the data
        """
        if data is None or data.empty:
            return {}
        
        try:
            start_date = data.index.min()
            end_date = data.index.max()
            
            # Calculate total years/months
            date_diff = end_date - start_date
            total_years = date_diff.days / 365.25
            
            info = {
                'total_records': len(data),
                'date_range': {
                    'start': start_date,
                    'end': end_date
                },
                'start_date': str(start_date.date()),
                'end_date': str(end_date.date()),
                'total_years': round(total_years, 1),
                'data_range_text': f"{start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')} ({total_years:.1f}+ years)",
                'columns': data.columns.tolist(),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.to_dict(),
                'current_price': data['Close'].iloc[-1] if 'Close' in data.columns else None,
                'current_date': str(end_date.date())
            }
            
            return info
            
        except Exception as e:
            return {
                'total_records': len(data),
                'date_range': {'start': data.index.min(), 'end': data.index.max()},
                'columns': data.columns.tolist(),
                'missing_values': data.isnull().sum().to_dict(),
                'data_types': data.dtypes.to_dict()
            }
