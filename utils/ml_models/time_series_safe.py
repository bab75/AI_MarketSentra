import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import streamlit as st

# Safe imports for time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

class TimeSeriesModels:
    """Time Series Models for Stock Prediction (Safe Version)"""
    
    def __init__(self):
        self.available_models = ['Linear Trend', 'Moving Average']
        self.trained_models = {}
        self.model_performances = {}
        
        # Add models based on available libraries
        if STATSMODELS_AVAILABLE:
            self.available_models.extend(['ARIMA', 'SARIMA'])
        if PMDARIMA_AVAILABLE:
            self.available_models.append('Auto ARIMA')
        if PROPHET_AVAILABLE:
            self.available_models.append('Prophet')
            
        if len(self.available_models) == 2:
            st.info("Advanced time series libraries not available. Using basic models.")
    
    def get_available_models(self):
        """Return list of available time series models"""
        return self.available_models
    
    def train_model(self, model_name, data, **kwargs):
        """
        Train a time series model
        
        Args:
            model_name (str): Name of the model
            data (DataFrame): Stock data with datetime index
            **kwargs: Additional parameters
            
        Returns:
            dict: Training results
        """
        try:
            # Ensure data has datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                data.index = pd.to_datetime(data.index)
            
            # Use Close price for time series analysis
            ts_data = data['Close'].dropna()
            
            if model_name == 'Linear Trend':
                return self._train_linear_trend(ts_data)
            elif model_name == 'Moving Average':
                return self._train_moving_average(ts_data, **kwargs)
            elif model_name == 'ARIMA' and STATSMODELS_AVAILABLE:
                return self._train_arima(ts_data, **kwargs)
            elif model_name == 'Prophet' and PROPHET_AVAILABLE:
                return self._train_prophet(data, **kwargs)
            else:
                return {
                    'error': f'Model {model_name} not available',
                    'model_name': model_name,
                    'predictions': [],
                    'rmse': float('inf')
                }
                
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None
    
    def _train_linear_trend(self, ts_data):
        """Train linear trend model"""
        try:
            # Create time index
            X = np.arange(len(ts_data)).reshape(-1, 1)
            y = ts_data.values
            
            # Train linear regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y, predictions))
            
            # Future predictions (next 5 days)
            future_X = np.arange(len(ts_data), len(ts_data) + 5).reshape(-1, 1)
            future_pred = model.predict(future_X)
            
            results = {
                'model_name': 'Linear Trend',
                'rmse': rmse,
                'predictions': future_pred.tolist(),
                'trend_slope': model.coef_[0],
                'trend_intercept': model.intercept_
            }
            
            self.trained_models['Linear Trend'] = model
            self.model_performances['Linear Trend'] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in linear trend model: {str(e)}")
            return None
    
    def _train_moving_average(self, ts_data, window=20):
        """Train moving average model"""
        try:
            # Calculate moving average
            ma = ts_data.rolling(window=window).mean()
            
            # Use last moving average value for prediction
            last_ma = ma.iloc[-1]
            
            # Simple prediction: extend last MA value
            predictions = [last_ma] * 5
            
            # Calculate RMSE on available data
            valid_ma = ma.dropna()
            valid_actual = ts_data[ma.notna()]
            
            if len(valid_ma) > 0:
                rmse = np.sqrt(mean_squared_error(valid_actual, valid_ma))
            else:
                rmse = float('inf')
            
            results = {
                'model_name': 'Moving Average',
                'window': window,
                'rmse': rmse,
                'predictions': predictions,
                'last_ma': last_ma
            }
            
            self.model_performances['Moving Average'] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in moving average model: {str(e)}")
            return None
    
    def _train_arima(self, ts_data, order=(1, 1, 1)):
        """Train ARIMA model"""
        if not STATSMODELS_AVAILABLE:
            return None
            
        try:
            # Fit ARIMA model
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=5)
            
            # Calculate in-sample RMSE
            residuals = fitted_model.resid
            rmse = np.sqrt(np.mean(residuals ** 2))
            
            results = {
                'model_name': 'ARIMA',
                'order': order,
                'aic': fitted_model.aic,
                'rmse': rmse,
                'predictions': forecast.tolist()
            }
            
            self.trained_models['ARIMA'] = fitted_model
            self.model_performances['ARIMA'] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in ARIMA model: {str(e)}")
            return None
    
    def _train_prophet(self, data, **kwargs):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            return None
            
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data['Close']
            })
            
            # Initialize and fit Prophet
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            model.fit(prophet_data)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=5)
            forecast = model.predict(future)
            
            # Get predictions for next 5 days
            future_pred = forecast.tail(5)['yhat'].tolist()
            
            # Calculate RMSE on historical data
            historical_pred = forecast[:-5]['yhat']
            actual = prophet_data['y']
            rmse = np.sqrt(mean_squared_error(actual, historical_pred))
            
            results = {
                'model_name': 'Prophet',
                'rmse': rmse,
                'predictions': future_pred
            }
            
            self.trained_models['Prophet'] = model
            self.model_performances['Prophet'] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in Prophet model: {str(e)}")
            return None
    
    def get_model_performance(self, model_name):
        """Get performance metrics for a specific model"""
        return self.model_performances.get(model_name, {})
    
    def get_all_performances(self):
        """Get all model performances"""
        return self.model_performances