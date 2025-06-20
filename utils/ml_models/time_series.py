import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
import pmdarima as pm
from hmmlearn import hmm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesModels:
    """Time Series Models for Stock Prediction"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_performances = {}
    
    def get_available_models(self):
        """Return list of available time series models"""
        return [
            'ARIMA', 'SARIMA', 'Auto ARIMA', 'Exponential Smoothing',
            'VAR (Vector Autoregression)', 'Hidden Markov Model',
            'Seasonal Decomposition', 'Prophet'
        ]
    
    def train_arima_model(self, data, order=(1, 1, 1), target_col='Close'):
        """
        Train ARIMA model
        
        Args:
            data (DataFrame): Time series data
            order (tuple): ARIMA order (p, d, q)
            target_col (str): Target column name
            
        Returns:
            dict: Model results
        """
        try:
            series = data[target_col].dropna()
            
            # Fit ARIMA model
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            # Make predictions
            predictions = fitted_model.forecast(steps=30)  # 30-day forecast
            
            # Calculate in-sample fit metrics
            fitted_values = fitted_model.fittedvalues
            mse = mean_squared_error(series[1:], fitted_values)  # Skip first value due to differencing
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(series[1:], fitted_values)
            
            results = {
                'model': fitted_model,
                'predictions': predictions,
                'fitted_values': fitted_values,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'model_summary': fitted_model.summary()
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error training ARIMA model: {str(e)}")
            return None
    
    def train_sarima_model(self, data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), target_col='Close'):
        """
        Train SARIMA model
        
        Args:
            data (DataFrame): Time series data
            order (tuple): ARIMA order (p, d, q)
            seasonal_order (tuple): Seasonal order (P, D, Q, s)
            target_col (str): Target column name
            
        Returns:
            dict: Model results
        """
        try:
            series = data[target_col].dropna()
            
            # Fit SARIMA model
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            
            # Make predictions
            predictions = fitted_model.forecast(steps=30)
            
            # Calculate metrics
            fitted_values = fitted_model.fittedvalues
            mse = mean_squared_error(series[len(series)-len(fitted_values):], fitted_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(series[len(series)-len(fitted_values):], fitted_values)
            
            results = {
                'model': fitted_model,
                'predictions': predictions,
                'fitted_values': fitted_values,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error training SARIMA model: {str(e)}")
            return None
    
    def train_auto_arima_model(self, data, target_col='Close', seasonal=True):
        """
        Train Auto ARIMA model using pmdarima
        
        Args:
            data (DataFrame): Time series data
            target_col (str): Target column name
            seasonal (bool): Whether to consider seasonal patterns
            
        Returns:
            dict: Model results
        """
        try:
            series = data[target_col].dropna()
            
            # Auto ARIMA
            if seasonal and len(series) >= 24:  # Need at least 2 seasons
                model = pm.auto_arima(
                    series,
                    seasonal=True,
                    m=12,  # Monthly seasonality
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False
                )
            else:
                model = pm.auto_arima(
                    series,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False
                )
            
            # Make predictions
            predictions, conf_int = model.predict(n_periods=30, return_conf_int=True)
            
            # Calculate in-sample metrics
            fitted_values = model.fittedvalues()
            mse = mean_squared_error(series[len(series)-len(fitted_values):], fitted_values)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(series[len(series)-len(fitted_values):], fitted_values)
            
            results = {
                'model': model,
                'predictions': predictions,
                'confidence_intervals': conf_int,
                'fitted_values': fitted_values,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'aic': model.aic(),
                'order': model.order,
                'seasonal_order': model.seasonal_order if seasonal else None
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error training Auto ARIMA model: {str(e)}")
            return None
    
    def train_exponential_smoothing_model(self, data, target_col='Close', seasonal='add'):
        """
        Train Exponential Smoothing model
        
        Args:
            data (DataFrame): Time series data
            target_col (str): Target column name
            seasonal (str): Type of seasonal component ('add', 'mul', or None)
            
        Returns:
            dict: Model results
        """
        try:
            series = data[target_col].dropna()
            
            # Determine seasonal periods
            seasonal_periods = 12 if len(series) >= 24 else None
            
            if seasonal_periods and seasonal:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods
                )
            else:
                model = ExponentialSmoothing(series, trend='add')
            
            fitted_model = model.fit()
            
            # Make predictions
            predictions = fitted_model.forecast(steps=30)
            
            # Calculate metrics
            fitted_values = fitted_model.fittedvalues
            mse = mean_squared_error(series[1:], fitted_values[1:])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(series[1:], fitted_values[1:])
            
            results = {
                'model': fitted_model,
                'predictions': predictions,
                'fitted_values': fitted_values,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'aic': fitted_model.aic,
                'parameters': fitted_model.params
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error training Exponential Smoothing model: {str(e)}")
            return None
    
    def train_var_model(self, data, maxlags=15, target_cols=['Open', 'High', 'Low', 'Close']):
        """
        Train Vector Autoregression (VAR) model
        
        Args:
            data (DataFrame): Time series data
            maxlags (int): Maximum number of lags to consider
            target_cols (list): List of target columns
            
        Returns:
            dict: Model results
        """
        try:
            # Prepare multivariate data
            series_data = data[target_cols].dropna()
            
            if len(series_data) < maxlags + 10:
                st.warning("Insufficient data for VAR model")
                return None
            
            # Fit VAR model
            model = VAR(series_data)
            
            # Select optimal lag order
            lag_order = model.select_order(maxlags=maxlags)
            optimal_lag = lag_order.aic
            
            fitted_model = model.fit(optimal_lag)
            
            # Make predictions
            predictions = fitted_model.forecast(series_data.values[-optimal_lag:], steps=30)
            
            # Calculate metrics for each variable
            fitted_values = fitted_model.fittedvalues
            metrics = {}
            
            for i, col in enumerate(target_cols):
                actual_values = series_data[col].iloc[optimal_lag:]
                fitted_col = fitted_values.iloc[:, i]
                
                mse = mean_squared_error(actual_values, fitted_col)
                metrics[col] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'mae': mean_absolute_error(actual_values, fitted_col)
                }
            
            results = {
                'model': fitted_model,
                'predictions': predictions,
                'fitted_values': fitted_values,
                'metrics': metrics,
                'optimal_lag': optimal_lag,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error training VAR model: {str(e)}")
            return None
    
    def train_hmm_model(self, data, n_states=3, target_col='Close'):
        """
        Train Hidden Markov Model
        
        Args:
            data (DataFrame): Time series data
            n_states (int): Number of hidden states
            target_col (str): Target column name
            
        Returns:
            dict: Model results
        """
        try:
            # Prepare data - use returns instead of prices
            series = data[target_col].dropna()
            returns = series.pct_change().dropna()
            
            # Reshape for HMM
            X = returns.values.reshape(-1, 1)
            
            # Fit Gaussian HMM
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", random_state=42)
            model.fit(X)
            
            # Predict hidden states
            hidden_states = model.predict(X)
            
            # Generate future predictions (simplified approach)
            # Sample from the learned distribution
            future_samples = model.sample(30)[0].flatten()
            
            # Convert returns back to prices
            last_price = series.iloc[-1]
            predicted_prices = [last_price]
            
            for return_val in future_samples:
                next_price = predicted_prices[-1] * (1 + return_val)
                predicted_prices.append(next_price)
            
            predictions = np.array(predicted_prices[1:])  # Exclude initial price
            
            # Calculate log-likelihood as a measure of fit
            log_likelihood = model.score(X)
            
            results = {
                'model': model,
                'predictions': predictions,
                'hidden_states': hidden_states,
                'log_likelihood': log_likelihood,
                'transition_matrix': model.transmat_,
                'means': model.means_,
                'covariances': model.covars_,
                'n_states': n_states
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error training HMM model: {str(e)}")
            return None
    
    def seasonal_decomposition(self, data, target_col='Close', model='additive', period=12):
        """
        Perform seasonal decomposition
        
        Args:
            data (DataFrame): Time series data
            target_col (str): Target column name
            model (str): 'additive' or 'multiplicative'
            period (int): Seasonal period
            
        Returns:
            dict: Decomposition results
        """
        try:
            series = data[target_col].dropna()
            
            if len(series) < 2 * period:
                st.warning(f"Insufficient data for seasonal decomposition. Need at least {2 * period} observations.")
                return None
            
            # Perform decomposition
            decomposition = seasonal_decompose(series, model=model, period=period)
            
            # Simple forecast based on trend and seasonal components
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal.dropna()
            
            # Extrapolate trend (simple linear extrapolation)
            if len(trend) >= 2:
                trend_slope = (trend.iloc[-1] - trend.iloc[-2])
                future_trend = [trend.iloc[-1] + (i + 1) * trend_slope for i in range(30)]
            else:
                future_trend = [trend.iloc[-1]] * 30
            
            # Repeat seasonal pattern
            seasonal_cycle = seasonal.iloc[-period:].values
            future_seasonal = np.tile(seasonal_cycle, (30 // period) + 1)[:30]
            
            # Combine trend and seasonal for forecast
            if model == 'additive':
                predictions = np.array(future_trend) + future_seasonal
            else:
                predictions = np.array(future_trend) * future_seasonal
            
            results = {
                'decomposition': decomposition,
                'predictions': predictions,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'model_type': model,
                'period': period
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error in seasonal decomposition: {str(e)}")
            return None
    
    def train_model(self, model_name, data, **kwargs):
        """
        Train a specific time series model
        
        Args:
            model_name (str): Name of the model to train
            data (DataFrame): Time series data
            **kwargs: Additional model parameters
            
        Returns:
            dict: Training results
        """
        try:
            if model_name == 'ARIMA':
                results = self.train_arima_model(data, **kwargs)
            elif model_name == 'SARIMA':
                results = self.train_sarima_model(data, **kwargs)
            elif model_name == 'Auto ARIMA':
                results = self.train_auto_arima_model(data, **kwargs)
            elif model_name == 'Exponential Smoothing':
                results = self.train_exponential_smoothing_model(data, **kwargs)
            elif model_name == 'VAR (Vector Autoregression)':
                results = self.train_var_model(data, **kwargs)
            elif model_name == 'Hidden Markov Model':
                results = self.train_hmm_model(data, **kwargs)
            elif model_name == 'Seasonal Decomposition':
                results = self.seasonal_decomposition(data, **kwargs)
            else:
                raise ValueError(f"Model {model_name} not implemented")
            
            if results:
                self.trained_models[model_name] = results['model']
                self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None
    
    def get_model_config(self, model_name):
        """
        Get configuration parameters for time series models
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration parameters
        """
        configs = {
            'ARIMA': {
                'order': (1, 1, 1),
                'target_col': 'Close'
            },
            'SARIMA': {
                'order': (1, 1, 1),
                'seasonal_order': (1, 1, 1, 12),
                'target_col': 'Close'
            },
            'Auto ARIMA': {
                'seasonal': True,
                'target_col': 'Close'
            },
            'Exponential Smoothing': {
                'seasonal': 'add',
                'target_col': 'Close'
            },
            'VAR (Vector Autoregression)': {
                'maxlags': 15,
                'target_cols': ['Open', 'High', 'Low', 'Close']
            },
            'Hidden Markov Model': {
                'n_states': 3,
                'target_col': 'Close'
            },
            'Seasonal Decomposition': {
                'model': 'additive',
                'period': 12,
                'target_col': 'Close'
            }
        }
        
        return configs.get(model_name, {})
