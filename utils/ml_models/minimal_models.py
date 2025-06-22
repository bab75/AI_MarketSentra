import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from hmmlearn.hmm import GaussianHMM
from .deep_learning_safe import DeepLearningModels

class MinimalModelManager:
    """Minimal ML Model Manager with core scikit-learn models and integration for deep learning"""
    
    def __init__(self):
        self.models = {
            'Classical ML': {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=1.0),
                'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
                'Support Vector Regression': SVR(kernel='rbf', C=1.0),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5)
            },
            'Ensemble Methods': {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1
                ),
                'LightGBM': LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1
                ),
                'CatBoost': CatBoostRegressor(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    l2_leaf_reg=3,
                    random_seed=42,
                    verbose=False,
                    thread_count=-1
                ),
                'Gradient Boosting Machine': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            },  
            'Clustering': {
                'K-Means': KMeans(n_clusters=3, random_state=42)
            },
            'Anomaly Detection': {
                'Isolation Forest': IsolationForest(contamination=0.1, random_state=42)
            },
            'Dimensionality Reduction': {
                'PCA': PCA(n_components=2)
            },
            'Time Series Specialized': {
                'SARIMA': 'sarima_model',
                'Exponential Smoothing': 'exp_smoothing_model', 
                'Hidden Markov Models': 'hmm_model'
            }
        }
        
        self.trained_models = {}
        self.model_performances = {}
        self.scalers = {}
        self.deep_learning_manager = DeepLearningModels()
    
    def get_available_models(self):
        """Get all available models organized by category"""
        available = {}
        for category, models in self.models.items():
            available[category] = list(models.keys())
        
        dl_models = self.deep_learning_manager.get_available_models()
        available.update(dl_models)
        
        return available
    
    def get_model_count(self):
        """Get total count of available models"""
        total = 0
        counts = {}
        for category, models in self.models.items():
            count = len(models)
            counts[category] = count
            total += count
        counts['Total'] = total
        return counts
    
    def train_and_predict(self, data, category, model_name, **kwargs):
        """Train a model and make predictions"""
        try:
            if data[['Open', 'High', 'Low', 'Close', 'Volume']].isna().any().any():
                return {
                    'error': 'Input data contains NaN values',
                    'model_name': model_name,
                    'category': category,
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            if category == 'Deep Learning Models':
                return self.deep_learning_manager.train_and_predict(data, model_name, **kwargs)
            
            if category not in self.models:
                return {
                    'error': f'Category {category} not found',
                    'model_name': model_name,
                    'category': category,
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            if model_name not in self.models[category]:
                return {
                    'error': f'Model {model_name} not found in {category}',
                    'model_name': model_name,
                    'category': category,
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            if category in ['Classical ML', 'Ensemble Methods']:
                return self._train_supervised_model(data, category, model_name, **kwargs)
            elif category == 'Clustering':
                return self._train_clustering_model(data, model_name, **kwargs)
            elif category == 'Anomaly Detection':
                return self._train_anomaly_model(data, model_name, **kwargs)
            elif category == 'Dimensionality Reduction':
                return self._train_dimensionality_model(data, model_name, **kwargs)
            elif category == 'Time Series Specialized':
                return self._train_time_series_model(data, model_name, **kwargs)
            else:
                return {
                    'error': f'Category {category} not implemented',
                    'model_name': model_name,
                    'category': category,
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
                
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return {
                'error': str(e),
                'model_name': model_name,
                'category': category,
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _prepare_features(self, data, lookback=5):
        """Prepare features for ML models"""
        try:
            features_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            features_df = features_df.copy()
            features_df['Price_Change'] = features_df['Close'].pct_change()
            features_df['Volume_Change'] = features_df['Volume'].pct_change()
            features_df['High_Low_Ratio'] = features_df['High'] / features_df['Low']
            features_df['Close_Open_Ratio'] = features_df['Close'] / features_df['Open']
            features_df['MA_5'] = features_df['Close'].rolling(5).mean()
            features_df['MA_10'] = features_df['Close'].rolling(10).mean()
            features_df = features_df.dropna()
            
            if len(features_df) < lookback + 1:
                return None, None
            
            X, y = [], []
            for i in range(lookback, len(features_df)):
                feature_row = features_df.iloc[i-lookback:i].values.flatten()
                target = features_df['Close'].iloc[i]
                X.append(feature_row)
                y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return None, None
    
    def _train_supervised_model(self, data, category, model_name, **kwargs):
        """Train supervised learning models"""
        try:
            X, y = self._prepare_features(data)
            if X is None or y is None:
                return {
                    'error': 'Could not prepare features',
                    'model_name': model_name,
                    'category': category,
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            model = self.models[category][model_name]
            model.fit(X_train, y_train)
            
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_r2 = r2_score(y_train, train_pred) * 100
            test_r2 = r2_score(y_test, test_pred) * 100
            
            latest_features = X[-1].reshape(1, -1)
            next_price = model.predict(latest_features)[0]
            
            results = {
                'model_name': model_name,
                'category': category,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'rmse': test_rmse,
                'accuracy': test_r2,
                'next_price': float(next_price),
                'confidence': float(max(0, min(100, test_r2)))
            }
            
            self.trained_models[f"{category}_{model_name}"] = model
            self.model_performances[f"{category}_{model_name}"] = results
            
            return results
            
        except Exception as e:
            return {
                'error': f'Training error: {str(e)}',
                'model_name': model_name,
                'category': category,
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_clustering_model(self, data, model_name, **kwargs):
        """Train clustering models"""
        try:
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            model = self.models['Clustering'][model_name]
            labels = model.fit_predict(scaled_features)
            
            results = {
                'model_name': model_name,
                'category': 'Clustering',
                'n_clusters': len(set(labels)),
                'labels': labels.tolist(),
                'next_price': float(data['Close'].iloc[-1]),
                'confidence': 50.0,
                'rmse': 0.0,
                'accuracy': 50.0
            }
            
            return results
            
        except Exception as e:
            return {
                'error': f'Clustering error: {str(e)}',
                'model_name': model_name,
                'category': 'Clustering',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_anomaly_model(self, data, model_name, **kwargs):
        """Train anomaly detection models"""
        try:
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            model = self.models['Anomaly Detection'][model_name]
            predictions = model.fit_predict(scaled_features)
            anomalies = predictions == -1
            
            results = {
                'model_name': model_name,
                'category': 'Anomaly Detection',
                'n_anomalies': int(np.sum(anomalies)),
                'anomaly_rate': float(np.mean(anomalies) * 100),
                'predictions': predictions.tolist(),
                'next_price': float(data['Close'].iloc[-1]),
                'confidence': float(100 - (np.mean(anomalies) * 100)),
                'rmse': 0.0,
                'accuracy': float(100 - (np.mean(anomalies) * 100))
            }
            
            return results
            
        except Exception as e:
            return {
                'error': f'Anomaly detection error: {str(e)}',
                'model_name': model_name,
                'category': 'Anomaly Detection',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_dimensionality_model(self, data, model_name, **kwargs):
        """Train dimensionality reduction models"""
        try:
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            model = self.models['Dimensionality Reduction'][model_name]
            transformed = model.fit_transform(scaled_features)
            
            results = {
                'model_name': model_name,
                'category': 'Dimensionality Reduction',
                'n_components': transformed.shape[1],
                'explained_variance': model.explained_variance_ratio_.tolist(),
                'total_variance': float(np.sum(model.explained_variance_ratio_)),
                'next_price': float(data['Close'].iloc[-1]),
                'confidence': 50.0,
                'rmse': 0.0,
                'accuracy': float(np.sum(model.explained_variance_ratio_)) * 100
            }
            
            return results
            
        except Exception as e:
            return {
                'error': f'Dimensionality reduction error: {str(e)}',
                'model_name': model_name,
                'category': 'Dimensionality Reduction',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_time_series_model(self, data, model_name, **kwargs):
        """Train specialized time series models"""
        try:
            # Prepare time series data
            ts_data = data['Close'].dropna()
            
            if model_name == 'sarima_model':
                return self._train_sarima(ts_data, **kwargs)
            elif model_name == 'exp_smoothing_model':
                return self._train_exponential_smoothing(ts_data, **kwargs)
            elif model_name == 'hmm_model':
                return self._train_hmm(data, **kwargs)
            else:
                return {
                    'error': f'Time series model {model_name} not implemented',
                    'model_name': model_name,
                    'category': 'Time Series Specialized',
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
                
        except Exception as e:
            return {
                'error': f'Time series error: {str(e)}',
                'model_name': model_name,
                'category': 'Time Series Specialized',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_sarima(self, ts_data, **kwargs):
        """Train SARIMA model"""
        try:
            # SARIMA parameters (p,d,q)(P,D,Q,s)
            order = kwargs.get('order', (1, 1, 1))
            seasonal_order = kwargs.get('seasonal_order', (1, 1, 1, 12))
            
            # Fit SARIMA model
            model = SARIMAX(ts_data, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            
            # Make forecast
            forecast = fitted_model.forecast(steps=1)
            next_price = float(forecast.iloc[0])
            
            # Calculate metrics
            residuals = fitted_model.resid
            rmse = np.sqrt(np.mean(residuals**2))
            current_price = float(ts_data.iloc[-1])
            confidence = max(0, min(100, 100 - (rmse / current_price * 100)))
            
            return {
                'model_name': 'SARIMA',
                'category': 'Time Series Specialized',
                'next_price': next_price,
                'confidence': confidence,
                'accuracy': confidence,
                'rmse': float(rmse),
                'aic': float(fitted_model.aic)
            }
            
        except Exception as e:
            return {
                'error': f'SARIMA error: {str(e)}',
                'model_name': 'SARIMA',
                'category': 'Time Series Specialized',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_exponential_smoothing(self, ts_data, **kwargs):
        """Train Exponential Smoothing model"""
        try:
            # Exponential Smoothing parameters
            trend = kwargs.get('trend', 'add')
            seasonal = kwargs.get('seasonal', 'add')
            seasonal_periods = kwargs.get('seasonal_periods', 12)
            
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(
                ts_data, 
                trend=trend, 
                seasonal=seasonal, 
                seasonal_periods=seasonal_periods
            )
            fitted_model = model.fit()
            
            # Make forecast
            forecast = fitted_model.forecast(steps=1)
            next_price = float(forecast[0])
            
            # Calculate metrics
            fitted_values = fitted_model.fittedvalues
            residuals = ts_data[len(ts_data)-len(fitted_values):] - fitted_values
            rmse = np.sqrt(np.mean(residuals**2))
            current_price = float(ts_data.iloc[-1])
            confidence = max(0, min(100, 100 - (rmse / current_price * 100)))
            
            return {
                'model_name': 'Exponential Smoothing',
                'category': 'Time Series Specialized',
                'next_price': next_price,
                'confidence': confidence,
                'accuracy': confidence,
                'rmse': float(rmse)
            }
            
        except Exception as e:
            return {
                'error': f'Exponential Smoothing error: {str(e)}',
                'model_name': 'Exponential Smoothing',
                'category': 'Time Series Specialized',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_hmm(self, data, **kwargs):
        """Train Hidden Markov Model"""
        try:
            # Prepare features for HMM
            features = data[['Close', 'Volume']].pct_change().dropna()
            
            # HMM parameters
            n_components = kwargs.get('n_components', 3)  # 3 market states
            
            # Fit HMM model
            model = GaussianHMM(n_components=n_components, random_state=42)
            model.fit(features.values)
            
            # Predict hidden states
            hidden_states = model.predict(features.values)
            current_state = hidden_states[-1]
            
            # Predict next price based on current state
            last_price = float(data['Close'].iloc[-1])
            state_returns = []
            
            for state in range(n_components):
                state_mask = hidden_states == state
                if np.sum(state_mask) > 0:
                    state_return = np.mean(features[state_mask]['Close'])
                    state_returns.append(state_return)
                else:
                    state_returns.append(0)
            
            predicted_return = state_returns[current_state]
            next_price = last_price * (1 + predicted_return)
            
            # Calculate RMSE
            actual_returns = features['Close'].values
            predicted_returns = np.zeros_like(actual_returns)
            for i, state in enumerate(hidden_states):
                predicted_returns[i] = state_returns[state]
            rmse = np.sqrt(np.mean((actual_returns - predicted_returns) ** 2) * last_price ** 2)
            confidence = max(0, min(100, 100 - (rmse / last_price * 100)))
            
            return {
                'model_name': 'Hidden Markov Models',
                'category': 'Time Series Specialized',
                'next_price': float(next_price),
                'confidence': float(confidence),
                'accuracy': float(confidence),
                'rmse': float(rmse),
                'current_state': int(current_state),
                'n_states': n_components
            }
            
        except Exception as e:
            return {
                'error': f'HMM error: {str(e)}',
                'model_name': 'Hidden Markov Models',
                'category': 'Time Series Specialized',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def get_global_performances(self):
        """Get all model performances"""
        return self.model_performances
