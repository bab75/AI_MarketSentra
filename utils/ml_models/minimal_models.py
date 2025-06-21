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
from .deep_learning_models import DeepLearningManager

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
            }
        }
        
        self.trained_models = {}
        self.model_performances = {}
        self.scalers = {}
        self.deep_learning_manager = DeepLearningManager()
    
    def get_available_models(self):
        """Get all available models organized by category"""
        available = {}
        for category, models in self.models.items():
            available[category] = list(models.keys())
        
        # Add deep learning models
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
            if category == 'Deep Learning Models':
                return self.deep_learning_manager.train_and_predict(data, model_name, **kwargs)
            
            if category not in self.models:
                return {'error': f'Category {category} not found'}
            
            if model_name not in self.models[category]:
                return {'error': f'Model {model_name} not found in {category}'}
            
            # Prepare data based on model category
            if category in ['Classical ML', 'Ensemble Methods']:
                return self._train_supervised_model(data, category, model_name, **kwargs)
            elif category == 'Clustering':
                return self._train_clustering_model(data, model_name, **kwargs)
            elif category == 'Anomaly Detection':
                return self._train_anomaly_model(data, model_name, **kwargs)
            elif category == 'Dimensionality Reduction':
                return self._train_dimensionality_model(data, model_name, **kwargs)
            else:
                return {'error': f'Category {category} not implemented'}
                
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_features(self, data, lookback=5):
        """Prepare features for ML models"""
        try:
            # Use basic OHLCV features
            features_df = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Add simple technical indicators
            features_df = features_df.copy()
            features_df['Price_Change'] = features_df['Close'].pct_change()
            features_df['Volume_Change'] = features_df['Volume'].pct_change()
            features_df['High_Low_Ratio'] = features_df['High'] / features_df['Low']
            features_df['Close_Open_Ratio'] = features_df['Close'] / features_df['Open']
            
            # Add moving averages
            features_df['MA_5'] = features_df['Close'].rolling(5).mean()
            features_df['MA_10'] = features_df['Close'].rolling(10).mean()
            
            # Drop NaN values
            features_df = features_df.dropna()
            
            if len(features_df) < lookback + 1:
                return None, None
            
            # Create feature matrix with lookback
            X = []
            y = []
            
            for i in range(lookback, len(features_df)):
                # Use last 'lookback' rows as features
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
            # Prepare features
            X, y = self._prepare_features(data)
            if X is None or y is None:
                return {'error': 'Could not prepare features'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Get model
            model = self.models[category][model_name]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            train_r2 = r2_score(y_train, train_pred) * 100
            test_r2 = r2_score(y_test, test_pred) * 100
            
            # Predict next price
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
                'next_price': next_price,
                'confidence': max(0, min(100, test_r2))
            }
            
            # Store model
            self.trained_models[f"{category}_{model_name}"] = model
            self.model_performances[f"{category}_{model_name}"] = results
            
            return results
            
        except Exception as e:
            return {'error': f'Training error: {str(e)}'}
    
    def _train_clustering_model(self, data, model_name, **kwargs):
        """Train clustering models"""
        try:
            # Prepare features
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Get model
            model = self.models['Clustering'][model_name]
            
            # Fit model
            labels = model.fit_predict(scaled_features)
            
            results = {
                'model_name': model_name,
                'category': 'Clustering',
                'n_clusters': len(set(labels)),
                'labels': labels.tolist(),
                'next_price': data['Close'].iloc[-1],
                'confidence': 50,
                'rmse': 0,
                'accuracy': 50.0
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Clustering error: {str(e)}'}
    
    def _train_anomaly_model(self, data, model_name, **kwargs):
        """Train anomaly detection models"""
        try:
            # Prepare features
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Get model
            model = self.models['Anomaly Detection'][model_name]
            
            # Fit model and predict
            predictions = model.fit_predict(scaled_features)
            anomalies = predictions == -1
            
            results = {
                'model_name': model_name,
                'category': 'Anomaly Detection',
                'n_anomalies': int(np.sum(anomalies)),
                'anomaly_rate': float(np.mean(anomalies) * 100),
                'predictions': predictions.tolist(),
                'next_price': data['Close'].iloc[-1],
                'confidence': 100 - (np.mean(anomalies) * 100),
                'rmse': 0,
                'accuracy': 100 - (np.mean(anomalies) * 100)
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Anomaly detection error: {str(e)}'}
    
    def _train_dimensionality_model(self, data, model_name, **kwargs):
        """Train dimensionality reduction models"""
        try:
            # Prepare features
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Get model
            model = self.models['Dimensionality Reduction'][model_name]
            
            # Fit and transform
            transformed = model.fit_transform(scaled_features)
            
            results = {
                'model_name': model_name,
                'category': 'Dimensionality Reduction',
                'n_components': transformed.shape[1],
                'explained_variance': model.explained_variance_ratio_.tolist(),
                'total_variance': float(np.sum(model.explained_variance_ratio_)),
                'next_price': data['Close'].iloc[-1],
                'confidence': 50,
                'rmse': 0,
                'accuracy': float(np.sum(model.explained_variance_ratio_)) * 100
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Dimensionality reduction error: {str(e)}'}
    
    def get_global_performances(self):
        """Get all model performances"""
        return self.model_performances
