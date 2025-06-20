import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import streamlit as st

class AnomalyDetectionModels:
    """Anomaly Detection Models for Stock Data (Safe Version)"""
    
    def __init__(self):
        self.models = {
            'Isolation Forest': IsolationForest,
            'One-Class SVM': OneClassSVM,
            'Statistical Outliers': None  # Custom implementation
        }
        
        self.trained_models = {}
        self.model_performances = {}
        self.scalers = {}
    
    def get_available_models(self):
        """Return list of available anomaly detection models"""
        return list(self.models.keys())
    
    def train_model(self, model_name, data, contamination=0.1, **kwargs):
        """
        Train an anomaly detection model
        
        Args:
            model_name (str): Name of the model
            data (DataFrame): Stock data
            contamination (float): Expected proportion of outliers
            
        Returns:
            dict: Training results
        """
        try:
            # Prepare features
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            if model_name == 'Isolation Forest':
                model = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_estimators=100
                )
            elif model_name == 'One-Class SVM':
                model = OneClassSVM(
                    gamma='scale',
                    nu=contamination
                )
            elif model_name == 'Statistical Outliers':
                return self._train_statistical_outliers(data, **kwargs)
            else:
                raise ValueError(f"Unknown anomaly detection model: {model_name}")
            
            # Fit model and predict anomalies
            predictions = model.fit_predict(scaled_features)
            
            # Convert predictions (-1 for anomalies, 1 for normal)
            anomalies = predictions == -1
            
            # Calculate metrics
            n_anomalies = np.sum(anomalies)
            anomaly_rate = (n_anomalies / len(features)) * 100
            
            results = {
                'model_name': model_name,
                'n_anomalies': int(n_anomalies),
                'anomaly_rate': float(anomaly_rate),
                'anomaly_indices': np.where(anomalies)[0].tolist(),
                'predictions': predictions.tolist(),
                'contamination': contamination
            }
            
            self.trained_models[model_name] = model
            self.scalers[model_name] = scaler
            self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training anomaly detection model {model_name}: {str(e)}")
            return None
    
    def _train_statistical_outliers(self, data, z_threshold=3, **kwargs):
        """Train statistical outlier detection using z-score"""
        try:
            # Use Close prices for statistical analysis
            prices = data['Close'].dropna()
            
            # Calculate z-scores
            mean_price = prices.mean()
            std_price = prices.std()
            z_scores = np.abs((prices - mean_price) / std_price)
            
            # Identify outliers
            anomalies = z_scores > z_threshold
            
            # Calculate metrics
            n_anomalies = np.sum(anomalies)
            anomaly_rate = (n_anomalies / len(prices)) * 100
            
            results = {
                'model_name': 'Statistical Outliers',
                'n_anomalies': int(n_anomalies),
                'anomaly_rate': float(anomaly_rate),
                'anomaly_indices': np.where(anomalies)[0].tolist(),
                'z_threshold': z_threshold,
                'mean_price': float(mean_price),
                'std_price': float(std_price),
                'max_z_score': float(z_scores.max())
            }
            
            self.model_performances['Statistical Outliers'] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in statistical outlier detection: {str(e)}")
            return None
    
    def detect_anomalies(self, model_name, new_data):
        """
        Detect anomalies in new data using trained model
        
        Args:
            model_name (str): Name of trained model
            new_data (DataFrame): New data to analyze
            
        Returns:
            array: Anomaly predictions
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained yet")
            
            model = self.trained_models[model_name]
            scaler = self.scalers[model_name]
            
            # Prepare features
            features = new_data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            scaled_features = scaler.transform(features)
            
            # Predict anomalies
            predictions = model.predict(scaled_features)
            
            return predictions
            
        except Exception as e:
            st.error(f"Error detecting anomalies with {model_name}: {str(e)}")
            return None
    
    def get_model_performance(self, model_name):
        """Get performance metrics for a specific model"""
        return self.model_performances.get(model_name, {})
    
    def get_all_performances(self):
        """Get all model performances"""
        return self.model_performances
    
    def get_anomaly_score(self, model_name, data):
        """
        Get anomaly scores for data points
        
        Args:
            model_name (str): Name of trained model
            data (DataFrame): Data to score
            
        Returns:
            array: Anomaly scores
        """
        try:
            if model_name not in self.trained_models:
                return None
            
            model = self.trained_models[model_name]
            scaler = self.scalers[model_name]
            
            # Prepare features
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            scaled_features = scaler.transform(features)
            
            # Get anomaly scores if available
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(scaled_features)
                return scores
            else:
                return None
                
        except Exception as e:
            st.error(f"Error getting anomaly scores for {model_name}: {str(e)}")
            return None