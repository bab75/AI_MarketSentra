import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA as PCA_Anomaly
from pyod.models.mcd import MCD
from pyod.models.lscp import LSCP
from pyod.models.auto_encoder import AutoEncoder
import streamlit as st

class AnomalyDetectionModels:
    """Advanced Anomaly Detection Models for Financial Data"""
    
    def __init__(self):
        self.models = {
            'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
            'One-Class SVM': OneClassSVM(nu=0.1),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Elliptic Envelope': EllipticEnvelope(contamination=0.1, random_state=42),
            'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=0.1),
            'Statistical Z-Score': None,  # Custom implementation
            'Modified Z-Score': None,  # Custom implementation
            'Interquartile Range': None,  # Custom implementation
        }
        
        # PyOD models for advanced anomaly detection
        self.pyod_models = {
            'KNN Anomaly Detection': KNN(contamination=0.1),
            'LOF (Local Outlier Factor)': LOF(contamination=0.1),
            'CBLOF (Cluster-based LOF)': CBLOF(contamination=0.1, random_state=42),
            'HBOS (Histogram-based Outlier Score)': HBOS(contamination=0.1),
            'PyOD Isolation Forest': IForest(contamination=0.1, random_state=42),
            'PyOD One-Class SVM': OCSVM(contamination=0.1),
            'PCA Anomaly Detection': PCA_Anomaly(contamination=0.1, random_state=42),
            'MCD (Minimum Covariance Determinant)': MCD(contamination=0.1, random_state=42),
            'LSCP (Locally Selective Combination)': LSCP(contamination=0.1, random_state=42),
            'AutoEncoder Anomaly Detection': AutoEncoder(contamination=0.1, random_state=42)
        }
        
        self.trained_models = {}
        self.scalers = {}
        self.model_performances = {}
    
    def get_available_models(self):
        """Return list of available anomaly detection models"""
        sklearn_models = list(self.models.keys())
        pyod_models = list(self.pyod_models.keys())
        return {
            'Classical Models': sklearn_models,
            'Advanced PyOD Models': pyod_models
        }
    
    def prepare_features(self, data):
        """
        Prepare features for anomaly detection
        
        Args:
            data (DataFrame): Stock data
            
        Returns:
            tuple: (scaled_features, feature_names, scaler)
        """
        try:
            # Select relevant features for anomaly detection
            feature_columns = []
            
            # Price-based features
            if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
                feature_columns.extend(['Open', 'High', 'Low', 'Close'])
                
                # Price movements and ranges
                data['Price_Range'] = data['High'] - data['Low']
                data['Price_Change'] = data['Close'] - data['Open']
                data['Price_Change_Pct'] = (data['Close'] - data['Open']) / data['Open'] * 100
                feature_columns.extend(['Price_Range', 'Price_Change', 'Price_Change_Pct'])
            
            # Volume-based features
            if 'Volume' in data.columns:
                feature_columns.append('Volume')
                
                # Volume ratios and changes
                data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
                feature_columns.extend(['Volume_Ratio'])
            
            # Technical indicators
            technical_indicators = [
                'Daily_Return', 'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Upper', 'BB_Middle', 'BB_Lower'
            ]
            
            for indicator in technical_indicators:
                if indicator in data.columns:
                    feature_columns.append(indicator)
            
            # Time-based features
            data['Hour'] = data.index.hour if hasattr(data.index, 'hour') else 0
            data['Day_of_Week'] = data.index.dayofweek
            data['Month'] = data.index.month
            feature_columns.extend(['Day_of_Week', 'Month'])
            
            # Extract features and handle missing values
            features_df = data[feature_columns].copy()
            
            # Fill missing values with forward fill then backward fill
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            # Remove any remaining NaN values
            features_df = features_df.dropna()
            
            if len(features_df) == 0:
                raise ValueError("No valid features found after preprocessing")
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_df)
            
            return scaled_features, feature_columns, scaler, features_df.index
            
        except Exception as e:
            st.error(f"Error preparing features for anomaly detection: {str(e)}")
            return None, None, None, None
    
    def statistical_zscore_detection(self, data, threshold=3.0):
        """
        Statistical anomaly detection using Z-Score
        
        Args:
            data (array): Feature data
            threshold (float): Z-score threshold
            
        Returns:
            array: Anomaly labels (-1 for anomaly, 1 for normal)
        """
        try:
            z_scores = np.abs(stats.zscore(data, axis=0))
            anomaly_mask = np.any(z_scores > threshold, axis=1)
            labels = np.where(anomaly_mask, -1, 1)
            return labels
            
        except Exception as e:
            st.error(f"Error in Z-score anomaly detection: {str(e)}")
            return None
    
    def modified_zscore_detection(self, data, threshold=3.5):
        """
        Modified Z-Score anomaly detection using median
        
        Args:
            data (array): Feature data
            threshold (float): Modified Z-score threshold
            
        Returns:
            array: Anomaly labels (-1 for anomaly, 1 for normal)
        """
        try:
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            
            # Avoid division by zero
            mad = np.where(mad == 0, 1, mad)
            
            modified_z_scores = 0.6745 * (data - median) / mad
            anomaly_mask = np.any(np.abs(modified_z_scores) > threshold, axis=1)
            labels = np.where(anomaly_mask, -1, 1)
            return labels
            
        except Exception as e:
            st.error(f"Error in Modified Z-score anomaly detection: {str(e)}")
            return None
    
    def iqr_detection(self, data, factor=1.5):
        """
        Interquartile Range (IQR) anomaly detection
        
        Args:
            data (array): Feature data
            factor (float): IQR factor for outlier detection
            
        Returns:
            array: Anomaly labels (-1 for anomaly, 1 for normal)
        """
        try:
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            
            anomaly_mask = np.any((data < lower_bound) | (data > upper_bound), axis=1)
            labels = np.where(anomaly_mask, -1, 1)
            return labels
            
        except Exception as e:
            st.error(f"Error in IQR anomaly detection: {str(e)}")
            return None
    
    def train_model(self, model_name, data, contamination=0.1, **kwargs):
        """
        Train an anomaly detection model
        
        Args:
            model_name (str): Name of the model to train
            data (DataFrame): Stock data
            contamination (float): Expected proportion of outliers
            **kwargs: Additional model parameters
            
        Returns:
            dict: Training results and anomaly detection results
        """
        try:
            # Prepare features
            scaled_features, feature_names, scaler, feature_index = self.prepare_features(data)
            
            if scaled_features is None:
                return None
            
            # Store scaler for later use
            self.scalers[model_name] = scaler
            
            # Select and configure model
            if model_name in self.models:
                if model_name == 'Statistical Z-Score':
                    labels = self.statistical_zscore_detection(scaled_features, 
                                                             kwargs.get('threshold', 3.0))
                    model = None
                elif model_name == 'Modified Z-Score':
                    labels = self.modified_zscore_detection(scaled_features, 
                                                          kwargs.get('threshold', 3.5))
                    model = None
                elif model_name == 'Interquartile Range':
                    labels = self.iqr_detection(scaled_features, 
                                              kwargs.get('factor', 1.5))
                    model = None
                else:
                    model = self.models[model_name]
                    
                    # Update contamination parameter if supported
                    if hasattr(model, 'contamination'):
                        model.set_params(contamination=contamination)
                    elif hasattr(model, 'nu'):
                        model.set_params(nu=contamination)
                    
                    # Fit model and predict
                    if model_name == 'Local Outlier Factor':
                        labels = model.fit_predict(scaled_features)
                    else:
                        model.fit(scaled_features)
                        labels = model.predict(scaled_features)
            
            elif model_name in self.pyod_models:
                model = self.pyod_models[model_name]
                
                # Update contamination parameter
                model.contamination = contamination
                
                # Fit model and predict
                model.fit(scaled_features)
                labels = model.predict(scaled_features)
                
                # Convert PyOD labels (0=normal, 1=anomaly) to sklearn format (1=normal, -1=anomaly)
                labels = np.where(labels == 1, -1, 1)
            
            else:
                raise ValueError(f"Model {model_name} not found")
            
            # Calculate anomaly scores if available
            anomaly_scores = None
            if model is not None:
                if hasattr(model, 'decision_function'):
                    anomaly_scores = model.decision_function(scaled_features)
                elif hasattr(model, 'score_samples'):
                    anomaly_scores = model.score_samples(scaled_features)
                elif hasattr(model, 'decision_scores_'):
                    anomaly_scores = model.decision_scores_
            
            # Create results DataFrame
            results_df = pd.DataFrame(index=feature_index)
            results_df['Anomaly'] = labels == -1
            results_df['Anomaly_Score'] = anomaly_scores if anomaly_scores is not None else np.nan
            
            # Add original data
            results_df = pd.concat([results_df, data.loc[feature_index]], axis=1)
            
            # Calculate performance metrics
            n_anomalies = np.sum(labels == -1)
            anomaly_rate = n_anomalies / len(labels) * 100
            
            # Anomaly statistics
            if n_anomalies > 0:
                anomaly_data = results_df[results_df['Anomaly'] == True]
                anomaly_stats = {
                    'mean_price_change': anomaly_data.get('Price_Change_Pct', pd.Series()).mean(),
                    'mean_volume_ratio': anomaly_data.get('Volume_Ratio', pd.Series()).mean(),
                    'date_range': {
                        'start': anomaly_data.index.min(),
                        'end': anomaly_data.index.max()
                    }
                }
            else:
                anomaly_stats = {}
            
            results = {
                'model': model,
                'labels': labels,
                'anomaly_scores': anomaly_scores,
                'results_df': results_df,
                'n_anomalies': n_anomalies,
                'anomaly_rate': anomaly_rate,
                'feature_names': feature_names,
                'contamination': contamination,
                'anomaly_stats': anomaly_stats
            }
            
            # Store trained model
            self.trained_models[model_name] = results
            self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None
    
    def detect_anomalies_ensemble(self, data, models_list=None, voting='majority'):
        """
        Ensemble anomaly detection using multiple models
        
        Args:
            data (DataFrame): Stock data
            models_list (list): List of model names to use
            voting (str): 'majority' or 'unanimous'
            
        Returns:
            dict: Ensemble anomaly detection results
        """
        try:
            if models_list is None:
                models_list = ['Isolation Forest', 'One-Class SVM', 'Local Outlier Factor']
            
            # Prepare features
            scaled_features, feature_names, scaler, feature_index = self.prepare_features(data)
            
            if scaled_features is None:
                return None
            
            # Train individual models and collect predictions
            predictions = []
            model_results = {}
            
            for model_name in models_list:
                try:
                    result = self.train_model(model_name, data)
                    if result is not None:
                        predictions.append(result['labels'])
                        model_results[model_name] = result
                except Exception as e:
                    st.warning(f"Failed to train {model_name}: {str(e)}")
                    continue
            
            if len(predictions) == 0:
                raise ValueError("No models trained successfully")
            
            # Ensemble voting
            predictions_array = np.array(predictions)
            
            if voting == 'majority':
                # Majority vote: anomaly if majority of models predict anomaly
                anomaly_votes = np.sum(predictions_array == -1, axis=0)
                ensemble_labels = np.where(anomaly_votes > len(predictions) / 2, -1, 1)
            else:  # unanimous
                # Unanimous vote: anomaly only if all models agree
                ensemble_labels = np.where(np.all(predictions_array == -1, axis=0), -1, 1)
            
            # Create ensemble results
            results_df = pd.DataFrame(index=feature_index)
            results_df['Ensemble_Anomaly'] = ensemble_labels == -1
            
            # Add individual model predictions
            for i, model_name in enumerate(models_list):
                if i < len(predictions):
                    results_df[f'{model_name}_Anomaly'] = predictions[i] == -1
            
            # Add original data
            results_df = pd.concat([results_df, data.loc[feature_index]], axis=1)
            
            # Calculate ensemble metrics
            n_anomalies = np.sum(ensemble_labels == -1)
            anomaly_rate = n_anomalies / len(ensemble_labels) * 100
            
            # Model agreement analysis
            agreement_scores = []
            for i in range(len(ensemble_labels)):
                model_predictions = predictions_array[:, i]
                agreement = np.sum(model_predictions == ensemble_labels[i]) / len(predictions)
                agreement_scores.append(agreement)
            
            results = {
                'ensemble_labels': ensemble_labels,
                'results_df': results_df,
                'individual_results': model_results,
                'n_anomalies': n_anomalies,
                'anomaly_rate': anomaly_rate,
                'agreement_scores': agreement_scores,
                'mean_agreement': np.mean(agreement_scores),
                'voting_method': voting,
                'models_used': models_list
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error in ensemble anomaly detection: {str(e)}")
            return None
    
    def real_time_anomaly_detection(self, data, model_name, window_size=100):
        """
        Real-time anomaly detection using sliding window
        
        Args:
            data (DataFrame): Stock data
            model_name (str): Name of the model to use
            window_size (int): Size of the sliding window
            
        Returns:
            dict: Real-time anomaly detection results
        """
        try:
            if len(data) < window_size:
                raise ValueError(f"Insufficient data. Need at least {window_size} data points")
            
            # Initialize results
            anomaly_flags = []
            anomaly_scores_list = []
            
            # Sliding window approach
            for i in range(window_size, len(data)):
                window_data = data.iloc[i-window_size:i]
                
                # Train model on window data
                result = self.train_model(model_name, window_data)
                
                if result is not None:
                    # Get the anomaly flag for the latest data point
                    latest_anomaly = result['labels'][-1] == -1
                    anomaly_flags.append(latest_anomaly)
                    
                    # Get anomaly score if available
                    if result['anomaly_scores'] is not None:
                        anomaly_scores_list.append(result['anomaly_scores'][-1])
                    else:
                        anomaly_scores_list.append(np.nan)
                else:
                    anomaly_flags.append(False)
                    anomaly_scores_list.append(np.nan)
            
            # Create results DataFrame
            results_index = data.index[window_size:]
            results_df = pd.DataFrame(index=results_index)
            results_df['Real_Time_Anomaly'] = anomaly_flags
            results_df['Real_Time_Score'] = anomaly_scores_list
            
            # Add original data
            results_df = pd.concat([results_df, data.loc[results_index]], axis=1)
            
            results = {
                'results_df': results_df,
                'n_anomalies': np.sum(anomaly_flags),
                'anomaly_rate': np.mean(anomaly_flags) * 100,
                'window_size': window_size,
                'model_name': model_name
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error in real-time anomaly detection: {str(e)}")
            return None
    
    def get_model_config(self, model_name):
        """
        Get configuration parameters for anomaly detection models
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration parameters
        """
        configs = {
            'Isolation Forest': {
                'contamination': 0.1,
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42
            },
            'One-Class SVM': {
                'nu': 0.1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'DBSCAN': {
                'eps': 0.5,
                'min_samples': 5,
                'metric': 'euclidean'
            },
            'Elliptic Envelope': {
                'contamination': 0.1,
                'support_fraction': None,
                'random_state': 42
            },
            'Local Outlier Factor': {
                'n_neighbors': 20,
                'contamination': 0.1,
                'algorithm': 'auto'
            },
            'Statistical Z-Score': {
                'threshold': 3.0
            },
            'Modified Z-Score': {
                'threshold': 3.5
            },
            'Interquartile Range': {
                'factor': 1.5
            },
            'KNN Anomaly Detection': {
                'contamination': 0.1,
                'n_neighbors': 5,
                'method': 'largest'
            },
            'HBOS (Histogram-based Outlier Score)': {
                'contamination': 0.1,
                'n_bins': 10,
                'alpha': 0.1
            },
            'AutoEncoder Anomaly Detection': {
                'contamination': 0.1,
                'epochs': 100,
                'batch_size': 32,
                'hidden_neurons': [64, 32, 32, 64]
            }
        }
        
        return configs.get(model_name, {'contamination': 0.1})
    
    def analyze_anomaly_patterns(self, anomaly_results, data):
        """
        Analyze patterns in detected anomalies
        
        Args:
            anomaly_results (dict): Results from anomaly detection
            data (DataFrame): Original data
            
        Returns:
            dict: Anomaly pattern analysis
        """
        try:
            results_df = anomaly_results['results_df']
            anomalies = results_df[results_df['Anomaly'] == True]
            
            if len(anomalies) == 0:
                return {'message': 'No anomalies detected'}
            
            # Time-based patterns
            time_patterns = {
                'by_hour': anomalies.groupby(anomalies.index.hour).size().to_dict(),
                'by_day_of_week': anomalies.groupby(anomalies.index.dayofweek).size().to_dict(),
                'by_month': anomalies.groupby(anomalies.index.month).size().to_dict(),
                'by_year': anomalies.groupby(anomalies.index.year).size().to_dict()
            }
            
            # Price-based patterns
            price_patterns = {}
            if 'Close' in anomalies.columns:
                price_patterns['price_range'] = {
                    'min': anomalies['Close'].min(),
                    'max': anomalies['Close'].max(),
                    'mean': anomalies['Close'].mean(),
                    'std': anomalies['Close'].std()
                }
            
            # Volume patterns
            volume_patterns = {}
            if 'Volume' in anomalies.columns:
                volume_patterns['volume_stats'] = {
                    'min': anomalies['Volume'].min(),
                    'max': anomalies['Volume'].max(),
                    'mean': anomalies['Volume'].mean(),
                    'std': anomalies['Volume'].std()
                }
            
            # Consecutive anomalies
            anomaly_indices = anomalies.index
            consecutive_groups = []
            current_group = [anomaly_indices[0]]
            
            for i in range(1, len(anomaly_indices)):
                if (anomaly_indices[i] - anomaly_indices[i-1]).days <= 1:
                    current_group.append(anomaly_indices[i])
                else:
                    if len(current_group) > 1:
                        consecutive_groups.append(current_group)
                    current_group = [anomaly_indices[i]]
            
            if len(current_group) > 1:
                consecutive_groups.append(current_group)
            
            analysis = {
                'total_anomalies': len(anomalies),
                'anomaly_rate': len(anomalies) / len(results_df) * 100,
                'time_patterns': time_patterns,
                'price_patterns': price_patterns,
                'volume_patterns': volume_patterns,
                'consecutive_anomaly_groups': len(consecutive_groups),
                'longest_consecutive_streak': max([len(group) for group in consecutive_groups]) if consecutive_groups else 0,
                'anomaly_severity': {
                    'high': len(anomalies[anomalies.get('Anomaly_Score', 0) > 0.5]) if 'Anomaly_Score' in anomalies.columns else 0,
                    'medium': len(anomalies[(anomalies.get('Anomaly_Score', 0) > 0.2) & (anomalies.get('Anomaly_Score', 0) <= 0.5)]) if 'Anomaly_Score' in anomalies.columns else 0,
                    'low': len(anomalies[anomalies.get('Anomaly_Score', 0) <= 0.2]) if 'Anomaly_Score' in anomalies.columns else 0
                }
            }
            
            return analysis
            
        except Exception as e:
            st.error(f"Error analyzing anomaly patterns: {str(e)}")
            return {}
