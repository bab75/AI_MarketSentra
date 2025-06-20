import pandas as pd
import numpy as np
from .classical_ml import ClassicalMLModels
from .ensemble_methods_safe import EnsembleModels
from .deep_learning_safe import DeepLearningModels
from .time_series_safe import TimeSeriesModels
from .clustering_safe import ClusteringAndDimensionalityReduction
from .anomaly_detection_safe import AnomalyDetectionModels
from .reinforcement_learning_safe import ReinforcementLearningModels
from sklearn.model_selection import train_test_split
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    """Central manager for all ML models"""
    
    def __init__(self):
        # Initialize all model categories
        self.classical_ml = ClassicalMLModels()
        self.ensemble_methods = EnsembleModels()
        self.deep_learning = DeepLearningModels()
        self.time_series = TimeSeriesModels()
        self.clustering = ClusteringAndDimensionalityReduction()
        self.anomaly_detection = AnomalyDetectionModels()
        self.reinforcement_learning = ReinforcementLearningModels()
        
        # Model category mapping
        self.model_categories = {
            'Classical Machine Learning': self.classical_ml,
            'Ensemble Methods': self.ensemble_methods,
            'Deep Learning': self.deep_learning,
            'Time Series': self.time_series,
            'Clustering & Dimensionality Reduction': self.clustering,
            'Anomaly Detection': self.anomaly_detection,
            'Reinforcement Learning': self.reinforcement_learning
        }
        
        # Global model performances
        self.global_performances = {}
        
    def get_available_models(self):
        """
        Get all available models organized by category
        
        Returns:
            dict: Dictionary of model categories and their models
        """
        available_models = {}
        
        for category, model_handler in self.model_categories.items():
            try:
                if category == 'Clustering & Dimensionality Reduction':
                    models = model_handler.get_available_models()
                    available_models[category] = models
                elif category == 'Anomaly Detection':
                    models = model_handler.get_available_models()
                    available_models[category] = models
                elif category == 'Reinforcement Learning':
                    models = model_handler.get_available_models()
                    available_models[category] = models
                else:
                    models = model_handler.get_available_models()
                    available_models[category] = models
            except Exception as e:
                st.warning(f"Error getting models for {category}: {str(e)}")
                available_models[category] = []
        
        return available_models
    
    def get_model_count(self):
        """
        Get total count of available models
        
        Returns:
            dict: Model counts by category and total
        """
        model_counts = {}
        total_count = 0
        
        available_models = self.get_available_models()
        
        for category, models in available_models.items():
            if isinstance(models, dict):
                # For nested categories (like clustering)
                category_count = sum(len(model_list) for model_list in models.values())
            elif isinstance(models, list):
                category_count = len(models)
            else:
                category_count = 0
            
            model_counts[category] = category_count
            total_count += category_count
        
        model_counts['Total'] = total_count
        return model_counts
    
    def train_and_predict(self, data, category, model_name, **kwargs):
        """
        Train a model and make predictions
        
        Args:
            data (DataFrame): Stock data
            category (str): Model category
            model_name (str): Name of the model
            **kwargs: Additional parameters
            
        Returns:
            dict: Training and prediction results
        """
        try:
            if category not in self.model_categories:
                raise ValueError(f"Category {category} not found")
            
            model_handler = self.model_categories[category]
            
            # Handle different model categories
            if category == 'Classical Machine Learning':
                return self._train_classical_ml(model_handler, data, model_name, **kwargs)
            
            elif category == 'Ensemble Methods':
                return self._train_ensemble(model_handler, data, model_name, **kwargs)
            
            elif category == 'Deep Learning':
                return self._train_deep_learning(model_handler, data, model_name, **kwargs)
            
            elif category == 'Time Series':
                return self._train_time_series(model_handler, data, model_name, **kwargs)
            
            elif category == 'Clustering & Dimensionality Reduction':
                return self._train_clustering(model_handler, data, model_name, **kwargs)
            
            elif category == 'Anomaly Detection':
                return self._train_anomaly_detection(model_handler, data, model_name, **kwargs)
            
            elif category == 'Reinforcement Learning':
                return self._train_reinforcement_learning(model_handler, data, model_name, **kwargs)
            
            else:
                raise ValueError(f"Unknown category: {category}")
                
        except Exception as e:
            st.error(f"Error training {model_name} in {category}: {str(e)}")
            return None
    
    def _train_classical_ml(self, model_handler, data, model_name, **kwargs):
        """Train classical ML models"""
        try:
            # Prepare features for ML
            from ..data_processor import DataProcessor
            processor = DataProcessor()
            
            features, target = processor.prepare_ml_features(data)
            if features is None or target is None:
                return None
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train model
            results = model_handler.train_model(model_name, X_train, y_train, X_test, y_test)
            
            if results:
                # Make prediction for next price
                latest_features = features.iloc[-1].values
                next_price = model_handler.predict_next_price(model_name, latest_features)
                
                results.update({
                    'next_price': next_price if next_price else target.iloc[-1],
                    'confidence': results.get('accuracy', 0),
                    'rmse': results.get('rmse', 0)
                })
                
                # Store global performance
                self.global_performances[f"{model_name} (Classical ML)"] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in classical ML training: {str(e)}")
            return None
    
    def _train_ensemble(self, model_handler, data, model_name, **kwargs):
        """Train ensemble models"""
        try:
            from ..data_processor import DataProcessor
            processor = DataProcessor()
            
            features, target = processor.prepare_ml_features(data)
            if features is None or target is None:
                return None
            
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Determine task type
            task_type = kwargs.get('task_type', 'regression')
            
            results = model_handler.train_model(model_name, X_train, y_train, X_test, y_test, task_type)
            
            if results:
                latest_features = features.iloc[-1].values
                next_price = model_handler.predict_next_price(model_name, latest_features)
                
                results.update({
                    'next_price': next_price if next_price else target.iloc[-1],
                    'confidence': results.get('accuracy', 0),
                    'rmse': results.get('rmse', 0)
                })
                
                self.global_performances[f"{model_name} (Ensemble)"] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in ensemble training: {str(e)}")
            return None
    
    def _train_deep_learning(self, model_handler, data, model_name, **kwargs):
        """Train deep learning models"""
        try:
            # Prepare data for deep learning (sequences)
            features_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            sequence_length = kwargs.get('sequence_length', 60)
            X, y = model_handler.prepare_sequences(features_data, sequence_length)
            
            if X is None or y is None:
                return None
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            epochs = kwargs.get('epochs', 50)
            batch_size = kwargs.get('batch_size', 32)
            
            results = model_handler.train_model(
                model_name, X_train, y_train, X_test, y_test,
                epochs=epochs, batch_size=batch_size
            )
            
            if results:
                # Predict next price
                latest_sequence = X[-1]
                next_price = model_handler.predict_next_price(model_name, latest_sequence)
                
                results.update({
                    'next_price': next_price if next_price else data['Close'].iloc[-1],
                    'confidence': results.get('accuracy', 0),
                    'rmse': results.get('rmse', 0)
                })
                
                self.global_performances[f"{model_name} (Deep Learning)"] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in deep learning training: {str(e)}")
            return None
    
    def _train_time_series(self, model_handler, data, model_name, **kwargs):
        """Train time series models"""
        try:
            results = model_handler.train_model(model_name, data, **kwargs)
            
            if results:
                # Get prediction
                predictions = results.get('predictions', [])
                next_price = predictions[0] if len(predictions) > 0 else data['Close'].iloc[-1]
                
                # Calculate confidence based on model type
                confidence = 0
                if 'aic' in results:
                    # Lower AIC is better, convert to confidence score
                    confidence = max(0, min(100, 100 - results['aic'] / 100))
                
                results.update({
                    'next_price': next_price,
                    'confidence': confidence,
                    'rmse': results.get('rmse', 0)
                })
                
                self.global_performances[f"{model_name} (Time Series)"] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in time series training: {str(e)}")
            return None
    
    def _train_clustering(self, model_handler, data, model_name, **kwargs):
        """Train clustering models"""
        try:
            # Determine if it's clustering or dimensionality reduction
            available_models = model_handler.get_available_models()
            
            if model_name in available_models.get('Clustering', []):
                results = model_handler.train_clustering_model(model_name, data, **kwargs)
                model_type = 'Clustering'
            elif model_name in available_models.get('Dimensionality Reduction', []):
                results = model_handler.train_dimensionality_reduction(model_name, data, **kwargs)
                model_type = 'Dimensionality Reduction'
            else:
                raise ValueError(f"Model {model_name} not found in clustering models")
            
            if results:
                # For clustering/dimensionality reduction, we don't predict prices
                # but we can provide insights
                results.update({
                    'next_price': data['Close'].iloc[-1],  # Current price
                    'confidence': 50,  # Neutral confidence
                    'rmse': 0,
                    'model_type': model_type
                })
                
                self.global_performances[f"{model_name} ({model_type})"] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in clustering/dimensionality reduction training: {str(e)}")
            return None
    
    def _train_anomaly_detection(self, model_handler, data, model_name, **kwargs):
        """Train anomaly detection models"""
        try:
            contamination = kwargs.get('contamination', 0.1)
            results = model_handler.train_model(model_name, data, contamination=contamination)
            
            if results:
                # For anomaly detection, provide current price and anomaly info
                results.update({
                    'next_price': data['Close'].iloc[-1],
                    'confidence': 100 - results.get('anomaly_rate', 0),  # Less anomalies = higher confidence
                    'rmse': 0,
                    'anomaly_info': {
                        'n_anomalies': results.get('n_anomalies', 0),
                        'anomaly_rate': results.get('anomaly_rate', 0)
                    }
                })
                
                self.global_performances[f"{model_name} (Anomaly Detection)"] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in anomaly detection training: {str(e)}")
            return None
    
    def _train_reinforcement_learning(self, model_handler, data, model_name, **kwargs):
        """Train reinforcement learning models"""
        try:
            total_timesteps = kwargs.get('total_timesteps', 10000)
            results = model_handler.train_model(model_name, data, total_timesteps=total_timesteps)
            
            if results:
                # For RL, we can't directly predict price, but we can provide portfolio performance
                evaluation = results.get('evaluation', {})
                mean_reward = evaluation.get('mean_reward', 0)
                
                results.update({
                    'next_price': data['Close'].iloc[-1],
                    'confidence': max(0, min(100, 50 + mean_reward * 10)),  # Convert reward to confidence
                    'rmse': 0,
                    'rl_info': {
                        'mean_reward': mean_reward,
                        'model_type': results.get('model_type', 'unknown')
                    }
                })
                
                self.global_performances[f"{model_name} (Reinforcement Learning)"] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error in reinforcement learning training: {str(e)}")
            return None
    
    def compare_models(self, data, model_list, test_size=0.2):
        """
        Compare performance of multiple models
        
        Args:
            data (DataFrame): Stock data
            model_list (list): List of tuples (category, model_name)
            test_size (float): Test set size
            
        Returns:
            dict: Comparison results
        """
        try:
            comparison_results = {}
            
            for category, model_name in model_list:
                try:
                    results = self.train_and_predict(data, category, model_name)
                    if results:
                        comparison_results[f"{model_name} ({category})"] = {
                            'accuracy': results.get('accuracy', 0),
                            'rmse': results.get('rmse', 0),
                            'confidence': results.get('confidence', 0),
                            'next_price': results.get('next_price', 0),
                            'category': category
                        }
                except Exception as e:
                    st.warning(f"Failed to train {model_name} ({category}): {str(e)}")
                    continue
            
            # Rank models by performance
            if comparison_results:
                # Sort by accuracy (higher is better)
                sorted_results = sorted(
                    comparison_results.items(),
                    key=lambda x: x[1].get('accuracy', 0),
                    reverse=True
                )
                
                comparison_results['ranking'] = [item[0] for item in sorted_results]
                comparison_results['best_model'] = sorted_results[0][0] if sorted_results else None
            
            return comparison_results
            
        except Exception as e:
            st.error(f"Error comparing models: {str(e)}")
            return {}
    
    def get_model_recommendations(self, data, task_type='prediction'):
        """
        Get model recommendations based on data characteristics
        
        Args:
            data (DataFrame): Stock data
            task_type (str): Type of task ('prediction', 'anomaly', 'clustering', 'trading')
            
        Returns:
            dict: Model recommendations
        """
        try:
            data_size = len(data)
            has_volume = 'Volume' in data.columns
            time_span = (data.index.max() - data.index.min()).days
            
            recommendations = {
                'recommended_models': [],
                'data_characteristics': {
                    'size': data_size,
                    'has_volume': has_volume,
                    'time_span_days': time_span,
                    'frequency': 'daily'  # Assuming daily data
                },
                'reasoning': []
            }
            
            if task_type == 'prediction':
                if data_size > 1000:
                    recommendations['recommended_models'].extend([
                        ('Deep Learning', 'LSTM'),
                        ('Deep Learning', 'GRU'),
                        ('Ensemble Methods', 'XGBoost'),
                        ('Ensemble Methods', 'Random Forest')
                    ])
                    recommendations['reasoning'].append("Large dataset suitable for deep learning and ensemble methods")
                else:
                    recommendations['recommended_models'].extend([
                        ('Classical Machine Learning', 'Linear Regression'),
                        ('Classical Machine Learning', 'Decision Tree'),
                        ('Time Series', 'ARIMA')
                    ])
                    recommendations['reasoning'].append("Smaller dataset better suited for classical methods")
                
                if time_span > 365:
                    recommendations['recommended_models'].extend([
                        ('Time Series', 'SARIMA'),
                        ('Time Series', 'Auto ARIMA')
                    ])
                    recommendations['reasoning'].append("Long time series suitable for seasonal models")
            
            elif task_type == 'anomaly':
                recommendations['recommended_models'].extend([
                    ('Anomaly Detection', 'Isolation Forest'),
                    ('Anomaly Detection', 'One-Class SVM'),
                    ('Anomaly Detection', 'Local Outlier Factor')
                ])
                recommendations['reasoning'].append("Standard anomaly detection models for financial data")
            
            elif task_type == 'clustering':
                recommendations['recommended_models'].extend([
                    ('Clustering & Dimensionality Reduction', 'K-Means'),
                    ('Clustering & Dimensionality Reduction', 'DBSCAN'),
                    ('Clustering & Dimensionality Reduction', 'PCA')
                ])
                recommendations['reasoning'].append("Clustering models for pattern discovery")
            
            elif task_type == 'trading':
                if data_size > 500:
                    recommendations['recommended_models'].extend([
                        ('Reinforcement Learning', 'PPO (Proximal Policy Optimization)'),
                        ('Reinforcement Learning', 'DQN (Deep Q-Network)')
                    ])
                    recommendations['reasoning'].append("Sufficient data for reinforcement learning trading")
                else:
                    recommendations['recommended_models'].extend([
                        ('Reinforcement Learning', 'Q-Learning')
                    ])
                    recommendations['reasoning'].append("Classical RL for smaller datasets")
            
            return recommendations
            
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            return {'recommended_models': [], 'reasoning': []}
    
    def get_global_performance_summary(self):
        """
        Get summary of all trained models' performance
        
        Returns:
            dict: Performance summary
        """
        if not self.global_performances:
            return {'message': 'No models trained yet'}
        
        summary = {
            'total_models': len(self.global_performances),
            'best_accuracy': 0,
            'best_model': None,
            'average_accuracy': 0,
            'category_performance': {}
        }
        
        accuracies = []
        for model_name, results in self.global_performances.items():
            accuracy = results.get('accuracy', 0)
            accuracies.append(accuracy)
            
            if accuracy > summary['best_accuracy']:
                summary['best_accuracy'] = accuracy
                summary['best_model'] = model_name
            
            # Extract category from model name
            category = model_name.split('(')[-1].rstrip(')')
            if category not in summary['category_performance']:
                summary['category_performance'][category] = []
            summary['category_performance'][category].append(accuracy)
        
        summary['average_accuracy'] = np.mean(accuracies) if accuracies else 0
        
        # Calculate average by category
        for category in summary['category_performance']:
            category_accuracies = summary['category_performance'][category]
            summary['category_performance'][category] = {
                'average_accuracy': np.mean(category_accuracies),
                'model_count': len(category_accuracies)
            }
        
        return summary
    
    def export_model_results(self, format='json'):
        """
        Export model results in specified format
        
        Args:
            format (str): Export format ('json', 'csv')
            
        Returns:
            str or DataFrame: Exported results
        """
        try:
            if not self.global_performances:
                return None
            
            if format == 'json':
                import json
                return json.dumps(self.global_performances, indent=2, default=str)
            
            elif format == 'csv':
                # Convert to DataFrame
                results_data = []
                for model_name, results in self.global_performances.items():
                    results_data.append({
                        'Model': model_name,
                        'Accuracy': results.get('accuracy', 0),
                        'RMSE': results.get('rmse', 0),
                        'Confidence': results.get('confidence', 0),
                        'Next_Price': results.get('next_price', 0)
                    })
                
                return pd.DataFrame(results_data)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            st.error(f"Error exporting results: {str(e)}")
            return None
