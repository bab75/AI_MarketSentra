import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    BaggingRegressor, BaggingClassifier,
    VotingRegressor, VotingClassifier
)
import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import streamlit as st

class EnsembleModels:
    """Ensemble Machine Learning Models for Stock Prediction"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=42),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'Bagging': BaggingRegressor(n_estimators=10, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
        }
        
        self.classification_models = {
            'Random Forest (Classification)': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting (Classification)': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'AdaBoost (Classification)': AdaBoostClassifier(n_estimators=50, random_state=42),
            'Extra Trees (Classification)': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'XGBoost (Classification)': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        self.trained_models = {}
        self.model_performances = {}
    
    def get_available_models(self):
        """Return list of available ensemble models"""
        return list(self.models.keys()) + list(self.classification_models.keys())
    
    def train_model(self, model_name, X_train, y_train, X_test=None, y_test=None, task_type='regression'):
        """
        Train a specific ensemble model
        
        Args:
            model_name (str): Name of the model to train
            X_train (array): Training features
            y_train (array): Training target
            X_test (array): Test features
            y_test (array): Test target
            task_type (str): 'regression' or 'classification'
            
        Returns:
            dict: Training results and metrics
        """
        try:
            # Select appropriate model based on task type
            if task_type == 'classification' or '(Classification)' in model_name:
                if model_name in self.classification_models:
                    model = self.classification_models[model_name]
                else:
                    # Convert regression model name to classification
                    class_model_name = f"{model_name} (Classification)"
                    if class_model_name in self.classification_models:
                        model = self.classification_models[class_model_name]
                    else:
                        raise ValueError(f"Classification model {model_name} not found")
            else:
                if model_name in self.models:
                    model = self.models[model_name]
                else:
                    raise ValueError(f"Regression model {model_name} not found")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test) if X_test is not None else None
            
            # Calculate metrics
            results = {
                'model_name': model_name,
                'task_type': task_type,
                'train_size': len(X_train),
                'test_size': len(X_test) if X_test is not None else 0
            }
            
            if task_type == 'classification' or '(Classification)' in model_name:
                # Classification metrics
                train_accuracy = accuracy_score(y_train, train_pred) * 100
                results['train_accuracy'] = train_accuracy
                
                if X_test is not None and y_test is not None:
                    test_accuracy = accuracy_score(y_test, test_pred) * 100
                    results['test_accuracy'] = test_accuracy
                    results['accuracy'] = test_accuracy
                else:
                    results['accuracy'] = train_accuracy
            else:
                # Regression metrics
                train_mse = mean_squared_error(y_train, train_pred)
                train_rmse = np.sqrt(train_mse)
                train_r2 = r2_score(y_train, train_pred) * 100
                
                results.update({
                    'train_mse': train_mse,
                    'train_rmse': train_rmse,
                    'train_r2': train_r2
                })
                
                if X_test is not None and y_test is not None:
                    test_mse = mean_squared_error(y_test, test_pred)
                    test_rmse = np.sqrt(test_mse)
                    test_r2 = r2_score(y_test, test_pred) * 100
                    
                    results.update({
                        'test_mse': test_mse,
                        'test_rmse': test_rmse,
                        'test_r2': test_r2,
                        'rmse': test_rmse,
                        'accuracy': test_r2
                    })
                else:
                    results.update({
                        'rmse': train_rmse,
                        'accuracy': train_r2
                    })
            
            # Store the trained model
            self.trained_models[model_name] = model
            self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None
    
    def predict_next_price(self, model_name, features):
        """
        Predict next price using trained model
        
        Args:
            model_name (str): Name of the trained model
            features (array): Features for prediction
            
        Returns:
            float: Predicted price
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained yet")
            
            model = self.trained_models[model_name]
            
            # Ensure features is 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            prediction = model.predict(features)
            return float(prediction[0])
            
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {str(e)}")
            return None
    
    def get_model_performance(self, model_name):
        """Get performance metrics for a specific model"""
        return self.model_performances.get(model_name, {})
    
    def get_all_performances(self):
        """Get all model performances"""
        return self.model_performances
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance for tree-based models
        
        Args:
            model_name (str): Name of the trained model
            feature_names (list): Names of features
            
        Returns:
            dict: Feature importance scores
        """
        try:
            if model_name not in self.trained_models:
                return {}
            
            model = self.trained_models[model_name]
            
            # Check if model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                if feature_names is None:
                    feature_names = [f'Feature_{i}' for i in range(len(importances))]
                
                return dict(zip(feature_names, importances))
            else:
                return {}
                
        except Exception as e:
            st.error(f"Error getting feature importance for {model_name}: {str(e)}")
            return {}