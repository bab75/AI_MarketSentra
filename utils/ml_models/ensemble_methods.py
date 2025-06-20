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
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    cb = None
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
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE and lgb is not None:
            self.models['LightGBM'] = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE and cb is not None:
            self.models['CatBoost'] = cb.CatBoostRegressor(n_estimators=100, random_state=42, verbose=False)
        
        self.classification_models = {
            'Random Forest (Classification)': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting (Classification)': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'AdaBoost (Classification)': AdaBoostClassifier(n_estimators=50, random_state=42),
            'Extra Trees (Classification)': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'XGBoost (Classification)': xgb.XGBClassifier(n_estimators=100, random_state=42)
        }
        
        # Add CatBoost classification if available
        if CATBOOST_AVAILABLE and cb is not None:
            self.classification_models['CatBoost (Classification)'] = cb.CatBoostClassifier(n_estimators=100, random_state=42, verbose=False)
        
        # Add LightGBM classification if available
        if LIGHTGBM_AVAILABLE and lgb is not None:
            self.classification_models['LightGBM (Classification)'] = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        
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
                
                # Convert target to classification (price direction)
                y_train_class = (y_train > y_train.shift(1)).astype(int).dropna()
                X_train_class = X_train.iloc[1:]  # Align with y_train_class
                
                model.fit(X_train_class, y_train_class)
                
                if X_test is not None and y_test is not None:
                    y_test_class = (y_test > y_test.shift(1)).astype(int).dropna()
                    X_test_class = X_test.iloc[1:]
                    
                    predictions = model.predict(X_test_class)
                    probabilities = model.predict_proba(X_test_class)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    accuracy = accuracy_score(y_test_class, predictions)
                    
                    results = {
                        'model': model,
                        'predictions': predictions,
                        'probabilities': probabilities,
                        'accuracy': accuracy * 100,
                        'model_type': 'classification'
                    }
                else:
                    results = {
                        'model': model,
                        'model_type': 'classification'
                    }
            
            else:
                # Regression models
                if model_name not in self.models:
                    raise ValueError(f"Model {model_name} not found")
                
                model = self.models[model_name]
                model.fit(X_train, y_train)
                
                if X_test is not None and y_test is not None:
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, predictions)
                    
                    # Calculate percentage accuracy (within 5% tolerance)
                    tolerance = 0.05
                    accuracy = np.mean(np.abs((predictions - y_test) / y_test) <= tolerance) * 100
                    
                    results = {
                        'model': model,
                        'predictions': predictions,
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'accuracy': accuracy,
                        'model_type': 'regression'
                    }
                else:
                    results = {
                        'model': model,
                        'model_type': 'regression'
                    }
            
            # Store trained model
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
            float: Predicted price or probability
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained")
            
            model = self.trained_models[model_name]
            prediction = model.predict(features.reshape(1, -1))
            
            return prediction[0]
            
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {str(e)}")
            return None
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance for ensemble models
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            array: Feature importance scores
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained")
            
            model = self.trained_models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            else:
                return None
                
        except Exception as e:
            st.error(f"Error getting feature importance for {model_name}: {str(e)}")
            return None
    
    def hyperparameter_tuning(self, model_name, X, y, param_grid=None, cv=3):
        """
        Perform hyperparameter tuning for ensemble models
        
        Args:
            model_name (str): Name of the model
            X (array): Features
            y (array): Target
            param_grid (dict): Parameter grid for tuning
            cv (int): Cross-validation folds
            
        Returns:
            dict: Best parameters and score
        """
        try:
            if model_name not in self.models and model_name not in self.classification_models:
                raise ValueError(f"Model {model_name} not found")
            
            # Get model
            if model_name in self.models:
                model = self.models[model_name]
                scoring = 'r2'
            else:
                model = self.classification_models[model_name]
                scoring = 'accuracy'
                # Convert to classification target
                y = (y > y.shift(1)).astype(int).dropna()
                X = X.iloc[1:]
            
            # Default parameter grids
            if param_grid is None:
                param_grid = self.get_default_param_grid(model_name)
            
            # Perform grid search
            grid_search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_
            }
            
        except Exception as e:
            st.error(f"Error in hyperparameter tuning for {model_name}: {str(e)}")
            return None
    
    def get_default_param_grid(self, model_name):
        """Get default parameter grid for hyperparameter tuning"""
        param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'LightGBM': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'CatBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'depth': [3, 5, 7]
            }
        }
        
        return param_grids.get(model_name, {})
    
    def create_voting_ensemble(self, model_names, X_train, y_train, task_type='regression'):
        """
        Create a voting ensemble from multiple models
        
        Args:
            model_names (list): List of model names to include
            X_train (array): Training features
            y_train (array): Training target
            task_type (str): 'regression' or 'classification'
            
        Returns:
            object: Trained voting ensemble
        """
        try:
            estimators = []
            
            for name in model_names:
                if task_type == 'regression' and name in self.models:
                    estimators.append((name, self.models[name]))
                elif task_type == 'classification' and name in self.classification_models:
                    estimators.append((name, self.classification_models[name]))
            
            if not estimators:
                raise ValueError("No valid models found for ensemble")
            
            # Create voting ensemble
            if task_type == 'regression':
                ensemble = VotingRegressor(estimators=estimators)
            else:
                ensemble = VotingClassifier(estimators=estimators, voting='soft')
                # Convert to classification target
                y_train = (y_train > y_train.shift(1)).astype(int).dropna()
                X_train = X_train.iloc[1:]
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            return ensemble
            
        except Exception as e:
            st.error(f"Error creating voting ensemble: {str(e)}")
            return None
    
    def get_model_config(self, model_name):
        """
        Get configuration parameters for a specific ensemble model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration parameters
        """
        configs = {
            'Random Forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'bootstrap': True,
                'random_state': 42
            },
            'Gradient Boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'XGBoost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42
            },
            'LightGBM': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 1.0,
                'colsample_bytree': 1.0,
                'random_state': 42
            },
            'CatBoost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'bootstrap_type': 'Bayesian',
                'random_state': 42
            }
        }
        
        return configs.get(model_name, {})
