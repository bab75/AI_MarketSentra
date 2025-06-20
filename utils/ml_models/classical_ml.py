import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import streamlit as st

class ClassicalMLModels:
    """Classical Machine Learning Models for Stock Prediction"""
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'Gaussian Naive Bayes': GaussianNB()
        }
        
        self.trained_models = {}
        self.model_performances = {}
    
    def get_available_models(self):
        """Return list of available classical ML models"""
        return list(self.models.keys())
    
    def train_model(self, model_name, X_train, y_train, X_test=None, y_test=None):
        """
        Train a specific classical ML model
        
        Args:
            model_name (str): Name of the model to train
            X_train (array): Training features
            y_train (array): Training target
            X_test (array): Test features
            y_test (array): Test target
            
        Returns:
            dict: Training results and metrics
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            # Handle classification models for direction prediction
            if model_name == 'Gaussian Naive Bayes':
                # Convert regression to classification (up/down prediction)
                y_train_class = (y_train > y_train.shift(1)).astype(int)
                y_train_class = y_train_class.dropna()
                X_train_class = X_train.iloc[1:]  # Align with y_train_class
                
                model.fit(X_train_class, y_train_class)
                
                if X_test is not None and y_test is not None:
                    y_test_class = (y_test > y_test.shift(1)).astype(int)
                    y_test_class = y_test_class.dropna()
                    X_test_class = X_test.iloc[1:]
                    
                    predictions = model.predict(X_test_class)
                    accuracy = accuracy_score(y_test_class, predictions)
                    
                    results = {
                        'model': model,
                        'predictions': predictions,
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
            float: Predicted price or direction
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
    
    def cross_validate_model(self, model_name, X, y, cv=5):
        """
        Perform cross-validation on a model
        
        Args:
            model_name (str): Name of the model
            X (array): Features
            y (array): Target
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation results
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            
            if model_name == 'Gaussian Naive Bayes':
                # Classification cross-validation
                y_class = (y > y.shift(1)).astype(int).dropna()
                X_class = X.iloc[1:]
                scores = cross_val_score(model, X_class, y_class, cv=cv, scoring='accuracy')
            else:
                # Regression cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            
            return {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'all_scores': scores.tolist()
            }
            
        except Exception as e:
            st.error(f"Error in cross-validation for {model_name}: {str(e)}")
            return None
    
    def get_feature_importance(self, model_name):
        """
        Get feature importance for tree-based models
        
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
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_)
            else:
                return None
                
        except Exception as e:
            st.error(f"Error getting feature importance for {model_name}: {str(e)}")
            return None
    
    def compare_models(self, X, y, test_size=0.2):
        """
        Compare performance of all classical ML models
        
        Args:
            X (array): Features
            y (array): Target
            test_size (float): Proportion of test data
            
        Returns:
            dict: Comparison results for all models
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )
            
            comparison_results = {}
            
            for model_name in self.models.keys():
                results = self.train_model(model_name, X_train, y_train, X_test, y_test)
                if results:
                    comparison_results[model_name] = results
            
            return comparison_results
            
        except Exception as e:
            st.error(f"Error comparing models: {str(e)}")
            return {}
    
    def get_model_config(self, model_name):
        """
        Get configuration parameters for a specific model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration parameters
        """
        configs = {
            'Linear Regression': {
                'fit_intercept': True,
                'normalize': False
            },
            'Ridge Regression': {
                'alpha': 1.0,
                'fit_intercept': True,
                'max_iter': 1000
            },
            'Lasso Regression': {
                'alpha': 1.0,
                'fit_intercept': True,
                'max_iter': 1000
            },
            'Elastic Net': {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'fit_intercept': True,
                'max_iter': 1000
            },
            'Decision Tree': {
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'Support Vector Regression': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            },
            'K-Nearest Neighbors': {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto'
            },
            'Gaussian Naive Bayes': {
                'var_smoothing': 1e-9
            }
        }
        
        return configs.get(model_name, {})
    
    def update_model_params(self, model_name, params):
        """
        Update model parameters
        
        Args:
            model_name (str): Name of the model
            params (dict): New parameters
        """
        try:
            if model_name in self.models:
                self.models[model_name].set_params(**params)
                return True
            return False
            
        except Exception as e:
            st.error(f"Error updating parameters for {model_name}: {str(e)}")
            return False
