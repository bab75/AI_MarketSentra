import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import streamlit as st

# Safe TensorFlow import
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except (ImportError, AttributeError, TypeError):
    TENSORFLOW_AVAILABLE = False
    tf = None

class DeepLearningModels:
    """Deep Learning Models for Stock Prediction (Safe Version)"""
    
    def __init__(self):
        self.available_models = []
        self.trained_models = {}
        self.model_performances = {}
        self.scalers = {}
        
        # Only add models if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self.available_models = {
                'Deep Learning Models': ['LSTM', 'GRU', 'CNN-LSTM', 'Simple Neural Network']
            }
        else:
            st.warning("TensorFlow not available. Deep learning models are disabled.")
    
    def get_available_models(self):
        """Return dictionary of available deep learning models"""
        return self.available_models
    
    def prepare_sequences(self, data, sequence_length=60):
        """
        Prepare sequences for deep learning models
        
        Args:
            data (DataFrame): Stock data
            sequence_length (int): Length of input sequences
            
        Returns:
            tuple: (X, y, scaler)
        """
        if not TENSORFLOW_AVAILABLE:
            st.error("TensorFlow not available for sequence preparation")
            return None, None, None
            
        try:
            # Use Close price as target
            values = data['Close'].values
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(values.reshape(-1, 1))
            
            # Store scaler for later use
            self.scalers['Close'] = scaler
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(scaled)):
                X.append(scaled[i-sequence_length:i, 0])
                y.append(scaled[i, 0])
            
            return np.array(X), np.array(y), scaler
            
        except Exception as e:
            st.error(f"Error preparing sequences: {str(e)}")
            return None, None, None
    
    def train_model(self, model_name, X_train, y_train, X_test=None, y_test=None, 
                   epochs=50, batch_size=32, **kwargs):
        """
        Train a deep learning model
        
        Args:
            model_name (str): Name of the model
            X_train (array): Training features
            y_train (array): Training target
            X_test (array): Test features
            y_test (array): Test target
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            
        Returns:
            dict: Training results
        """
        if not TENSORFLOW_AVAILABLE:
            return {
                'error': 'TensorFlow not available',
                'model_name': model_name,
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
        
        try:
            # Reshape data for LSTM/GRU
            if model_name in ['LSTM', 'GRU', 'CNN-LSTM']:
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                if X_test is not None:
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build model
            model = self._build_model(model_name, X_train.shape)
            if model is None:
                return {
                    'error': f'Failed to build model {model_name}',
                    'model_name': model_name,
                    'category': 'Deep Learning Models',
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            # Compile model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )
            
            # Make predictions
            train_pred = model.predict(X_train, verbose=0)
            test_pred = model.predict(X_test, verbose=0) if X_test is not None else None
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_r2 = r2_score(y_train, train_pred) * 100
            
            results = {
                'model_name': model_name,
                'category': 'Deep Learning Models',
                'train_rmse': float(train_rmse),
                'train_r2': float(train_r2),
                'epochs_trained': epochs,
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else None
            }
            
            if X_test is not None and y_test is not None:
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_r2 = r2_score(y_test, test_pred) * 100
                
                # Get next price (inverse-scaled)
                last_pred = test_pred[-1][0]
                if 'Close' in self.scalers:
                    last_pred = self.scalers['Close'].inverse_transform([[last_pred]])[0][0]
                
                results.update({
                    'test_rmse': float(test_rmse),
                    'test_r2': float(test_r2),
                    'rmse': float(test_rmse),
                    'accuracy': float(test_r2),
                    'next_price': float(last_pred),
                    'confidence': float(test_r2)
                })
            else:
                # Get next price (inverse-scaled)
                last_pred = train_pred[-1][0]
                if 'Close' in self.scalers:
                    last_pred = self.scalers['Close'].inverse_transform([[last_pred]])[0][0]
                
                results.update({
                    'rmse': float(train_rmse),
                    'accuracy': float(train_r2),
                    'next_price': float(last_pred),
                    'confidence': float(train_r2)
                })
            
            # Store trained model
            self.trained_models[model_name] = model
            self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return {
                'error': str(e),
                'model_name': model_name,
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _build_model(self, model_name, input_shape):
        """Build specific model architecture"""
        if not TENSORFLOW_AVAILABLE:
            return None
            
        try:
            model = Sequential()
            
            if model_name == 'LSTM':
                model.add(LSTM(50, return_sequences=True, input_shape=input_shape[1:]))
                model.add(Dropout(0.2))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(25))
                model.add(Dense(1))
                
            elif model_name == 'GRU':
                model.add(GRU(50, return_sequences=True, input_shape=input_shape[1:]))
                model.add(Dropout(0.2))
                model.add(GRU(50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(25))
                model.add(Dense(1))
                
            elif model_name == 'CNN-LSTM':
                model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape[1:]))
                model.add(MaxPooling1D(pool_size=2))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dropout(0.2))
                model.add(Dense(25))
                model.add(Dense(1))
                
            elif model_name == 'Simple Neural Network':
                model.add(Flatten(input_shape=input_shape[1:]))
                model.add(Dense(50, activation='relu'))
                model.add(Dropout(0.2))
                model.add(Dense(25, activation='relu'))
                model.add(Dense(1))
                
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            return model
            
        except Exception as e:
            st.error(f"Error building model {model_name}: {str(e)}")
            return None
    
    def predict_next_price(self, model_name, sequence):
    """
    Predict next price using trained model
    
    Args:
        model_name (str): Name of trained model
        sequence (array): Input sequence
        
    Returns:
        float: Predicted price
    """
    if not TENSORFLOW_AVAILABLE:
        return None
        
    try:
        if model_name not in self.trained_models:
            return None
        
        model = self.trained_models[model_name]
        
        # Validate and reshape sequence
        if sequence is None or not np.any(sequence):
            st.error(f"Invalid sequence for {model_name}")
            return None
            
        if model_name in ['LSTM', 'GRU', 'CNN-LSTM']:
            if sequence.ndim == 1:
                sequence = sequence.reshape(1, sequence.shape[0], 1)
            elif sequence.ndim == 2:
                sequence = sequence.reshape(1, sequence.shape[0], 1)
        elif model_name == 'Simple Neural Network':
            if sequence.ndim == 1:
                expected_features = 25  # 5 features * 5 lookback
                if sequence.shape[0] != expected_features:
                    st.error(f"Expected {expected_features} features, got {sequence.shape[0]}")
                    return None
                sequence = sequence.reshape(1, expected_features)
            elif sequence.ndim == 2:
                sequence = sequence.reshape(1, sequence.shape[1])
        
        # Check for NaNs
        if np.any(np.isnan(sequence)):
            st.error(f"NaN values detected in sequence for {model_name}")
            return None
            
        # Make prediction
        prediction = model.predict(sequence, verbose=0)
        
        # Inverse transform if scaler available
        if 'Close' in self.scalers:
            prediction = self.scalers['Close'].inverse_transform(prediction)
        
        return float(prediction[0][0])
        
        except Exception as e:
        st.error(f"Error predicting with {model_name}: {str(e)}")
        return None
    
    def train_and_predict(self, data, model_name, **kwargs):
        """
        Train a deep learning model and predict next price
        
        Args:
            data (DataFrame): Input data with Close column
            model_name (str): Name of the model to train
            **kwargs: Additional parameters (e.g., sequence_length, epochs)
            
        Returns:
            dict: Training results with predictions and metrics
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                return {
                    'error': 'TensorFlow not available',
                    'model_name': model_name,
                    'category': 'Deep Learning Models',
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            if model_name not in self.available_models['Deep Learning Models']:
                return {
                    'error': f'Model {model_name} not found',
                    'model_name': model_name,
                    'category': 'Deep Learning Models',
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            # Prepare data
            sequence_length = kwargs.get('sequence_length', 60)
            X, y, scaler = self.prepare_sequences(data, sequence_length=sequence_length)
            if X is None or y is None or scaler is None:
                return {
                    'error': 'Could not prepare data',
                    'model_name': model_name,
                    'category': 'Deep Learning Models',
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Train model
            results = self.train_model(
                model_name,
                X_train,
                y_train,
                X_test=X_test,
                y_test=y_test,
                epochs=kwargs.get('epochs', 50),
                batch_size=kwargs.get('batch_size', 32)
            )
            
            if 'error' in results:
                return results
            
            # Predict next price using last sequence
            last_sequence = X[-1]
            next_price = self.predict_next_price(model_name, last_sequence)
            if next_price is None:
                next_price = results['next_price']  # Fallback to train_model's prediction
            
            # Update results with next_price
            results.update({
                'next_price': float(next_price),
                'confidence': results.get('confidence', results['accuracy'])
            })
            
            return results
            
        except Exception as e:
            st.error(f"Error in train_and_predict for {model_name}: {str(e)}")
            return {
                'error': str(e),
                'model_name': model_name,
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def get_model_performance(self, model_name):
        """Get performance metrics for a specific model"""
        return self.model_performances.get(model_name, {})
    
    def get_all_performances(self):
        """Get all model performances"""
        return self.model_performances
