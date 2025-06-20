import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
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
            self.available_models = ['LSTM', 'GRU', 'CNN-LSTM', 'Simple Neural Network']
        else:
            st.warning("TensorFlow not available. Deep learning models are disabled.")
    
    def get_available_models(self):
        """Return list of available deep learning models"""
        return self.available_models
    
    def prepare_sequences(self, data, sequence_length=60):
        """
        Prepare sequences for deep learning models
        
        Args:
            data (DataFrame): Stock data
            sequence_length (int): Length of input sequences
            
        Returns:
            tuple: (X, y) arrays for training
        """
        if not TENSORFLOW_AVAILABLE:
            st.error("TensorFlow not available for sequence preparation")
            return None, None
            
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
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            st.error(f"Error preparing sequences: {str(e)}")
            return None, None
    
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
                'accuracy': 0,
                'rmse': float('inf')
            }
        
        try:
            # Reshape data for LSTM/GRU
            if model_name in ['LSTM', 'GRU', 'CNN-LSTM']:
                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                if X_test is not None:
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build model based on type
            model = self._build_model(model_name, X_train.shape)
            
            if model is None:
                return None
            
            # Compile model
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
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
                'train_rmse': train_rmse,
                'train_r2': train_r2,
                'epochs_trained': epochs,
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None
            }
            
            if X_test is not None and y_test is not None:
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_r2 = r2_score(y_test, test_pred) * 100
                
                results.update({
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
            
            # Store trained model
            self.trained_models[model_name] = model
            self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None
    
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
                # Flatten for simple NN
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
            
            # Reshape sequence if needed
            if sequence.ndim == 1:
                sequence = sequence.reshape(1, sequence.shape[0], 1)
            elif sequence.ndim == 2:
                sequence = sequence.reshape(1, sequence.shape[0], sequence.shape[1])
            
            # Make prediction
            prediction = model.predict(sequence, verbose=0)
            
            # Inverse transform if scaler available
            if 'Close' in self.scalers:
                prediction = self.scalers['Close'].inverse_transform(prediction)
            
            return float(prediction[0][0])
            
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {str(e)}")
            return None
    
    def get_model_performance(self, model_name):
        """Get performance metrics for a specific model"""
        return self.model_performances.get(model_name, {})
    
    def get_all_performances(self):
        """Get all model performances"""
        return self.model_performances