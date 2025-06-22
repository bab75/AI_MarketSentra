import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Conv1D, MaxPooling1D, Flatten, SimpleRNN
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

class DeepLearningManager:
    """Deep Learning Models for Financial Time Series"""
    
    def __init__(self):
        self.models = {
            'Deep Learning Models': {
                'Feedforward Neural Network': 'feedforward_nn',
                'Convolutional Neural Network': 'cnn',
                'Recurrent Neural Network': 'rnn',
                'Long Short-Term Memory': 'lstm',
                'Gated Recurrent Units': 'gru',
                'Autoencoder': 'autoencoder',
                'Multilayer Perceptron': 'mlp'
            }
        }
        
        self.trained_models = {}
        self.scalers = {}
    
    def get_available_models(self):
        """Get all available deep learning models"""
        return self.models
    
    def train_and_predict(self, data, model_name, **kwargs):
        """Train deep learning model and make predictions"""
        try:
            # Prepare sequential data
            X, y, scaler = self._prepare_sequential_data(data)
            if X is None or y is None or scaler is None:
                return {'error': 'Could not prepare sequential data'}
            
            if model_name == 'lstm':
                return self._train_lstm(X, y, scaler, data, **kwargs)
            elif model_name == 'gru':
                return self._train_gru(X, y, scaler, data, **kwargs)
            elif model_name == 'feedforward_nn':
                return self._train_feedforward(X, y, scaler, data, **kwargs)
            elif model_name == 'cnn':
                return self._train_cnn(X, y, scaler, data, **kwargs)
            elif model_name == 'rnn':
                return self._train_rnn(X, y, scaler, data, **kwargs)
            elif model_name == 'autoencoder':
                return self._train_autoencoder(X, y, scaler, data, **kwargs)
            elif model_name == 'mlp':
                return self._train_mlp(X, y, scaler, data, **kwargs)
            else:
                return {'error': f'Model {model_name} not implemented'}
                
        except Exception as e:
            st.error(f"Deep learning error: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_sequential_data(self, data, sequence_length=60):
        """Prepare sequential data for deep learning"""
        try:
            # Use basic OHLCV features
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            
            # Scale features
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(scaled_features)):
                X.append(scaled_features[i-sequence_length:i])
                y.append(scaled_features[i, 3])  # Close price
            
            return np.array(X), np.array(y), scaler
        except Exception as e:
            st.error(f"Error preparing sequential data: {str(e)}")
            return None, None, None
    
    def _train_lstm(self, X, y, scaler, data, **kwargs):
        """Train LSTM model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Build model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            # Evaluate on test set
            y_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred) * 100)
            
            # Make prediction for next price
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            # Transform back to original scale
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'LSTM',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'accuracy': max(0, min(100, r2)),
                'confidence': max(0, min(100, r2)),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training LSTM: {str(e)}")
            return {'error': str(e)}
    
    def _train_gru(self, X, y, scaler, data, **kwargs):
        """Train GRU model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Build model
            model = Sequential([
                GRU(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                GRU(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            # Evaluate on test set
            y_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred) * 100)
            
            # Make prediction for next price
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            # Transform back to original scale
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'GRU',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'accuracy': max(0, min(100, r2)),
                'confidence': max(0, min(100, r2)),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training GRU: {str(e)}")
            return {'error': str(e)}
    
    def _train_feedforward(self, X, y, scaler, data, **kwargs):
        """Train Feedforward Neural Network model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Build model
            model = Sequential([
                Flatten(input_shape=(X.shape[1], X.shape[2])),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            # Evaluate on test set
            y_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred) * 100)
            
            # Make prediction for next price
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            # Transform back to original scale
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Feedforward Neural Network',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'accuracy': max(0, min(100, r2)),
                'confidence': max(0, min(100, r2)),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training Feedforward NN: {str(e)}")
            return {'error': str(e)}
    
    def _train_cnn(self, X, y, scaler, data, **kwargs):
        """Train Convolutional Neural Network model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Build model
            model = Sequential([
                Conv1D(64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
                MaxPooling1D(pool_size=2),
                Conv1D(32, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            # Evaluate on test set
            y_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred) * 100)
            
            # Make prediction for next price
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            # Transform back to original scale
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Convolutional Neural Network',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'accuracy': max(0, min(100, r2)),
                'confidence': max(0, min(100, r2)),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training CNN: {str(e)}")
            return {'error': str(e)}
    
    def _train_rnn(self, X, y, scaler, data, **kwargs):
        """Train Recurrent Neural Network model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Build model
            model = Sequential([
                SimpleRNN(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                SimpleRNN(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            # Evaluate on test set
            y_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred) * 100)
            
            # Make prediction for next price
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            # Transform back to original scale
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Recurrent Neural Network',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'accuracy': max(0, min(100, r2)),
                'confidence': max(0, min(100, r2)),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training RNN: {str(e)}")
            return {'error': str(e)}
    
    def _train_autoencoder(self, X, y, scaler, data, **kwargs):
        """Train Autoencoder model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Build encoder
            input_layer = Input(shape=(X.shape[1], X.shape[2]))
            encoded = LSTM(32, activation='relu', return_sequences=False)(input_layer)
            encoded = Dense(16, activation='relu')(encoded)
            
            # Build decoder
            decoded = Dense(32, activation='relu')(encoded)
            decoded = Dense(X.shape[2], activation='sigmoid')(decoded)
            
            # Autoencoder model
            autoencoder = Model(input_layer, decoded)
            
            # Prediction model (use encoder + dense for price prediction)
            price_predictor = Sequential([
                LSTM(32, activation='relu', input_shape=(X.shape[1], X.shape[2])),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            price_predictor.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            # Train autoencoder
            X_train_reshaped = X_train[:, -1, :]  # Use last timestep for reconstruction
            X_test_reshaped = X_test[:, -1, :]
            autoencoder.fit(X_train_reshaped, X_train_reshaped, batch_size=32, epochs=50, verbose=0)
            
            # Train price predictor
            price_predictor.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            # Evaluate on test set
            y_pred = price_predictor.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred) * 100)
            
            # Make prediction for next price
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            prediction = price_predictor.predict(last_sequence, verbose=0)[0][0]
            
            # Transform back to original scale
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Autoencoder',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'accuracy': max(0, min(100, r2)),
                'confidence': max(0, min(100, r2)),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training Autoencoder: {str(e)}")
            return {'error': str(e)}
    
    def _train_mlp(self, X, y, scaler, data, **kwargs):
        """Train Multilayer Perceptron model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Build model
            model = Sequential([
                Flatten(input_shape=(X.shape[1], X.shape[2])),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
            
            # Evaluate on test set
            y_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred) * 100)
            
            # Make prediction for next price
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            # Transform back to original scale
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Multilayer Perceptron',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'accuracy': max(0, min(100, r2)),
                'confidence': max(0, min(100, r2)),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training MLP: {str(e)}")
            return {'error': str(e)}
