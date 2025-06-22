import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Conv1D, MaxPooling1D, Flatten
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
            if data[['Open', 'High', 'Low', 'Close', 'Volume']].isna().any().any():
                return {
                    'error': 'Input data contains NaN values',
                    'model_name': model_name,
                    'category': 'Deep Learning Models',
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            X, y, scaler = self._prepare_sequential_data(data)
            if X is None or y is None or scaler is None:
                return {
                    'error': 'Failed to prepare sequential data',
                    'model_name': model_name,
                    'category': 'Deep Learning Models',
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            if model_name == 'lstm':
                return self._train_lstm(X_train, X_test, y_train, y_test, scaler, data, **kwargs)
            elif model_name == 'gru':
                return self._train_gru(X_train, X_test, y_train, y_test, scaler, data, **kwargs)
            elif model_name == 'feedforward_nn':
                return self._train_feedforward(X_train, X_test, y_train, y_test, scaler, data, **kwargs)
            elif model_name == 'cnn':
                return self._train_cnn(X_train, X_test, y_train, y_test, scaler, data, **kwargs)
            elif model_name == 'rnn':
                return self._train_rnn(X_train, X_test, y_train, y_test, scaler, data, **kwargs)
            elif model_name == 'autoencoder':
                return self._train_autoencoder(X_train, X_test, y_train, y_test, scaler, data, **kwargs)
            elif model_name == 'mlp':
                return self._train_mlp(X_train, X_test, y_train, y_test, scaler, data, **kwargs)
            else:
                return {
                    'error': f'Model {model_name} not implemented',
                    'model_name': model_name,
                    'category': 'Deep Learning Models',
                    'next_price': 0.0,
                    'accuracy': 0.0,
                    'confidence': 0.0,
                    'rmse': float('inf')
                }
                
        except Exception as e:
            st.error(f"Deep learning error: {str(e)}")
            return {
                'error': str(e),
                'model_name': model_name,
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _prepare_sequential_data(self, data, sequence_length=60):
        """Prepare sequential data for deep learning"""
        try:
            data = data.dropna()
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)
            self.scalers['Close'] = scaler
            
            X, y = [], []
            for i in range(sequence_length, len(scaled_features)):
                X.append(scaled_features[i-sequence_length:i])
                y.append(scaled_features[i, 3])
            
            X, y = np.array(X), np.array(y)
            if np.any(np.isnan(X)) or np.any(np.isnan(y)):
                return None, None, None
                
            return X, y, scaler
        except Exception as e:
            st.error(f"Error preparing sequential data: {str(e)}")
            return None, None, None
    
    def _train_lstm(self, X_train, X_test, y_train, y_test, scaler, data, **kwargs):
        """Train LSTM model"""
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
            
            test_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            r2 = float(r2_score(y_test, test_pred) * 100)
            
            last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Long Short-Term Memory',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'confidence': float(max(0, min(100, r2))),
                'accuracy': float(r2),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training LSTM: {str(e)}")
            return {
                'error': str(e),
                'model_name': 'Long Short-Term Memory',
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_gru(self, X_train, X_test, y_train, y_test, scaler, data, **kwargs):
        """Train GRU model"""
        try:
            model = Sequential([
                GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                GRU(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
            
            test_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            r2 = float(r2_score(y_test, test_pred) * 100)
            
            last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Gated Recurrent Units',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'confidence': float(max(0, min(100, r2))),
                'accuracy': float(r2),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training GRU: {str(e)}")
            return {
                'error': str(e),
                'model_name': 'Gated Recurrent Units',
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_feedforward(self, X_train, X_test, y_train, y_test, scaler, data, **kwargs):
        """Train Feedforward Neural Network model"""
        try:
            model = Sequential([
                Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
            
            test_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            r2 = float(r2_score(y_test, test_pred) * 100)
            
            last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Feedforward Neural Network',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'confidence': float(max(0, min(100, r2))),
                'accuracy': float(r2),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training Feedforward NN: {str(e)}")
            return {
                'error': str(e),
                'model_name': 'Feedforward Neural Network',
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_cnn(self, X_train, X_test, y_train, y_test, scaler, data, **kwargs):
        """Train Convolutional Neural Network model"""
        try:
            model = Sequential([
                Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
                MaxPooling1D(pool_size=2),
                Conv1D(32, kernel_size=3, activation='relu'),
                Flatten(),
                Dense(50, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
            
            test_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            r2 = float(r2_score(y_test, test_pred) * 100)
            
            last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Convolutional Neural Network',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'confidence': float(max(0, min(100, r2))),
                'accuracy': float(r2),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training CNN: {str(e)}")
            return {
                'error': str(e),
                'model_name': 'Convolutional Neural Network',
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_rnn(self, X_train, X_test, y_train, y_test, scaler, data, **kwargs):
        """Train Recurrent Neural Network model"""
        try:
            from tensorflow.keras.layers import SimpleRNN
            model = Sequential([
                SimpleRNN(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                SimpleRNN(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
            
            test_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            r2 = float(r2_score(y_test, test_pred) * 100)
            
            last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Recurrent Neural Network',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'confidence': float(max(0, min(100, r2))),
                'accuracy': float(r2),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training RNN: {str(e)}")
            return {
                'error': str(e),
                'model_name': 'Recurrent Neural Network',
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_autoencoder(self, X_train, X_test, y_train, y_test, scaler, data, **kwargs):
        """Train Autoencoder model"""
        try:
            input_shape = (X_train.shape[1], X_train.shape[2])
            input_layer = Input(shape=input_shape)
            encoded = LSTM(32)(input_layer)
            decoded = tf.keras.layers.RepeatVector(input_shape[0])(encoded)
            decoded = LSTM(input_shape[1], return_sequences=True)(decoded)
            
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, verbose=0)
            
            predictor = Sequential([
                Input(shape=input_shape),
                LSTM(32),
                Dense(25),
                Dense(1)
            ])
            
            predictor.compile(optimizer='adam', loss='mean_squared_error')
            predictor.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
            
            test_pred = predictor.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            r2 = float(r2_score(y_test, test_pred) * 100)
            
            last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
            prediction = predictor.predict(last_sequence, verbose=0)[0][0]
            
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Autoencoder',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'confidence': float(max(0, min(100, r2))),
                'accuracy': float(r2),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training Autoencoder: {str(e)}")
            return {
                'error': str(e),
                'model_name': 'Autoencoder',
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
    
    def _train_mlp(self, X_train, X_test, y_train, y_test, scaler, data, **kwargs):
        """Train Multilayer Perceptron model"""
        try:
            model = Sequential([
                Flatten(input_shape=(X_train.shape[1], X_train.shape[2])),
                Dense(100, activation='relu'),
                Dropout(0.2),
                Dense(50, activation='relu'),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
            
            test_pred = model.predict(X_test, verbose=0)
            rmse = float(np.sqrt(mean_squared_error(y_test, test_pred)))
            r2 = float(r2_score(y_test, test_pred) * 100)
            
            last_sequence = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
            prediction = model.predict(last_sequence, verbose=0)[0][0]
            
            dummy_array = np.zeros((1, 5))
            dummy_array[0, 3] = prediction
            prediction_original = scaler.inverse_transform(dummy_array)[0, 3]
            
            return {
                'model_name': 'Multilayer Perceptron',
                'category': 'Deep Learning Models',
                'next_price': float(prediction_original),
                'confidence': float(max(0, min(100, r2))),
                'accuracy': float(r2),
                'rmse': rmse
            }
        except Exception as e:
            st.error(f"Error training MLP: {str(e)}")
            return {
                'error': str(e),
                'model_name': 'Multilayer Perceptron',
                'category': 'Deep Learning Models',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
