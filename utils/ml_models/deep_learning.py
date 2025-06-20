import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten,
    Dropout, BatchNormalization, Input, Attention,
    MultiHeadAttention, LayerNormalization, Embedding
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

class DeepLearningModels:
    """Deep Learning Models for Stock Prediction"""
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.scalers = {}
        self.model_performances = {}
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
    
    def get_available_models(self):
        """Return list of available deep learning models"""
        return [
            'LSTM', 'GRU', 'CNN-LSTM', 'Bidirectional LSTM',
            'Deep Neural Network', 'Attention LSTM', 'Transformer',
            'Stacked LSTM', 'CNN', 'Autoencoder'
        ]
    
    def prepare_sequences(self, data, sequence_length=60, target_col='Close'):
        """
        Prepare sequences for time series prediction
        
        Args:
            data (DataFrame): Input data
            sequence_length (int): Length of input sequences
            target_col (str): Name of target column
            
        Returns:
            tuple: (X, y) sequences for training
        """
        try:
            # Scale the data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            self.scalers[target_col] = scaler
            
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, data.columns.get_loc(target_col)])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            st.error(f"Error preparing sequences: {str(e)}")
            return None, None
    
    def build_lstm_model(self, input_shape, units=50, dropout_rate=0.2):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def build_gru_model(self, input_shape, units=50, dropout_rate=0.2):
        """Build GRU model"""
        model = Sequential([
            GRU(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            GRU(units, return_sequences=True),
            Dropout(dropout_rate),
            GRU(units),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def build_cnn_lstm_model(self, input_shape, filters=64, kernel_size=3, lstm_units=50):
        """Build CNN-LSTM model"""
        model = Sequential([
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=filters//2, kernel_size=kernel_size, activation='relu'),
            LSTM(lstm_units, return_sequences=True),
            LSTM(lstm_units),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def build_bidirectional_lstm_model(self, input_shape, units=50, dropout_rate=0.2):
        """Build Bidirectional LSTM model"""
        from tensorflow.keras.layers import Bidirectional
        
        model = Sequential([
            Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape),
            Dropout(dropout_rate),
            Bidirectional(LSTM(units)),
            Dropout(dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def build_dnn_model(self, input_shape, hidden_units=[128, 64, 32], dropout_rate=0.3):
        """Build Deep Neural Network model"""
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        
        for units in hidden_units:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def build_attention_lstm_model(self, input_shape, lstm_units=50):
        """Build LSTM model with attention mechanism"""
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
        lstm_out = LSTM(lstm_units, return_sequences=True)(lstm_out)
        
        # Attention layer
        attention_out = MultiHeadAttention(num_heads=4, key_dim=lstm_units)(lstm_out, lstm_out)
        attention_out = LayerNormalization()(attention_out + lstm_out)
        
        # Global average pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
        
        # Output layers
        dense_out = Dense(25, activation='relu')(pooled)
        outputs = Dense(1)(dense_out)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def build_transformer_model(self, input_shape, d_model=64, num_heads=4, num_layers=2):
        """Build Transformer model"""
        inputs = Input(shape=input_shape)
        
        # Positional encoding
        x = Dense(d_model)(inputs)
        
        # Transformer blocks
        for _ in range(num_layers):
            # Multi-head attention
            attention_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
            attention_out = LayerNormalization()(attention_out + x)
            
            # Feed forward network
            ffn_out = Dense(d_model * 2, activation='relu')(attention_out)
            ffn_out = Dense(d_model)(ffn_out)
            x = LayerNormalization()(ffn_out + attention_out)
        
        # Global average pooling and output
        pooled = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = Dense(1)(pooled)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def build_stacked_lstm_model(self, input_shape, units=[100, 50, 25], dropout_rate=0.2):
        """Build Stacked LSTM model"""
        model = Sequential()
        
        for i, unit in enumerate(units):
            return_sequences = i < len(units) - 1
            if i == 0:
                model.add(LSTM(unit, return_sequences=return_sequences, input_shape=input_shape))
            else:
                model.add(LSTM(unit, return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))
        
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def build_cnn_model(self, input_shape, filters=[64, 32], kernel_sizes=[3, 3]):
        """Build CNN model"""
        model = Sequential()
        
        for i, (filter_size, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            if i == 0:
                model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, 
                                activation='relu', input_shape=input_shape))
            else:
                model.add(Conv1D(filters=filter_size, kernel_size=kernel_size, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
        
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    def build_autoencoder_model(self, input_shape, encoding_dim=32):
        """Build Autoencoder model for feature learning"""
        # Encoder
        input_layer = Input(shape=input_shape)
        encoded = LSTM(encoding_dim)(input_layer)
        
        # Decoder
        decoded = tf.keras.layers.RepeatVector(input_shape[0])(encoded)
        decoded = LSTM(input_shape[1], return_sequences=True)(decoded)
        
        # Autoencoder
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Prediction model
        predictor = Sequential([
            Input(shape=input_shape),
            LSTM(encoding_dim),
            Dense(25),
            Dense(1)
        ])
        predictor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        
        return {'autoencoder': autoencoder, 'predictor': predictor}
    
    def train_model(self, model_name, X_train, y_train, X_test=None, y_test=None, 
                   epochs=50, batch_size=32, validation_split=0.2):
        """
        Train a specific deep learning model
        
        Args:
            model_name (str): Name of the model to train
            X_train (array): Training features
            y_train (array): Training target
            X_test (array): Test features
            y_test (array): Test target
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Validation split ratio
            
        Returns:
            dict: Training results and metrics
        """
        try:
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            # Build model based on name
            if model_name == 'LSTM':
                model = self.build_lstm_model(input_shape)
            elif model_name == 'GRU':
                model = self.build_gru_model(input_shape)
            elif model_name == 'CNN-LSTM':
                model = self.build_cnn_lstm_model(input_shape)
            elif model_name == 'Bidirectional LSTM':
                model = self.build_bidirectional_lstm_model(input_shape)
            elif model_name == 'Deep Neural Network':
                model = self.build_dnn_model(input_shape)
            elif model_name == 'Attention LSTM':
                model = self.build_attention_lstm_model(input_shape)
            elif model_name == 'Transformer':
                model = self.build_transformer_model(input_shape)
            elif model_name == 'Stacked LSTM':
                model = self.build_stacked_lstm_model(input_shape)
            elif model_name == 'CNN':
                model = self.build_cnn_model(input_shape)
            elif model_name == 'Autoencoder':
                models = self.build_autoencoder_model(input_shape)
                # Train autoencoder first
                models['autoencoder'].fit(X_train, X_train, epochs=epochs//2, 
                                        batch_size=batch_size, verbose=0)
                model = models['predictor']
            else:
                raise ValueError(f"Model {model_name} not implemented")
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            results = {'model': model, 'history': history.history}
            
            if X_test is not None and y_test is not None:
                predictions = model.predict(X_test, verbose=0)
                
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, predictions)
                
                # Calculate percentage accuracy
                tolerance = 0.05
                accuracy = np.mean(np.abs((predictions.flatten() - y_test) / y_test) <= tolerance) * 100
                
                results.update({
                    'predictions': predictions.flatten(),
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'accuracy': accuracy
                })
            
            # Store trained model
            self.trained_models[model_name] = model
            self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None
    
    def predict_next_price(self, model_name, features, scaler_key='Close'):
        """
        Predict next price using trained model
        
        Args:
            model_name (str): Name of the trained model
            features (array): Features for prediction
            scaler_key (str): Key for the scaler used
            
        Returns:
            float: Predicted price
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained")
            
            model = self.trained_models[model_name]
            
            # Reshape features for prediction
            features_reshaped = features.reshape(1, features.shape[0], features.shape[1])
            
            # Make prediction
            prediction = model.predict(features_reshaped, verbose=0)
            
            # Inverse transform if scaler is available
            if scaler_key in self.scalers:
                # Create dummy array for inverse transform
                dummy = np.zeros((1, len(self.scalers[scaler_key].scale_)))
                dummy[0, 0] = prediction[0, 0]  # Assuming target is first column
                prediction_scaled = self.scalers[scaler_key].inverse_transform(dummy)
                return prediction_scaled[0, 0]
            
            return prediction[0, 0]
            
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {str(e)}")
            return None
    
    def get_model_config(self, model_name):
        """
        Get configuration parameters for a specific deep learning model
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration parameters
        """
        configs = {
            'LSTM': {
                'units': 50,
                'dropout_rate': 0.2,
                'sequence_length': 60,
                'epochs': 50,
                'batch_size': 32
            },
            'GRU': {
                'units': 50,
                'dropout_rate': 0.2,
                'sequence_length': 60,
                'epochs': 50,
                'batch_size': 32
            },
            'CNN-LSTM': {
                'filters': 64,
                'kernel_size': 3,
                'lstm_units': 50,
                'sequence_length': 60,
                'epochs': 50,
                'batch_size': 32
            },
            'Transformer': {
                'd_model': 64,
                'num_heads': 4,
                'num_layers': 2,
                'sequence_length': 60,
                'epochs': 50,
                'batch_size': 32
            }
        }
        
        return configs.get(model_name, {
            'sequence_length': 60,
            'epochs': 50,
            'batch_size': 32
        })
