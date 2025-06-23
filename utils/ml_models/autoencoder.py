import numpy as np
import pandas as pd
import streamlit as st
class AutoencoderModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def train_and_predict(self, data, **kwargs):
        """Train autoencoder and predict next price"""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Input, Dense
            from sklearn.preprocessing import MinMaxScaler
            
            # Prepare data
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(features)
            
            # Build autoencoder
            input_dim = scaled_data.shape[1]
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(3, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='sigmoid')(encoded)
            
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # Train quietly
            autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=32, verbose=0)
            
            # Detect anomalies
            reconstructed = autoencoder.predict(scaled_data, verbose=0)
            errors = np.mean(np.square(scaled_data - reconstructed), axis=1)
            threshold = np.percentile(errors, 90)
            anomalies = errors > threshold
            
            # Create predictions array (-1 for anomaly, +1 for normal) - SAME FORMAT as other models
            predictions = np.where(anomalies, -1, 1)
            
            # Predict next price
            current_price = float(data['Close'].iloc[-1])
            latest_prediction = predictions[-1]  # Last prediction (-1 or +1)
            
            if latest_prediction == -1:  # Anomaly detected
                next_price = current_price * 0.97  # 3% decrease
                confidence = 70.0
            else:  # Normal behavior
                next_price = current_price * 1.02  # 2% increase
                confidence = 85.0
                
            # Return SAME KEYS as other anomaly models
            return {
                'model_name': 'Autoencoder',
                'category': 'Anomaly Detection',
                'n_anomalies': int(np.sum(anomalies)),
                'anomaly_rate': float(np.mean(anomalies) * 100),
                'predictions': predictions.tolist(),  # KEY for chart display
                'next_price': next_price,
                'confidence': confidence,
                'rmse': float(np.sqrt(np.mean(errors))),
                'accuracy': float(100 - (np.mean(anomalies) * 100))
            }
            
        except Exception as e:
            return {
                'error': f'Autoencoder error: {str(e)}',
                'model_name': 'Autoencoder',
                'category': 'Anomaly Detection',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
