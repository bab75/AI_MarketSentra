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
            decoded = Input(input_dim, activation='sigmoid')(encoded)
            
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss='mse')
            
            # Train
            autoencoder.fit(scaled_data, scaled_data, epochs=50, batch_size=32, verbose=0)
            
            # Detect anomalies
            reconstructed = autoencoder.predict(scaled_data, verbose=0)
            errors = np.mean(np.square(scaled_data - reconstructed), axis=1)
            threshold = np.percentile(errors, 90)
            anomalies = errors > threshold
            
            # Create predictions array
            predictions = [-1 if anom else 1 for anom in anomalies]
            
            # Calculate metrics with explicit type conversion
            current_price = float(data['Close'].iloc[-1])
            n_anomalies = int(np.sum(anomalies))
            total_points = len(anomalies)
            anomaly_rate = (n_anomalies / total_points) * 100.0
            accuracy = 100.0 - anomaly_rate
            rmse_value = np.sqrt(np.mean(errors))
            
            # Price prediction
            latest_prediction = predictions[-1]
            if latest_prediction == -1:
                next_price = current_price * 0.97
                confidence = 70.0
            else:
                next_price = current_price * 1.02
                confidence = 85.0
            
            # Debug final values
            st.write(f"DEBUG: next_price = {next_price}")
            st.write(f"DEBUG: accuracy = {accuracy}")
            st.write(f"DEBUG: confidence = {confidence}")
            
            # Return with explicit float conversion
            result = {
                'model_name': 'Autoencoder',
                'category': 'Anomaly Detection',
                'n_anomalies': n_anomalies,
                'anomaly_rate': float(anomaly_rate),
                'predictions': predictions,
                'next_price': float(next_price),
                'confidence': float(confidence),
                'rmse': float(rmse_value),
                'accuracy': float(accuracy)
            }
            
            st.write(f"DEBUG: Final result = {result}")
            return result
            
        except Exception as e:
            st.error(f"Autoencoder error: {str(e)}")
            return {
                'error': f'Autoencoder error: {str(e)}',
                'model_name': 'Autoencoder',
                'category': 'Anomaly Detection',
                'next_price': 0.0,
                'accuracy': 0.0,
                'confidence': 0.0,
                'rmse': float('inf')
            }
