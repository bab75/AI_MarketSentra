import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st

class ClusteringAndDimensionalityReduction:
    """Clustering and Dimensionality Reduction Models (Safe Version)"""
    
    def __init__(self):
        self.clustering_models = {
            'K-Means': KMeans,
            'DBSCAN': DBSCAN
        }
        
        self.dimensionality_models = {
            'PCA': PCA
        }
        
        self.trained_models = {}
        self.model_performances = {}
        self.scalers = {}
    
    def get_available_models(self):
        """Return available clustering and dimensionality reduction models"""
        return {
            'Clustering': list(self.clustering_models.keys()),
            'Dimensionality Reduction': list(self.dimensionality_models.keys())
        }
    
    def train_clustering_model(self, model_name, data, **kwargs):
        """Train a clustering model"""
        try:
            # Prepare features
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            if model_name == 'K-Means':
                n_clusters = kwargs.get('n_clusters', 3)
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif model_name == 'DBSCAN':
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            else:
                raise ValueError(f"Unknown clustering model: {model_name}")
            
            # Fit model
            labels = model.fit_predict(scaled_features)
            
            # Calculate silhouette score if possible
            try:
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(scaled_features, labels)
                else:
                    silhouette = 0
            except:
                silhouette = 0
            
            results = {
                'model_name': model_name,
                'n_clusters': len(set(labels)),
                'silhouette_score': silhouette,
                'labels': labels.tolist(),
                'cluster_centers': model.cluster_centers_.tolist() if hasattr(model, 'cluster_centers_') else []
            }
            
            self.trained_models[model_name] = model
            self.scalers[model_name] = scaler
            self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training clustering model {model_name}: {str(e)}")
            return None
    
    def train_dimensionality_reduction(self, model_name, data, **kwargs):
        """Train a dimensionality reduction model"""
        try:
            # Prepare features
            features = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
            
            # Scale features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            if model_name == 'PCA':
                n_components = kwargs.get('n_components', 2)
                model = PCA(n_components=n_components)
            else:
                raise ValueError(f"Unknown dimensionality reduction model: {model_name}")
            
            # Fit and transform
            transformed_data = model.fit_transform(scaled_features)
            
            # Calculate explained variance ratio
            explained_variance = model.explained_variance_ratio_
            total_variance = sum(explained_variance)
            
            results = {
                'model_name': model_name,
                'n_components': transformed_data.shape[1],
                'explained_variance_ratio': explained_variance.tolist(),
                'total_explained_variance': total_variance,
                'transformed_data': transformed_data.tolist()
            }
            
            self.trained_models[model_name] = model
            self.scalers[model_name] = scaler
            self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training dimensionality reduction model {model_name}: {str(e)}")
            return None
    
    def get_model_performance(self, model_name):
        """Get performance metrics for a specific model"""
        return self.model_performances.get(model_name, {})
    
    def get_all_performances(self):
        """Get all model performances"""
        return self.model_performances