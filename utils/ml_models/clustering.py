import numpy as np
import pandas as pd
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, 
    SpectralClustering, MeanShift, Birch
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import streamlit as st

class ClusteringAndDimensionalityReduction:
    """Clustering and Dimensionality Reduction Models"""
    
    def __init__(self):
        self.clustering_models = {
            'K-Means': KMeans(n_clusters=3, random_state=42),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
            'Agglomerative Clustering': AgglomerativeClustering(n_clusters=3),
            'Spectral Clustering': SpectralClustering(n_clusters=3, random_state=42),
            'Mean Shift': MeanShift(),
            'Birch': Birch(n_clusters=3),
            'Gaussian Mixture': GaussianMixture(n_components=3, random_state=42)
        }
        
        self.dimensionality_models = {
            'PCA': PCA(n_components=0.95),
            'Independent Component Analysis': FastICA(n_components=None, random_state=42),
            'Non-negative Matrix Factorization': NMF(n_components=10, random_state=42),
            't-SNE': TSNE(n_components=2, random_state=42),
            'Isomap': Isomap(n_components=2)
        }
        
        self.trained_models = {}
        self.scaler = StandardScaler()
        self.model_performances = {}
    
    def get_available_models(self):
        """Return list of available clustering and dimensionality reduction models"""
        clustering_models = list(self.clustering_models.keys())
        dimensionality_models = list(self.dimensionality_models.keys())
        return {
            'Clustering': clustering_models,
            'Dimensionality Reduction': dimensionality_models
        }
    
    def prepare_features(self, data):
        """
        Prepare features for clustering and dimensionality reduction
        
        Args:
            data (DataFrame): Stock data
            
        Returns:
            array: Scaled feature array
        """
        try:
            # Select relevant features for clustering
            feature_columns = []
            
            # Price-based features
            if 'Open' in data.columns:
                feature_columns.extend(['Open', 'High', 'Low', 'Close'])
            
            # Volume feature
            if 'Volume' in data.columns:
                feature_columns.append('Volume')
            
            # Technical indicators if available
            technical_indicators = [
                'Daily_Return', 'MA_5', 'MA_10', 'MA_20', 'MA_50',
                'Volatility', 'RSI', 'MACD', 'MACD_Signal'
            ]
            
            for indicator in technical_indicators:
                if indicator in data.columns:
                    feature_columns.append(indicator)
            
            # Extract features and handle missing values
            features = data[feature_columns].dropna()
            
            if len(features) == 0:
                raise ValueError("No valid features found after removing NaN values")
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features, features.index, feature_columns
            
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return None, None, None
    
    def train_clustering_model(self, model_name, data, n_clusters=None):
        """
        Train a clustering model
        
        Args:
            model_name (str): Name of the clustering model
            data (DataFrame): Stock data
            n_clusters (int): Number of clusters (if applicable)
            
        Returns:
            dict: Clustering results
        """
        try:
            if model_name not in self.clustering_models:
                raise ValueError(f"Clustering model {model_name} not found")
            
            # Prepare features
            scaled_features, feature_index, feature_columns = self.prepare_features(data)
            
            if scaled_features is None:
                return None
            
            # Get model and update parameters if needed
            model = self.clustering_models[model_name]
            
            if n_clusters and hasattr(model, 'n_clusters'):
                model.set_params(n_clusters=n_clusters)
            elif n_clusters and hasattr(model, 'n_components'):
                model.set_params(n_components=n_clusters)
            
            # Fit model
            if model_name == 'Gaussian Mixture':
                labels = model.fit_predict(scaled_features)
                cluster_centers = model.means_
            else:
                labels = model.fit_predict(scaled_features)
                cluster_centers = getattr(model, 'cluster_centers_', None)
            
            # Calculate clustering metrics
            unique_labels = np.unique(labels)
            n_clusters_found = len(unique_labels)
            
            metrics = {}
            if n_clusters_found > 1 and -1 not in unique_labels:  # Valid clustering
                metrics['silhouette_score'] = silhouette_score(scaled_features, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(scaled_features, labels)
            
            # Create results DataFrame
            results_df = pd.DataFrame(index=feature_index)
            results_df['Cluster'] = labels
            
            # Add original features for analysis
            original_features = data.loc[feature_index, feature_columns]
            results_df = pd.concat([results_df, original_features], axis=1)
            
            results = {
                'model': model,
                'labels': labels,
                'cluster_centers': cluster_centers,
                'results_df': results_df,
                'metrics': metrics,
                'n_clusters': n_clusters_found,
                'feature_columns': feature_columns,
                'scaled_features': scaled_features
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error training clustering model {model_name}: {str(e)}")
            return None
    
    def train_dimensionality_reduction(self, model_name, data, n_components=None):
        """
        Train a dimensionality reduction model
        
        Args:
            model_name (str): Name of the dimensionality reduction model
            data (DataFrame): Stock data
            n_components (int): Number of components
            
        Returns:
            dict: Dimensionality reduction results
        """
        try:
            if model_name not in self.dimensionality_models:
                raise ValueError(f"Dimensionality reduction model {model_name} not found")
            
            # Prepare features
            scaled_features, feature_index, feature_columns = self.prepare_features(data)
            
            if scaled_features is None:
                return None
            
            # Get model and update parameters if needed
            model = self.dimensionality_models[model_name]
            
            if n_components and hasattr(model, 'n_components'):
                if model_name == 'PCA' and n_components < 1:
                    # Keep as ratio for PCA
                    model.set_params(n_components=n_components)
                else:
                    model.set_params(n_components=min(n_components, scaled_features.shape[1]))
            
            # Fit and transform
            if model_name in ['t-SNE', 'Isomap']:
                # These models don't have separate fit and transform
                transformed_features = model.fit_transform(scaled_features)
                components = None
                explained_variance = None
            else:
                transformed_features = model.fit_transform(scaled_features)
                components = getattr(model, 'components_', None)
                explained_variance = getattr(model, 'explained_variance_ratio_', None)
            
            # Create results DataFrame
            n_components_actual = transformed_features.shape[1]
            component_columns = [f'Component_{i+1}' for i in range(n_components_actual)]
            
            results_df = pd.DataFrame(
                transformed_features,
                index=feature_index,
                columns=component_columns
            )
            
            results = {
                'model': model,
                'transformed_features': transformed_features,
                'results_df': results_df,
                'components': components,
                'explained_variance': explained_variance,
                'feature_columns': feature_columns,
                'n_components': n_components_actual
            }
            
            # Add specific metrics for PCA
            if model_name == 'PCA' and explained_variance is not None:
                results['cumulative_variance'] = np.cumsum(explained_variance)
                results['total_variance_explained'] = np.sum(explained_variance)
            
            return results
            
        except Exception as e:
            st.error(f"Error training dimensionality reduction model {model_name}: {str(e)}")
            return None
    
    def optimal_clusters_elbow(self, data, max_clusters=10):
        """
        Find optimal number of clusters using elbow method
        
        Args:
            data (DataFrame): Stock data
            max_clusters (int): Maximum number of clusters to try
            
        Returns:
            dict: Elbow analysis results
        """
        try:
            scaled_features, _, _ = self.prepare_features(data)
            
            if scaled_features is None:
                return None
            
            inertias = []
            k_range = range(1, min(max_clusters + 1, len(scaled_features)))
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_features)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow detection
            differences = np.diff(inertias)
            second_differences = np.diff(differences)
            
            # Find the point with maximum second difference (elbow)
            if len(second_differences) > 0:
                optimal_k = np.argmax(second_differences) + 2  # +2 due to double differencing
            else:
                optimal_k = 3  # Default
            
            return {
                'k_range': list(k_range),
                'inertias': inertias,
                'optimal_k': optimal_k,
                'differences': differences.tolist(),
                'second_differences': second_differences.tolist()
            }
            
        except Exception as e:
            st.error(f"Error in elbow analysis: {str(e)}")
            return None
    
    def silhouette_analysis(self, data, k_range=None):
        """
        Perform silhouette analysis for different number of clusters
        
        Args:
            data (DataFrame): Stock data
            k_range (list): Range of k values to analyze
            
        Returns:
            dict: Silhouette analysis results
        """
        try:
            scaled_features, _, _ = self.prepare_features(data)
            
            if scaled_features is None:
                return None
            
            if k_range is None:
                k_range = range(2, min(11, len(scaled_features)))
            
            silhouette_scores = []
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(scaled_features)
                score = silhouette_score(scaled_features, labels)
                silhouette_scores.append(score)
            
            # Find optimal k
            optimal_k = k_range[np.argmax(silhouette_scores)]
            
            return {
                'k_range': list(k_range),
                'silhouette_scores': silhouette_scores,
                'optimal_k': optimal_k,
                'max_score': max(silhouette_scores)
            }
            
        except Exception as e:
            st.error(f"Error in silhouette analysis: {str(e)}")
            return None
    
    def analyze_clusters(self, clustering_results, data):
        """
        Analyze clustering results to provide insights
        
        Args:
            clustering_results (dict): Results from clustering
            data (DataFrame): Original data
            
        Returns:
            dict: Cluster analysis insights
        """
        try:
            if clustering_results is None:
                return None
            
            results_df = clustering_results['results_df']
            labels = clustering_results['labels']
            
            cluster_analysis = {}
            
            for cluster_id in np.unique(labels):
                if cluster_id == -1:  # Noise points in DBSCAN
                    continue
                
                cluster_data = results_df[results_df['Cluster'] == cluster_id]
                
                # Calculate cluster statistics
                cluster_stats = {}
                for col in clustering_results['feature_columns']:
                    if col in cluster_data.columns:
                        cluster_stats[col] = {
                            'mean': cluster_data[col].mean(),
                            'std': cluster_data[col].std(),
                            'min': cluster_data[col].min(),
                            'max': cluster_data[col].max()
                        }
                
                cluster_analysis[f'Cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(results_df) * 100,
                    'statistics': cluster_stats,
                    'dates': cluster_data.index.tolist()
                }
            
            return cluster_analysis
            
        except Exception as e:
            st.error(f"Error analyzing clusters: {str(e)}")
            return None
    
    def get_model_config(self, model_name, model_type='clustering'):
        """
        Get configuration parameters for clustering/dimensionality reduction models
        
        Args:
            model_name (str): Name of the model
            model_type (str): 'clustering' or 'dimensionality_reduction'
            
        Returns:
            dict: Model configuration parameters
        """
        if model_type == 'clustering':
            configs = {
                'K-Means': {'n_clusters': 3, 'max_iter': 300, 'random_state': 42},
                'DBSCAN': {'eps': 0.5, 'min_samples': 5},
                'Agglomerative Clustering': {'n_clusters': 3, 'linkage': 'ward'},
                'Spectral Clustering': {'n_clusters': 3, 'random_state': 42},
                'Mean Shift': {'bandwidth': None},
                'Birch': {'n_clusters': 3, 'threshold': 0.5},
                'Gaussian Mixture': {'n_components': 3, 'covariance_type': 'full', 'random_state': 42}
            }
        else:
            configs = {
                'PCA': {'n_components': 0.95, 'random_state': None},
                'Independent Component Analysis': {'n_components': None, 'random_state': 42},
                'Non-negative Matrix Factorization': {'n_components': 10, 'random_state': 42},
                't-SNE': {'n_components': 2, 'perplexity': 30, 'random_state': 42},
                'Isomap': {'n_components': 2, 'n_neighbors': 5}
            }
        
        return configs.get(model_name, {})
