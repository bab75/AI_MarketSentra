import json
import os
import streamlit as st
from typing import Dict, Any, Optional
import yaml

class ConfigManager:
    """Manage configuration for all ML models and application settings"""
    
    def __init__(self, config_file='config/model_configs.json'):
        self.config_file = config_file
        self.config_dir = os.path.dirname(config_file)
        self.default_configs = self._load_default_configs()
        self.user_configs = self._load_user_configs()
        
        # Ensure config directory exists
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
    
    def _load_default_configs(self):
        """Load default configurations for all model categories"""
        return {
            'classical_ml': {
                'Linear Regression': {
                    'fit_intercept': True,
                    'normalize': False,
                    'n_jobs': None,
                    'positive': False
                },
                'Ridge Regression': {
                    'alpha': 1.0,
                    'fit_intercept': True,
                    'normalize': False,
                    'max_iter': 1000,
                    'tol': 0.001,
                    'solver': 'auto'
                },
                'Lasso Regression': {
                    'alpha': 1.0,
                    'fit_intercept': True,
                    'normalize': False,
                    'precompute': False,
                    'max_iter': 1000,
                    'tol': 0.0001,
                    'warm_start': False,
                    'positive': False,
                    'selection': 'cyclic'
                },
                'Elastic Net': {
                    'alpha': 1.0,
                    'l1_ratio': 0.5,
                    'fit_intercept': True,
                    'normalize': False,
                    'precompute': False,
                    'max_iter': 1000,
                    'tol': 0.0001,
                    'warm_start': False,
                    'positive': False,
                    'selection': 'cyclic'
                },
                'Decision Tree': {
                    'criterion': 'squared_error',
                    'splitter': 'best',
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'min_weight_fraction_leaf': 0.0,
                    'max_features': None,
                    'random_state': 42,
                    'max_leaf_nodes': None,
                    'min_impurity_decrease': 0.0
                },
                'Support Vector Regression': {
                    'kernel': 'rbf',
                    'degree': 3,
                    'gamma': 'scale',
                    'coef0': 0.0,
                    'tol': 0.001,
                    'C': 1.0,
                    'epsilon': 0.1,
                    'shrinking': True,
                    'cache_size': 200,
                    'verbose': False,
                    'max_iter': -1
                },
                'K-Nearest Neighbors': {
                    'n_neighbors': 5,
                    'weights': 'uniform',
                    'algorithm': 'auto',
                    'leaf_size': 30,
                    'p': 2,
                    'metric': 'minkowski',
                    'metric_params': None,
                    'n_jobs': None
                },
                'Gaussian Naive Bayes': {
                    'priors': None,
                    'var_smoothing': 1e-9
                }
            },
            'ensemble_methods': {
                'Random Forest': {
                    'n_estimators': 100,
                    'criterion': 'squared_error',
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'min_weight_fraction_leaf': 0.0,
                    'max_features': 'sqrt',
                    'max_leaf_nodes': None,
                    'min_impurity_decrease': 0.0,
                    'bootstrap': True,
                    'oob_score': False,
                    'n_jobs': None,
                    'random_state': 42,
                    'verbose': 0,
                    'warm_start': False,
                    'ccp_alpha': 0.0,
                    'max_samples': None
                },
                'Gradient Boosting': {
                    'loss': 'squared_error',
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 1.0,
                    'criterion': 'friedman_mse',
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'min_weight_fraction_leaf': 0.0,
                    'max_depth': 3,
                    'min_impurity_decrease': 0.0,
                    'init': None,
                    'random_state': 42,
                    'max_features': None,
                    'alpha': 0.9,
                    'verbose': 0,
                    'max_leaf_nodes': None,
                    'warm_start': False,
                    'validation_fraction': 0.1,
                    'n_iter_no_change': None,
                    'tol': 0.0001,
                    'ccp_alpha': 0.0
                },
                'XGBoost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'learning_rate': 0.1,
                    'subsample': 1.0,
                    'colsample_bytree': 1.0,
                    'colsample_bylevel': 1.0,
                    'colsample_bynode': 1.0,
                    'reg_alpha': 0,
                    'reg_lambda': 1,
                    'scale_pos_weight': 1,
                    'base_score': 0.5,
                    'random_state': 42,
                    'missing': None,
                    'num_parallel_tree': 1,
                    'monotone_constraints': None,
                    'interaction_constraints': None,
                    'importance_type': 'gain',
                    'gpu_id': -1,
                    'validate_parameters': 1,
                    'predictor': 'auto',
                    'enable_categorical': False
                },
                'LightGBM': {
                    'boosting_type': 'gbdt',
                    'objective': None,
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'feature_fraction': 1.0,
                    'bagging_fraction': 1.0,
                    'bagging_freq': 0,
                    'min_child_samples': 20,
                    'min_child_weight': 0.001,
                    'min_split_gain': 0.0,
                    'reg_alpha': 0.0,
                    'reg_lambda': 0.0,
                    'random_state': 42,
                    'n_estimators': 100,
                    'subsample': 1.0,
                    'subsample_freq': 0,
                    'colsample_bytree': 1.0,
                    'importance_type': 'split',
                    'n_jobs': -1,
                    'verbose': -1
                },
                'CatBoost': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'depth': 6,
                    'l2_leaf_reg': 3.0,
                    'model_size_reg': None,
                    'rsm': None,
                    'loss_function': 'RMSE',
                    'border_count': None,
                    'feature_border_type': None,
                    'per_float_feature_quantization': None,
                    'input_borders': None,
                    'output_borders': None,
                    'fold_permutation_block': None,
                    'od_pval': None,
                    'od_wait': None,
                    'od_type': None,
                    'nan_mode': None,
                    'counter_calc_method': None,
                    'leaf_estimation_iterations': None,
                    'leaf_estimation_method': None,
                    'thread_count': None,
                    'random_seed': 42,
                    'use_best_model': None,
                    'verbose': False,
                    'silent': None,
                    'logging_level': None,
                    'metric_period': None,
                    'ctr_leaf_count_limit': None,
                    'store_all_simple_ctr': None,
                    'max_ctr_complexity': None,
                    'has_time': None,
                    'allow_const_label': None,
                    'target_border': None,
                    'classes_count': None,
                    'class_weights': None,
                    'auto_class_weights': None,
                    'class_names': None,
                    'one_hot_max_size': None,
                    'random_strength': None,
                    'name': None,
                    'ignored_features': None,
                    'train_dir': None,
                    'custom_loss': None,
                    'custom_metric': None,
                    'eval_metric': None,
                    'bagging_temperature': None,
                    'save_snapshot': None,
                    'snapshot_file': None,
                    'snapshot_interval': None,
                    'fold_len_multiplier': None,
                    'used_ram_limit': None,
                    'gpu_ram_part': None,
                    'pinned_memory_size': None,
                    'allow_writing_files': None,
                    'final_ctr_computation_mode': None,
                    'approx_on_full_history': None,
                    'boosting_type': None,
                    'simple_ctr': None,
                    'combinations_ctr': None,
                    'per_feature_ctr': None,
                    'ctr_description': None,
                    'ctr_target_border_count': None,
                    'task_type': None,
                    'device_config': None,
                    'devices': None,
                    'bootstrap_type': 'Bayesian',
                    'subsample': None,
                    'sampling_unit': None,
                    'dev_score_calc_obj_block_size': None,
                    'max_depth': None,
                    'grow_policy': None,
                    'min_data_in_leaf': None,
                    'max_leaves': None,
                    'ignored_features': None,
                    'num_boost_round': None,
                    'feature_weights': None,
                    'first_feature_use_penalties': None,
                    'penalties_coefficient': None,
                    'per_object_feature_penalties': None,
                    'model_shrink_rate': None,
                    'model_shrink_mode': None,
                    'langevin': None,
                    'diffusion_temperature': None,
                    'posterior_sampling': None,
                    'boost_from_average': None
                }
            },
            'deep_learning': {
                'LSTM': {
                    'units': 50,
                    'dropout_rate': 0.2,
                    'recurrent_dropout': 0.0,
                    'activation': 'tanh',
                    'recurrent_activation': 'sigmoid',
                    'use_bias': True,
                    'kernel_initializer': 'glorot_uniform',
                    'recurrent_initializer': 'orthogonal',
                    'bias_initializer': 'zeros',
                    'unit_forget_bias': True,
                    'kernel_regularizer': None,
                    'recurrent_regularizer': None,
                    'bias_regularizer': None,
                    'activity_regularizer': None,
                    'kernel_constraint': None,
                    'recurrent_constraint': None,
                    'bias_constraint': None,
                    'return_sequences': True,
                    'return_state': False,
                    'go_backwards': False,
                    'stateful': False,
                    'time_major': False,
                    'unroll': False,
                    'sequence_length': 60,
                    'epochs': 50,
                    'batch_size': 32,
                    'validation_split': 0.2,
                    'optimizer': 'adam',
                    'loss': 'mean_squared_error',
                    'metrics': ['mae']
                },
                'GRU': {
                    'units': 50,
                    'activation': 'tanh',
                    'recurrent_activation': 'sigmoid',
                    'use_bias': True,
                    'kernel_initializer': 'glorot_uniform',
                    'recurrent_initializer': 'orthogonal',
                    'bias_initializer': 'zeros',
                    'kernel_regularizer': None,
                    'recurrent_regularizer': None,
                    'bias_regularizer': None,
                    'activity_regularizer': None,
                    'kernel_constraint': None,
                    'recurrent_constraint': None,
                    'bias_constraint': None,
                    'dropout': 0.0,
                    'recurrent_dropout': 0.0,
                    'return_sequences': True,
                    'return_state': False,
                    'go_backwards': False,
                    'stateful': False,
                    'unroll': False,
                    'time_major': False,
                    'reset_after': True,
                    'sequence_length': 60,
                    'epochs': 50,
                    'batch_size': 32,
                    'validation_split': 0.2,
                    'optimizer': 'adam',
                    'loss': 'mean_squared_error',
                    'metrics': ['mae']
                },
                'CNN-LSTM': {
                    'filters': 64,
                    'kernel_size': 3,
                    'lstm_units': 50,
                    'dropout_rate': 0.2,
                    'pool_size': 2,
                    'activation': 'relu',
                    'padding': 'valid',
                    'sequence_length': 60,
                    'epochs': 50,
                    'batch_size': 32,
                    'validation_split': 0.2,
                    'optimizer': 'adam',
                    'loss': 'mean_squared_error',
                    'metrics': ['mae']
                },
                'Transformer': {
                    'd_model': 64,
                    'num_heads': 4,
                    'num_layers': 2,
                    'dff': 256,
                    'dropout_rate': 0.1,
                    'sequence_length': 60,
                    'epochs': 50,
                    'batch_size': 32,
                    'validation_split': 0.2,
                    'optimizer': 'adam',
                    'loss': 'mean_squared_error',
                    'metrics': ['mae'],
                    'learning_rate': 0.001,
                    'beta_1': 0.9,
                    'beta_2': 0.999,
                    'epsilon': 1e-07,
                    'amsgrad': False
                }
            },
            'time_series': {
                'ARIMA': {
                    'order': [1, 1, 1],
                    'seasonal_order': None,
                    'trend': None,
                    'method': 'lbfgs',
                    'maxiter': 50,
                    'suppress_warnings': True,
                    'out_of_sample_size': 0,
                    'scoring': 'mse',
                    'scoring_args': None,
                    'target_col': 'Close'
                },
                'SARIMA': {
                    'order': [1, 1, 1],
                    'seasonal_order': [1, 1, 1, 12],
                    'trend': None,
                    'measurement_error': False,
                    'time_varying_regression': False,
                    'mle_regression': True,
                    'simple_differencing': False,
                    'enforce_stationarity': True,
                    'enforce_invertibility': True,
                    'hamilton_representation': False,
                    'concentrate_scale': False,
                    'trend_offset': 1,
                    'use_exact_diffuse': False,
                    'dates': None,
                    'freq': None,
                    'missing': 'none',
                    'target_col': 'Close'
                },
                'Auto ARIMA': {
                    'start_p': 0,
                    'start_q': 0,
                    'test': 'adf',
                    'max_p': 5,
                    'max_q': 5,
                    'd': None,
                    'seasonal': True,
                    'start_P': 0,
                    'start_Q': 0,
                    'max_P': 2,
                    'max_Q': 2,
                    'max_D': 1,
                    'max_order': 5,
                    'm': 12,
                    'n_jobs': 1,
                    'start_params': None,
                    'method': None,
                    'trend': None,
                    'solver': 'lbfgs',
                    'maxiter': 50,
                    'offset_test_args': None,
                    'seasonal_test_args': None,
                    'stepwise': True,
                    'suppress_warnings': True,
                    'error_action': 'trace',
                    'trace': False,
                    'random': False,
                    'random_state': None,
                    'n_fits': 10,
                    'out_of_sample_size': 0,
                    'scoring': 'mse',
                    'scoring_args': None,
                    'with_intercept': True,
                    'update_pdq': True,
                    'time_varying_regression': False,
                    'enforce_stationarity': True,
                    'enforce_invertibility': True,
                    'simple_differencing': False,
                    'measurement_error': False,
                    'mle_regression': True,
                    'hamilton_representation': False,
                    'concentrate_scale': False,
                    'target_col': 'Close'
                },
                'Hidden Markov Model': {
                    'n_components': 3,
                    'covariance_type': 'full',
                    'min_covar': 0.001,
                    'startprob_prior': 1.0,
                    'transmat_prior': 1.0,
                    'means_prior': None,
                    'means_weight': 0,
                    'covars_prior': None,
                    'covars_weight': None,
                    'algorithm': 'viterbi',
                    'random_state': 42,
                    'n_iter': 10,
                    'tol': 0.01,
                    'verbose': False,
                    'params': 'stmc',
                    'init_params': 'stmc',
                    'target_col': 'Close'
                }
            },
            'clustering': {
                'K-Means': {
                    'n_clusters': 3,
                    'init': 'k-means++',
                    'n_init': 10,
                    'max_iter': 300,
                    'tol': 0.0001,
                    'verbose': 0,
                    'random_state': 42,
                    'copy_x': True,
                    'algorithm': 'lloyd'
                },
                'DBSCAN': {
                    'eps': 0.5,
                    'min_samples': 5,
                    'metric': 'euclidean',
                    'metric_params': None,
                    'algorithm': 'auto',
                    'leaf_size': 30,
                    'p': None,
                    'n_jobs': None
                },
                'Gaussian Mixture': {
                    'n_components': 3,
                    'covariance_type': 'full',
                    'tol': 0.001,
                    'reg_covar': 1e-06,
                    'max_iter': 100,
                    'n_init': 1,
                    'init_params': 'kmeans',
                    'weights_init': None,
                    'means_init': None,
                    'precisions_init': None,
                    'random_state': 42,
                    'warm_start': False,
                    'verbose': 0,
                    'verbose_interval': 10
                }
            },
            'anomaly_detection': {
                'Isolation Forest': {
                    'n_estimators': 100,
                    'max_samples': 'auto',
                    'contamination': 0.1,
                    'max_features': 1.0,
                    'bootstrap': False,
                    'n_jobs': None,
                    'random_state': 42,
                    'verbose': 0,
                    'warm_start': False
                },
                'One-Class SVM': {
                    'kernel': 'rbf',
                    'degree': 3,
                    'gamma': 'scale',
                    'coef0': 0.0,
                    'tol': 0.001,
                    'nu': 0.1,
                    'shrinking': True,
                    'cache_size': 200,
                    'verbose': False,
                    'max_iter': -1
                },
                'Local Outlier Factor': {
                    'n_neighbors': 20,
                    'algorithm': 'auto',
                    'leaf_size': 30,
                    'metric': 'minkowski',
                    'p': 2,
                    'metric_params': None,
                    'contamination': 0.1,
                    'novelty': False,
                    'n_jobs': None
                },
                'Statistical Z-Score': {
                    'threshold': 3.0,
                    'method': 'standard'
                },
                'Modified Z-Score': {
                    'threshold': 3.5,
                    'method': 'modified'
                },
                'Interquartile Range': {
                    'factor': 1.5,
                    'method': 'iqr'
                }
            },
            'reinforcement_learning': {
                'PPO (Proximal Policy Optimization)': {
                    'learning_rate': 0.0003,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_range': 0.2,
                    'clip_range_vf': None,
                    'normalize_advantage': True,
                    'ent_coef': 0.0,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5,
                    'use_sde': False,
                    'sde_sample_freq': -1,
                    'target_kl': None,
                    'tensorboard_log': None,
                    'policy_kwargs': None,
                    'verbose': 0,
                    'seed': None,
                    'device': 'auto',
                    '_init_setup_model': True
                },
                'DQN (Deep Q-Network)': {
                    'learning_rate': 0.0001,
                    'buffer_size': 1000000,
                    'learning_starts': 50000,
                    'batch_size': 32,
                    'tau': 1.0,
                    'gamma': 0.99,
                    'train_freq': 4,
                    'gradient_steps': 1,
                    'replay_buffer_class': None,
                    'replay_buffer_kwargs': None,
                    'optimize_memory_usage': False,
                    'target_update_interval': 10000,
                    'exploration_fraction': 0.1,
                    'exploration_initial_eps': 1.0,
                    'exploration_final_eps': 0.05,
                    'max_grad_norm': 10,
                    'tensorboard_log': None,
                    'policy_kwargs': None,
                    'verbose': 0,
                    'seed': None,
                    'device': 'auto',
                    '_init_setup_model': True
                },
                'Q-Learning': {
                    'learning_rate': 0.1,
                    'discount_factor': 0.95,
                    'epsilon': 0.1,
                    'epsilon_min': 0.01,
                    'epsilon_decay': 0.995,
                    'state_bins': 10,
                    'episodes': 1000
                }
            },
            'application_settings': {
                'data_processing': {
                    'default_lookback_period': 30,
                    'min_data_points': 10,
                    'max_missing_ratio': 0.1,
                    'outlier_detection_threshold': 3.0,
                    'volatility_window': 20,
                    'ma_windows': [5, 10, 20, 50],
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'macd_signal': 9,
                    'bollinger_period': 20,
                    'bollinger_std': 2
                },
                'visualization': {
                    'default_chart_height': 500,
                    'candlestick_colors': {
                        'increasing': 'green',
                        'decreasing': 'red'
                    },
                    'color_palette': [
                        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                        '#bcbd22', '#17becf'
                    ],
                    'plot_template': 'plotly_white'
                },
                'model_training': {
                    'default_test_size': 0.2,
                    'default_validation_split': 0.2,
                    'random_state': 42,
                    'cross_validation_folds': 5,
                    'hyperparameter_tuning': {
                        'max_evals': 50,
                        'cv_folds': 3,
                        'scoring': 'r2'
                    }
                },
                'performance_metrics': {
                    'price_tolerance': 0.05,  # 5% tolerance for accuracy calculation
                    'confidence_threshold': 0.7,
                    'min_accuracy_threshold': 0.6,
                    'sharpe_ratio_risk_free_rate': 0.02,
                    'max_drawdown_threshold': 0.2,
                    'volatility_threshold': 0.3
                }
            }
        }
    
    def _load_user_configs(self):
        """Load user-defined configurations"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            st.warning(f"Error loading user configs: {str(e)}")
            return {}
    
    def save_user_configs(self):
        """Save user configurations to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.user_configs, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving user configs: {str(e)}")
            return False
    
    def get_model_config(self, category: str, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model
        
        Args:
            category (str): Model category
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration
        """
        # Normalize category name
        category_key = category.lower().replace(' ', '_').replace('&', 'and')
        
        # Check user configs first
        user_config = self.user_configs.get(category_key, {}).get(model_name, {})
        
        # Get default config
        default_config = self.default_configs.get(category_key, {}).get(model_name, {})
        
        # Merge configs (user overrides default)
        merged_config = default_config.copy()
        merged_config.update(user_config)
        
        return merged_config
    
    def update_model_config(self, category: str, model_name: str, config: Dict[str, Any]) -> bool:
        """
        Update configuration for a specific model
        
        Args:
            category (str): Model category
            model_name (str): Name of the model
            config (dict): New configuration
            
        Returns:
            bool: Success status
        """
        try:
            category_key = category.lower().replace(' ', '_').replace('&', 'and')
            
            if category_key not in self.user_configs:
                self.user_configs[category_key] = {}
            
            self.user_configs[category_key][model_name] = config
            
            return self.save_user_configs()
            
        except Exception as e:
            st.error(f"Error updating model config: {str(e)}")
            return False
    
    def reset_model_config(self, category: str, model_name: str) -> bool:
        """
        Reset model configuration to default
        
        Args:
            category (str): Model category
            model_name (str): Name of the model
            
        Returns:
            bool: Success status
        """
        try:
            category_key = category.lower().replace(' ', '_').replace('&', 'and')
            
            if category_key in self.user_configs and model_name in self.user_configs[category_key]:
                del self.user_configs[category_key][model_name]
                
                # Clean up empty categories
                if not self.user_configs[category_key]:
                    del self.user_configs[category_key]
                
                return self.save_user_configs()
            
            return True
            
        except Exception as e:
            st.error(f"Error resetting model config: {str(e)}")
            return False
    
    def get_application_setting(self, setting_path: str, default_value: Any = None) -> Any:
        """
        Get application setting using dot notation
        
        Args:
            setting_path (str): Path to setting (e.g., 'data_processing.default_lookback_period')
            default_value: Default value if setting not found
            
        Returns:
            Any: Setting value
        """
        try:
            keys = setting_path.split('.')
            
            # Check user configs first
            value = self.user_configs.get('application_settings', {})
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            
            if value is not None:
                return value
            
            # Check default configs
            value = self.default_configs.get('application_settings', {})
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default_value
            
            return value
            
        except Exception as e:
            st.error(f"Error getting application setting: {str(e)}")
            return default_value
    
    def update_application_setting(self, setting_path: str, value: Any) -> bool:
        """
        Update application setting using dot notation
        
        Args:
            setting_path (str): Path to setting
            value: New value
            
        Returns:
            bool: Success status
        """
        try:
            keys = setting_path.split('.')
            
            if 'application_settings' not in self.user_configs:
                self.user_configs['application_settings'] = {}
            
            # Navigate to the correct nested dict
            current = self.user_configs['application_settings']
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Set the value
            current[keys[-1]] = value
            
            return self.save_user_configs()
            
        except Exception as e:
            st.error(f"Error updating application setting: {str(e)}")
            return False
    
    def validate_config(self, category: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate model configuration
        
        Args:
            category (str): Model category
            model_name (str): Name of the model
            config (dict): Configuration to validate
            
        Returns:
            dict: Validation results with 'valid' boolean and 'errors' list
        """
        validation_result = {'valid': True, 'errors': []}
        
        try:
            # Get default config for comparison
            default_config = self.get_model_config(category, model_name)
            
            # Basic validation rules
            for key, value in config.items():
                if key in default_config:
                    default_value = default_config[key]
                    
                    # Type validation
                    if default_value is not None and not isinstance(value, type(default_value)):
                        if not (isinstance(default_value, (int, float)) and isinstance(value, (int, float))):
                            validation_result['errors'].append(
                                f"Parameter '{key}': Expected {type(default_value).__name__}, got {type(value).__name__}"
                            )
                    
                    # Value range validation for common parameters
                    if key in ['learning_rate', 'alpha', 'reg_alpha', 'reg_lambda'] and isinstance(value, (int, float)):
                        if value < 0:
                            validation_result['errors'].append(f"Parameter '{key}' must be non-negative")
                    
                    if key in ['n_estimators', 'max_iter', 'epochs'] and isinstance(value, int):
                        if value <= 0:
                            validation_result['errors'].append(f"Parameter '{key}' must be positive")
                    
                    if key in ['contamination', 'test_size', 'validation_split'] and isinstance(value, (int, float)):
                        if not (0 < value < 1):
                            validation_result['errors'].append(f"Parameter '{key}' must be between 0 and 1")
                    
                    if key in ['n_clusters', 'n_components'] and isinstance(value, int):
                        if value < 1:
                            validation_result['errors'].append(f"Parameter '{key}' must be at least 1")
            
            if validation_result['errors']:
                validation_result['valid'] = False
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def export_configs(self, format: str = 'json') -> Optional[str]:
        """
        Export all configurations
        
        Args:
            format (str): Export format ('json', 'yaml')
            
        Returns:
            str: Exported configuration string
        """
        try:
            all_configs = {
                'default_configs': self.default_configs,
                'user_configs': self.user_configs
            }
            
            if format.lower() == 'json':
                return json.dumps(all_configs, indent=2)
            elif format.lower() == 'yaml':
                return yaml.dump(all_configs, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            st.error(f"Error exporting configs: {str(e)}")
            return None
    
    def import_configs(self, config_string: str, format: str = 'json') -> bool:
        """
        Import configurations from string
        
        Args:
            config_string (str): Configuration string
            format (str): Import format ('json', 'yaml')
            
        Returns:
            bool: Success status
        """
        try:
            if format.lower() == 'json':
                imported_configs = json.loads(config_string)
            elif format.lower() == 'yaml':
                imported_configs = yaml.safe_load(config_string)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Validate structure
            if 'user_configs' in imported_configs:
                self.user_configs = imported_configs['user_configs']
                return self.save_user_configs()
            
            return False
            
        except Exception as e:
            st.error(f"Error importing configs: {str(e)}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configurations
        
        Returns:
            dict: Configuration summary
        """
        summary = {
            'total_categories': len(self.default_configs),
            'categories': {},
            'user_overrides': 0,
            'application_settings': {}
        }
        
        for category, models in self.default_configs.items():
            if category == 'application_settings':
                summary['application_settings'] = models
                continue
                
            summary['categories'][category] = {
                'total_models': len(models),
                'models': list(models.keys()),
                'user_overrides': 0
            }
            
            # Count user overrides
            category_key = category
            if category_key in self.user_configs:
                summary['categories'][category]['user_overrides'] = len(self.user_configs[category_key])
                summary['user_overrides'] += len(self.user_configs[category_key])
        
        return summary
    
    def create_config_backup(self) -> bool:
        """
        Create a backup of current user configurations
        
        Returns:
            bool: Success status
        """
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.config_dir}/config_backup_{timestamp}.json"
            
            with open(backup_file, 'w') as f:
                json.dump(self.user_configs, f, indent=2)
            
            st.success(f"Configuration backup created: {backup_file}")
            return True
            
        except Exception as e:
            st.error(f"Error creating config backup: {str(e)}")
            return False
