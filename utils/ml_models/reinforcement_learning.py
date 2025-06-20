import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class TradingEnvironment(gym.Env):
    """Custom Trading Environment for Reinforcement Learning"""
    
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001, lookback_window=10):
        super(TradingEnvironment, self).__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Prepare data
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        # Environment state
        self.reset()
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [price features, portfolio state, technical indicators]
        n_features = self.scaled_data.shape[1]
        obs_size = n_features * lookback_window + 3  # +3 for balance, position, profit
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        # Trading variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # Number of shares
        self.entry_price = 0
        self.total_profit = 0
        self.trades = []
        
    def reset(self, seed=None):
        """Reset the environment"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_profit = 0
        self.trades = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation"""
        # Price features for lookback window
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        price_features = self.scaled_data[start_idx:end_idx].flatten()
        
        # Pad if necessary
        if len(price_features) < self.lookback_window * self.scaled_data.shape[1]:
            padded_features = np.zeros(self.lookback_window * self.scaled_data.shape[1])
            padded_features[-len(price_features):] = price_features
            price_features = padded_features
        
        # Portfolio state
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_value = self.balance + self.position * current_price
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position / 100,  # Normalized position
            self.total_profit / self.initial_balance  # Normalized profit
        ])
        
        observation = np.concatenate([price_features, portfolio_state])
        return observation.astype(np.float32)
    
    def step(self, action):
        """Execute one step in the environment"""
        current_price = self.data.iloc[self.current_step]['Close']
        
        # Execute action
        reward = 0
        if action == 1:  # Buy
            reward = self._buy(current_price)
        elif action == 2:  # Sell
            reward = self._sell(current_price)
        # action == 0 is Hold, no action needed
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Calculate portfolio value for info
        portfolio_value = self.balance + self.position * current_price
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'total_profit': self.total_profit,
            'current_price': current_price
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _buy(self, price):
        """Execute buy action"""
        if self.balance > price * (1 + self.transaction_cost):
            shares_to_buy = int(self.balance / (price * (1 + self.transaction_cost)))
            cost = shares_to_buy * price * (1 + self.transaction_cost)
            
            self.balance -= cost
            self.position += shares_to_buy
            self.entry_price = price
            
            self.trades.append({
                'action': 'BUY',
                'price': price,
                'shares': shares_to_buy,
                'timestamp': self.data.index[self.current_step]
            })
            
            return 0.01  # Small positive reward for buying
        return -0.01  # Penalty for invalid buy
    
    def _sell(self, price):
        """Execute sell action"""
        if self.position > 0:
            proceeds = self.position * price * (1 - self.transaction_cost)
            profit = proceeds - (self.position * self.entry_price)
            
            self.balance += proceeds
            self.total_profit += profit
            
            self.trades.append({
                'action': 'SELL',
                'price': price,
                'shares': self.position,
                'profit': profit,
                'timestamp': self.data.index[self.current_step]
            })
            
            self.position = 0
            self.entry_price = 0
            
            # Reward based on profit
            return profit / self.initial_balance
        return -0.01  # Penalty for invalid sell
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        current_price = self.data.iloc[self.current_step]['Close']
        portfolio_value = self.balance + self.position * current_price
        
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Position: {self.position}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Total Profit: ${self.total_profit:.2f}")
        print("-" * 30)

class ReinforcementLearningModels:
    """Reinforcement Learning Models for Stock Trading"""
    
    def __init__(self):
        self.models = {
            'PPO (Proximal Policy Optimization)': PPO,
            'A2C (Advantage Actor-Critic)': A2C,
            'DQN (Deep Q-Network)': DQN,
            'SAC (Soft Actor-Critic)': SAC,
            'TD3 (Twin Delayed Deep Deterministic)': TD3
        }
        
        # Q-Learning implementation
        self.q_learning_models = {
            'Q-Learning': self._create_q_learning_model,
            'Double Q-Learning': self._create_double_q_learning_model
        }
        
        self.trained_models = {}
        self.environments = {}
        self.model_performances = {}
    
    def get_available_models(self):
        """Return list of available RL models"""
        stable_baselines_models = list(self.models.keys())
        q_learning_models = list(self.q_learning_models.keys())
        return {
            'Deep RL Models': stable_baselines_models,
            'Classical RL Models': q_learning_models
        }
    
    def create_trading_environment(self, data, initial_balance=10000, transaction_cost=0.001, lookback_window=10):
        """
        Create a trading environment for RL training
        
        Args:
            data (DataFrame): Stock data
            initial_balance (float): Initial trading balance
            transaction_cost (float): Transaction cost percentage
            lookback_window (int): Number of historical steps to include in observation
            
        Returns:
            TradingEnvironment: Custom trading environment
        """
        try:
            # Prepare data with technical indicators
            prepared_data = self._prepare_rl_data(data)
            
            env = TradingEnvironment(
                data=prepared_data,
                initial_balance=initial_balance,
                transaction_cost=transaction_cost,
                lookback_window=lookback_window
            )
            
            return env
            
        except Exception as e:
            st.error(f"Error creating trading environment: {str(e)}")
            return None
    
    def _prepare_rl_data(self, data):
        """Prepare data for RL training"""
        prepared_data = data.copy()
        
        # Add technical indicators if not present
        if 'Daily_Return' not in prepared_data.columns:
            prepared_data['Daily_Return'] = prepared_data['Close'].pct_change()
        
        if 'MA_10' not in prepared_data.columns:
            prepared_data['MA_10'] = prepared_data['Close'].rolling(window=10).mean()
        
        if 'MA_30' not in prepared_data.columns:
            prepared_data['MA_30'] = prepared_data['Close'].rolling(window=30).mean()
        
        if 'Volatility' not in prepared_data.columns:
            prepared_data['Volatility'] = prepared_data['Daily_Return'].rolling(window=20).std()
        
        # Add price momentum
        prepared_data['Price_Momentum'] = prepared_data['Close'] / prepared_data['Close'].shift(5) - 1
        
        # Add volume momentum if volume is available
        if 'Volume' in prepared_data.columns:
            prepared_data['Volume_Momentum'] = prepared_data['Volume'] / prepared_data['Volume'].shift(5) - 1
        
        # Remove NaN values
        prepared_data = prepared_data.dropna()
        
        return prepared_data
    
    def train_model(self, model_name, data, total_timesteps=10000, **kwargs):
        """
        Train a reinforcement learning model
        
        Args:
            model_name (str): Name of the RL model
            data (DataFrame): Stock data
            total_timesteps (int): Total training timesteps
            **kwargs: Additional model parameters
            
        Returns:
            dict: Training results
        """
        try:
            # Create environment
            env = self.create_trading_environment(data, **kwargs)
            if env is None:
                return None
            
            # Wrap environment
            env = Monitor(env)
            env = DummyVecEnv([lambda: env])
            
            # Store environment
            self.environments[model_name] = env
            
            if model_name in self.models:
                # Stable Baselines3 models
                model_class = self.models[model_name]
                
                # Model-specific parameters
                if model_name == 'DQN':
                    model = model_class('MlpPolicy', env, verbose=0, learning_rate=0.001)
                elif model_name == 'SAC':
                    model = model_class('MlpPolicy', env, verbose=0, learning_rate=0.001)
                elif model_name == 'TD3':
                    model = model_class('MlpPolicy', env, verbose=0, learning_rate=0.001)
                else:  # PPO, A2C
                    model = model_class('MlpPolicy', env, verbose=0, learning_rate=0.001)
                
                # Train model
                model.learn(total_timesteps=total_timesteps)
                
                # Evaluate model
                evaluation_results = self._evaluate_model(model, env, n_episodes=5)
                
                results = {
                    'model': model,
                    'environment': env,
                    'evaluation': evaluation_results,
                    'total_timesteps': total_timesteps,
                    'model_type': 'deep_rl'
                }
            
            elif model_name in self.q_learning_models:
                # Classical Q-Learning models
                q_model = self.q_learning_models[model_name](env.get_attr('observation_space')[0], 
                                                           env.get_attr('action_space')[0])
                
                # Train Q-Learning model
                training_results = self._train_q_learning(q_model, env, n_episodes=1000)
                
                results = {
                    'model': q_model,
                    'environment': env,
                    'training_results': training_results,
                    'model_type': 'q_learning'
                }
            
            else:
                raise ValueError(f"Model {model_name} not found")
            
            # Store trained model
            self.trained_models[model_name] = results
            self.model_performances[model_name] = results
            
            return results
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None
    
    def _evaluate_model(self, model, env, n_episodes=5):
        """Evaluate a trained RL model"""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'all_rewards': episode_rewards
        }
    
    def _create_q_learning_model(self, observation_space, action_space):
        """Create a Q-Learning model"""
        class QLearningAgent:
            def __init__(self, observation_space, action_space, learning_rate=0.1, 
                        discount_factor=0.95, epsilon=0.1):
                self.observation_space = observation_space
                self.action_space = action_space
                self.learning_rate = learning_rate
                self.discount_factor = discount_factor
                self.epsilon = epsilon
                
                # Discretize continuous observation space
                self.state_bins = 10
                self.q_table = np.zeros((self.state_bins ** 3, action_space.n))
                
            def _discretize_state(self, state):
                """Discretize continuous state to discrete state index"""
                # Use last 3 features as state representation
                relevant_features = state[-3:]
                
                # Bin each feature
                binned_features = []
                for feature in relevant_features:
                    # Normalize to [0, 1] range
                    normalized = (feature + 1) / 2  # Assuming features are in [-1, 1]
                    normalized = np.clip(normalized, 0, 1)
                    
                    # Discretize
                    bin_index = int(normalized * (self.state_bins - 1))
                    binned_features.append(bin_index)
                
                # Convert to single state index
                state_index = (binned_features[0] * self.state_bins * self.state_bins + 
                             binned_features[1] * self.state_bins + 
                             binned_features[2])
                
                return state_index
            
            def choose_action(self, state):
                """Choose action using epsilon-greedy policy"""
                state_index = self._discretize_state(state)
                
                if np.random.random() < self.epsilon:
                    return self.action_space.sample()
                else:
                    return np.argmax(self.q_table[state_index])
            
            def update(self, state, action, reward, next_state):
                """Update Q-table"""
                state_index = self._discretize_state(state)
                next_state_index = self._discretize_state(next_state)
                
                # Q-learning update
                best_next_action = np.argmax(self.q_table[next_state_index])
                td_target = reward + self.discount_factor * self.q_table[next_state_index][best_next_action]
                td_error = td_target - self.q_table[state_index][action]
                
                self.q_table[state_index][action] += self.learning_rate * td_error
        
        return QLearningAgent(observation_space, action_space)
    
    def _create_double_q_learning_model(self, observation_space, action_space):
        """Create a Double Q-Learning model"""
        class DoubleQLearningAgent:
            def __init__(self, observation_space, action_space, learning_rate=0.1, 
                        discount_factor=0.95, epsilon=0.1):
                self.observation_space = observation_space
                self.action_space = action_space
                self.learning_rate = learning_rate
                self.discount_factor = discount_factor
                self.epsilon = epsilon
                
                # Two Q-tables for double Q-learning
                self.state_bins = 10
                self.q_table_1 = np.zeros((self.state_bins ** 3, action_space.n))
                self.q_table_2 = np.zeros((self.state_bins ** 3, action_space.n))
                
            def _discretize_state(self, state):
                """Discretize continuous state to discrete state index"""
                relevant_features = state[-3:]
                
                binned_features = []
                for feature in relevant_features:
                    normalized = (feature + 1) / 2
                    normalized = np.clip(normalized, 0, 1)
                    bin_index = int(normalized * (self.state_bins - 1))
                    binned_features.append(bin_index)
                
                state_index = (binned_features[0] * self.state_bins * self.state_bins + 
                             binned_features[1] * self.state_bins + 
                             binned_features[2])
                
                return state_index
            
            def choose_action(self, state):
                """Choose action using epsilon-greedy policy with combined Q-tables"""
                state_index = self._discretize_state(state)
                
                if np.random.random() < self.epsilon:
                    return self.action_space.sample()
                else:
                    # Average of both Q-tables
                    combined_q = (self.q_table_1[state_index] + self.q_table_2[state_index]) / 2
                    return np.argmax(combined_q)
            
            def update(self, state, action, reward, next_state):
                """Update Q-tables using double Q-learning"""
                state_index = self._discretize_state(state)
                next_state_index = self._discretize_state(next_state)
                
                # Randomly choose which Q-table to update
                if np.random.random() < 0.5:
                    # Update Q-table 1
                    best_next_action = np.argmax(self.q_table_1[next_state_index])
                    td_target = reward + self.discount_factor * self.q_table_2[next_state_index][best_next_action]
                    td_error = td_target - self.q_table_1[state_index][action]
                    self.q_table_1[state_index][action] += self.learning_rate * td_error
                else:
                    # Update Q-table 2
                    best_next_action = np.argmax(self.q_table_2[next_state_index])
                    td_target = reward + self.discount_factor * self.q_table_1[next_state_index][best_next_action]
                    td_error = td_target - self.q_table_2[state_index][action]
                    self.q_table_2[state_index][action] += self.learning_rate * td_error
        
        return DoubleQLearningAgent(observation_space, action_space)
    
    def _train_q_learning(self, agent, env, n_episodes=1000):
        """Train Q-Learning agent"""
        episode_rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()[0]
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, truncated, info = env.step([action])
                
                agent.update(state, action, reward[0], next_state[0])
                
                state = next_state[0]
                episode_reward += reward[0]
            
            episode_rewards.append(episode_reward)
            
            # Decay epsilon
            if episode % 100 == 0:
                agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        return {
            'episode_rewards': episode_rewards,
            'mean_reward': np.mean(episode_rewards[-100:]),  # Last 100 episodes
            'final_epsilon': agent.epsilon
        }
    
    def backtest_strategy(self, model_name, test_data, initial_balance=10000):
        """
        Backtest a trained RL model on new data
        
        Args:
            model_name (str): Name of the trained model
            test_data (DataFrame): Test data
            initial_balance (float): Initial balance for backtesting
            
        Returns:
            dict: Backtesting results
        """
        try:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not trained")
            
            # Create test environment
            test_env = self.create_trading_environment(test_data, initial_balance=initial_balance)
            if test_env is None:
                return None
            
            # Get trained model
            model_info = self.trained_models[model_name]
            model = model_info['model']
            
            # Run backtest
            obs = test_env.reset()
            portfolio_values = [initial_balance]
            actions_taken = []
            trades = []
            
            while test_env.current_step < len(test_data) - 1:
                if model_info['model_type'] == 'deep_rl':
                    action, _ = model.predict(obs[0], deterministic=True)
                    action = action.item()
                else:  # Q-learning
                    action = model.choose_action(obs[0])
                
                obs, reward, done, truncated, info = test_env.step(action)
                
                portfolio_values.append(info['portfolio_value'])
                actions_taken.append(action)
                
                if done:
                    break
            
            # Calculate performance metrics
            final_portfolio_value = portfolio_values[-1]
            total_return = (final_portfolio_value - initial_balance) / initial_balance * 100
            
            # Calculate Sharpe ratio
            daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown) * 100
            
            results = {
                'initial_balance': initial_balance,
                'final_portfolio_value': final_portfolio_value,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'portfolio_values': portfolio_values,
                'actions_taken': actions_taken,
                'trades': test_env.trades,
                'n_trades': len(test_env.trades)
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error backtesting {model_name}: {str(e)}")
            return None
    
    def get_model_config(self, model_name):
        """
        Get configuration parameters for RL models
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Model configuration parameters
        """
        configs = {
            'PPO (Proximal Policy Optimization)': {
                'learning_rate': 0.001,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'clip_range': 0.2
            },
            'A2C (Advantage Actor-Critic)': {
                'learning_rate': 0.001,
                'n_steps': 5,
                'gamma': 0.99,
                'ent_coef': 0.01,
                'vf_coef': 0.25
            },
            'DQN (Deep Q-Network)': {
                'learning_rate': 0.001,
                'buffer_size': 50000,
                'learning_starts': 1000,
                'batch_size': 32,
                'gamma': 0.99,
                'target_update_interval': 1000
            },
            'SAC (Soft Actor-Critic)': {
                'learning_rate': 0.001,
                'buffer_size': 50000,
                'batch_size': 64,
                'gamma': 0.99,
                'tau': 0.005
            },
            'TD3 (Twin Delayed Deep Deterministic)': {
                'learning_rate': 0.001,
                'buffer_size': 50000,
                'batch_size': 64,
                'gamma': 0.99,
                'tau': 0.005
            },
            'Q-Learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'epsilon': 0.1,
                'state_bins': 10
            },
            'Double Q-Learning': {
                'learning_rate': 0.1,
                'discount_factor': 0.95,
                'epsilon': 0.1,
                'state_bins': 10
            }
        }
        
        # Environment parameters
        env_config = {
            'initial_balance': 10000,
            'transaction_cost': 0.001,
            'lookback_window': 10
        }
        
        model_config = configs.get(model_name, {})
        model_config.update(env_config)
        
        return model_config
