import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

class ReinforcementLearningModels:
    """Reinforcement Learning Models for Stock Trading (Safe Version)"""
    
    def __init__(self):
        self.available_models = ['Simple Q-Learning', 'Buy and Hold Strategy', 'Moving Average Strategy']
        self.trained_models = {}
        self.model_performances = {}
        
        st.info("Advanced RL libraries not available. Using simplified trading strategies.")
    
    def get_available_models(self):
        """Return list of available RL models"""
        return self.available_models
    
    def train_model(self, model_name, data, total_timesteps=10000, **kwargs):
        """
        Train a reinforcement learning model
        
        Args:
            model_name (str): Name of the model
            data (DataFrame): Stock data
            total_timesteps (int): Training timesteps
            
        Returns:
            dict: Training results
        """
        try:
            if model_name == 'Simple Q-Learning':
                return self._train_simple_q_learning(data, **kwargs)
            elif model_name == 'Buy and Hold Strategy':
                return self._train_buy_hold(data, **kwargs)
            elif model_name == 'Moving Average Strategy':
                return self._train_moving_average_strategy(data, **kwargs)
            else:
                return {
                    'error': f'Model {model_name} not available',
                    'model_name': model_name,
                    'mean_reward': 0
                }
                
        except Exception as e:
            st.error(f"Error training RL model {model_name}: {str(e)}")
            return None
    
    def _train_simple_q_learning(self, data, **kwargs):
        """Simple Q-learning implementation for stock trading"""
        try:
            prices = data['Close'].dropna().values
            
            # Simple state space: price trend (up/down/stable)
            # Action space: buy/sell/hold
            
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            
            # Simple strategy: buy when price is trending up, sell when down
            actions = []
            portfolio_value = 1000  # Starting value
            position = 0  # 0 = no position, 1 = long position
            
            for i, return_val in enumerate(returns):
                if return_val > 0.01 and position == 0:  # Buy signal
                    action = 'buy'
                    position = 1
                    portfolio_value *= (1 + return_val)
                elif return_val < -0.01 and position == 1:  # Sell signal
                    action = 'sell'
                    position = 0
                    portfolio_value *= (1 + return_val)
                else:
                    action = 'hold'
                    if position == 1:
                        portfolio_value *= (1 + return_val)
                
                actions.append(action)
            
            # Calculate performance
            total_return = (portfolio_value - 1000) / 1000 * 100
            mean_reward = total_return / len(returns)
            
            results = {
                'model_name': 'Simple Q-Learning',
                'total_return': total_return,
                'final_portfolio_value': portfolio_value,
                'mean_reward': mean_reward,
                'actions_taken': len([a for a in actions if a != 'hold']),
                'model_type': 'Q-Learning'
            }
            
            self.model_performances['Simple Q-Learning'] = results
            
            return {
                'evaluation': results,
                'model_name': 'Simple Q-Learning'
            }
            
        except Exception as e:
            st.error(f"Error in simple Q-learning: {str(e)}")
            return None
    
    def _train_buy_hold(self, data, **kwargs):
        """Buy and hold strategy"""
        try:
            prices = data['Close'].dropna().values
            
            # Simple buy and hold
            initial_price = prices[0]
            final_price = prices[-1]
            
            total_return = (final_price - initial_price) / initial_price * 100
            mean_reward = total_return / len(prices)
            
            results = {
                'model_name': 'Buy and Hold Strategy',
                'total_return': total_return,
                'initial_price': initial_price,
                'final_price': final_price,
                'mean_reward': mean_reward,
                'model_type': 'Buy and Hold'
            }
            
            self.model_performances['Buy and Hold Strategy'] = results
            
            return {
                'evaluation': results,
                'model_name': 'Buy and Hold Strategy'
            }
            
        except Exception as e:
            st.error(f"Error in buy and hold strategy: {str(e)}")
            return None
    
    def _train_moving_average_strategy(self, data, window=20, **kwargs):
        """Moving average crossover strategy"""
        try:
            prices = data['Close'].dropna()
            
            # Calculate moving averages
            short_ma = prices.rolling(window=window//2).mean()
            long_ma = prices.rolling(window=window).mean()
            
            # Generate signals
            signals = []
            position = 0
            portfolio_value = 1000
            
            for i in range(len(prices)):
                if i < window:
                    signals.append('hold')
                    continue
                
                if short_ma.iloc[i] > long_ma.iloc[i] and position == 0:
                    signals.append('buy')
                    position = 1
                elif short_ma.iloc[i] < long_ma.iloc[i] and position == 1:
                    signals.append('sell')
                    position = 0
                else:
                    signals.append('hold')
                
                # Update portfolio value
                if i > 0 and position == 1:
                    return_val = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
                    portfolio_value *= (1 + return_val)
            
            total_return = (portfolio_value - 1000) / 1000 * 100
            mean_reward = total_return / len(prices)
            
            results = {
                'model_name': 'Moving Average Strategy',
                'total_return': total_return,
                'final_portfolio_value': portfolio_value,
                'mean_reward': mean_reward,
                'window': window,
                'signals_generated': len([s for s in signals if s != 'hold']),
                'model_type': 'Moving Average Crossover'
            }
            
            self.model_performances['Moving Average Strategy'] = results
            
            return {
                'evaluation': results,
                'model_name': 'Moving Average Strategy'
            }
            
        except Exception as e:
            st.error(f"Error in moving average strategy: {str(e)}")
            return None
    
    def get_model_performance(self, model_name):
        """Get performance metrics for a specific model"""
        return self.model_performances.get(model_name, {})
    
    def get_all_performances(self):
        """Get all model performances"""
        return self.model_performances