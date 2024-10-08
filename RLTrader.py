import warnings
warnings.filterwarnings("ignore")

import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
from finta import TA
import numpy as np
import matplotlib.pyplot as plt
import torch

# Check CUDA availability
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")

# Read data from CSV
df = pd.read_csv("FILL IN", index_col='timestamp', parse_dates=True)

# Add a 'date' column
df['date'] = df.index

# Add technical indicators using finta
df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
macd = TA.MACD(df)
df['MACD'] = macd['MACD']
df['MACD_SIGNAL'] = macd['SIGNAL']
df['OBV'] = TA.OBV(df)

# Handle NaN values by filling with 0
df.fillna(0, inplace=True)

# Rename columns to match expected format
df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)

# Print debug information
print(df.head())  # To verify that columns are added correctly
print(df.isna().sum())  # Check for any remaining NaN values

# Define a custom environment
def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Close', 'Volume', 'RSI', 'MACD', 'MACD_SIGNAL', 'OBV']].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = add_signals

# Create the custom environment
frame_bound = (12, len(df) - 1)
env2 = MyCustomEnv(df=df, window_size=12, frame_bound=frame_bound)

# Verify the observation space
print("Observation space shape:", env2.observation_space.shape)
print("Frame bound:", frame_bound)

# Wrap the environment
env = DummyVecEnv([lambda: env2])

# Initialize and train the A2C model with additional parameters
model = A2C('MlpPolicy', env, 
            learning_rate=0.0004, 
            n_steps=20, 
            gamma=0.99, 
            vf_coef=0.2, 
            ent_coef=0.005, 
            max_grad_norm=0.5, 
            rms_prop_eps=1e-5, 
            use_rms_prop=True, 
            verbose=1, 
            device='cuda'
            )

try:
    model.learn(total_timesteps=1000000, log_interval=500)
except ValueError as e:
    print(f"ValueError during training: {e}")
    print(f"Observations: {env.reset()}")

# Initialize money tracking
initial_money = 100000  # Starting money
current_money = initial_money
money_over_time = [current_money]
trade_prices = []

# Test the model
obs = env.reset()
cumulative_reward = 0
num_trades = 0

while True:
    obs = np.expand_dims(obs[0], axis=0)  # Correctly handle the observation
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    cumulative_reward += float(reward)  # Convert reward to float
    num_trades += 1

    # Update current money (assuming reward is directly affecting the money)
    current_money += float(reward)  # Convert to float
    money_over_time.append(current_money)
    if isinstance(info, dict) and 'price' in info:
        trade_prices.append(info['price'])

    if done:
        break

# Convert current_money and cumulative_reward to a scalar for printing
current_money = float(current_money)
cumulative_reward = float(cumulative_reward)

# Print statistics
print("Episode finished")
print(f"Initial Money: ${initial_money:.2f}")
print(f"Final Money: ${current_money:.2f}")
print(f"Cumulative Reward: {cumulative_reward}")
print(f"Number of Trades: {num_trades}")
if trade_prices:
    print(f"Average Trade Price: {np.mean(trade_prices):.2f}")
    print(f"Max Trade Price: {np.max(trade_prices):.2f}")
    print(f"Min Trade Price: {np.min(trade_prices):.2f}")

# Plotting the results in the style similar to the second code snippet

# Plot money over time
plt.figure(figsize=(15, 6))
plt.plot(money_over_time, label='Money Over Time')
plt.xlabel('Steps')
plt.ylabel('Money')
plt.legend()
plt.title('Money Over Time During Testing')
plt.show()

# Plot trade prices if available
if trade_prices:
    plt.figure(figsize=(15, 6))
    plt.plot(trade_prices, label='Trade Prices', color='orange')
    plt.xlabel('Trades')
    plt.ylabel('Price')
    plt.legend()
    plt.title('Trade Prices Over Time')
    plt.show()