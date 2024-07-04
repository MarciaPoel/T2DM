from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from environment import PatientEnvironment
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

env = PatientEnvironment()
check_env(env, warn=True)
env = Monitor(env, filename='monitor_log_old.csv')

total_timesteps = 500000
model = DQN("MlpPolicy", env, verbose=1,
    learning_rate=0.0004,  # Adjusted learning rate
    batch_size=64,  # Adjusted batch size
    buffer_size=100000,  # Adjusted buffer size
    train_freq=4,  # Update the model every 4 steps
    exploration_final_eps=0.01,
    exploration_fraction=0.5,
    exploration_initial_eps=1.0
)

os.makedirs("old", exist_ok=True)
model.learn(total_timesteps=total_timesteps)

# Reduce the number of evaluation episodes for quicker testing
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=4)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

model.save("dqn_patient_old")
model = DQN.load("dqn_patient_old")

# Initialize CSV files
with open('patient_simulation_old.csv', mode='w', newline='') as file1, open('decisions_log_motiv_old.csv', mode='w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)

    writer1.writerow([
        'Patient', 'Step', 'Action', 'Performed', 'Age', 'Years_T2DM', 'Physical_Activity',
        'Motivation', 'Glucose_Level', 'Reward'
    ])

    writer2.writerow([
        'Patient', 'Step', 'Action', 'Observation', 'Reward'
    ])

    all_rewards = []
    all_glucose_levels = []
    all_motivation_levels = []
    glucose_levels = defaultdict(list)
    motivation_levels = defaultdict(list)
    action_summary = {action: 0 for action in range(env.action_space.n)}
    patient_rewards = defaultdict(list)
    episode_rewards = []  # To store rewards for each episode

    # Simulate the environment
    num_steps = 365  # One year of simulation
    patient_id = 0  # Assign a patient ID
    episode_count = 0  # Initialize episode counter

    # Initialize environment and variables for a new episode
    observation, _ = env.reset(seed=123)
    current_episode_rewards = []  # To store rewards for the current episode
    
    for step in range(num_steps):
        action, _states = model.predict(observation, deterministic=False)

        if isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.item()  # Convert 0-d array to scalar
        else:
            action = tuple(action)  # Convert array to tuple for hashability

        observation, reward, terminated, truncated, info = env.step(action)
        current_episode_rewards.append(reward)

        writer1.writerow([
            patient_id, step, action, info['action_performed'], env.state['age'], env.state['years_T2DM'], 
            env.state['physical_activity'], f"{env.state['motivation']:.1f}", f"{env.state['glucose_level']:.2f}", 
            f"{reward:.1}"
        ])

        writer2.writerow([
            patient_id, step, action, observation, reward
        ])

        all_rewards.append(reward)
        all_glucose_levels.append(env.state['glucose_level'])
        all_motivation_levels.append(env.state['motivation'])

        glucose_levels[step].append(env.state['glucose_level'])
        motivation_levels[step].append(env.state['motivation'])
        patient_rewards[patient_id].append(reward)
        action_summary[action] += 1

        # Add debug prints
        #print(f"Step: {step}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated or (step % num_steps == 0 and step > 0):
            print("Episode ended due to termination or truncation. Resetting environment.")
            episode_rewards.append(np.mean(current_episode_rewards))  # Record average reward for the episode
            current_episode_rewards = []  # Reset current episode rewards
            observation, _ = env.reset(seed=123)
            episode_count += 1  # Increment episode counter
            

print("Simulation completed, results saved in 'patient_simulation.csv' and 'decisions_log.csv'.")
print(f"Total episodes run: {episode_count}")

# Save action summary to CSV
with open('action_summary.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Action', 'Count'])
    for action, count in action_summary.items():
        writer.writerow([action, count])

df = pd.read_csv('patient_simulation_old.csv')

# Compute the averages per step
df_grouped = df.groupby('Step').mean().reset_index()

# Plot average glucose level and average motivation
plt.figure(figsize=(14, 6))

plt.plot(df_grouped['Step'], df_grouped['Glucose_Level'], label='Average Glucose Level', color='green')
plt.plot(df_grouped['Step'], df_grouped['Motivation'], label='Average Motivation', color='red')
plt.axhspan(100, 125, color='yellow', alpha=0.3, label='Target Range')
plt.xlabel('Step')
plt.ylabel('Levels')
plt.title('Average Glucose Level and Motivation Over Time')
plt.legend()
plt.xlim(0, num_steps)
plt.ylim(0, max(df_grouped['Glucose_Level'].max(), df_grouped['Motivation'].max()) * 1.1)  # Add some padding for better visibility

plt.tight_layout()
plt.savefig('old/average_glucose_and_motivation_plot.png')
plt.close()

# Plot average reward per step and per episode
plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
plt.plot(df_grouped['Step'], df_grouped['Reward'], label='Average Reward per Step', color='blue')
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.legend()
plt.xlim(0, num_steps)
#plt.ylim(df_grouped['Reward'].min() * 1.1, df_grouped['Reward'].max() * 1.1)  # Add some padding for better visibility

plt.subplot(2, 1, 2)
plt.plot(range(episode_count), episode_rewards, label='Average Reward per Episode', color='purple')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.legend()
plt.xlim(0, len(episode_rewards) - 1)
#plt.ylim(min(episode_rewards) * 1.1, max(episode_rewards) * 1.1)  # Add some padding for better visibility

plt.tight_layout()
plt.savefig('old/average_rewards_plot.png')
plt.show()
