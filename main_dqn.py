from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from Patient_env_one import PatientEnvironment
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

env = PatientEnvironment()
check_env(env, warn=True)

total_timesteps = 1000000
model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=total_timesteps)
model.save("dqn_patient")
model = DQN.load("dqn_patient")

# Initialize CSV files
with open('patient_simulation.csv', mode='w', newline='') as file1, open('decisions_log.csv', mode='w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)

    writer1.writerow([
        'Patient', 'Step', 'Action', 'Performed', 'Age', 'Years_T2DM', 'Physical_Activity',
        'Motivation', 'Glucose_Level', 'Weight', 'Stress_Level', 'Reward'
    ])

    writer2.writerow([
        'Patient', 'Step', 'Action', 'Observation', 'Reward'
    ])

    all_rewards = []
    glucose_levels = []
    motivation_levels = []
    stress_levels = []
    action_summary = {action: 0 for action in range(env.action_space.n)}
    patient_rewards = defaultdict(list)
    patient_actions = defaultdict(list)
    patient_glucose_levels = defaultdict(list)
    patient_stress_levels = defaultdict(list)
    patient_motivation_levels = defaultdict(list)

    # Simulate the environment
    num_steps = 365  # One year of simulation
    observation, _ = env.reset(seed=123)
    episode_rewards = []
    for step in range(num_steps):
        action, _ = model.predict(observation, deterministic=False)  # Ensure exploration by using deterministic=False

        # Ensure action is hashable
        if isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.item()  # Convert 0-d array to scalar
        else:
            action = tuple(action)  # Convert array to tuple for hashability

        observation, reward, terminated, truncated, info = env.step(action)
        action_performed = info['action_performed']
        episode_rewards.append(reward)
        glucose_levels.append(env.state['glucose_level'])
        motivation_levels.append(env.state['motivation'])
        stress_levels.append(env.state['stress_level'])
        action_summary[action] += 1
        patient_rewards[0].append(reward)
        patient_actions[0].append(action)
        patient_glucose_levels[0].append(env.state['glucose_level'])
        patient_motivation_levels[0].append(env.state['motivation'])
        patient_stress_levels[0].append(env.state['stress_level'])
        writer1.writerow([
            1, step + 1, action, action_performed,
            env.state['age'], env.state['years_T2DM'], env.state['physical_activity'],
            env.state['motivation'],
            round(env.state['glucose_level'], 2),
            round(env.state['weight'], 1),
            round(env.state['stress_level'], 2),
            reward
        ])
        writer2.writerow([
            1, step + 1, action, observation.tolist(), reward
        ])
        if terminated or truncated:
            break
    all_rewards.append(np.mean(episode_rewards))

print("Simulation completed, results saved in 'patient_simulation.csv' and 'decisions_log.csv'.")

# Save action summary to CSV
with open('action_summary.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Action', 'Count'])
    for action, count in action_summary.items():
        writer.writerow([action, count])

# Plotting the average reward per episode
plt.figure(figsize=(12, 8))
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward per Episode Over Time')
plt.savefig('average_reward_per_episode.png')
plt.close()

# Function to plot average metric over time
def plot_average_metric(metric_data, metric_name):
    plt.figure(figsize=(12, 8))
    plt.plot(metric_data)
    plt.xlabel('Step')
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Average {metric_name} Over Time')
    plt.savefig(f'average_{metric_name.lower()}.png')
    plt.close()

# Plotting metrics
plot_average_metric(glucose_levels, 'Glucose Level')
plot_average_metric(motivation_levels, 'Motivation')
plot_average_metric(stress_levels, 'Stress Level')

# Plotting the action summary
plt.figure(figsize=(12, 8))
actions = list(action_summary.keys())
counts = list(action_summary.values())
plt.bar(actions, counts)
plt.xlabel('Action')
plt.ylabel('Count')
plt.title('Action Summary')
plt.savefig('action_summary.png')
plt.close()

# Learning curve
plt.figure(figsize=(12, 8))
plt.plot(patient_rewards[0], label='Patient 1')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Learning Curve of the Agent')
plt.legend()
plt.savefig('learning_curve.png')
plt.close()
