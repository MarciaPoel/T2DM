import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from Patient_env import PatientEnvironment

# Create the environment and check it
env = PatientEnvironment(data_file="patients_data_grouped.csv")
check_env(env, warn=True)

# Training parameters
total_timesteps = 500_000
num_steps = 365  # One year of simulation

# Create the a2c model
model = A2C(
    "MlpPolicy",
    env,
    verbose=1
)

# Create a directory to save a2c results
os.makedirs("a2c", exist_ok=True)

# Setup callbacks
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./a2c/models/', name_prefix='a2c_patient')
eval_callback = EvalCallback(env, best_model_save_path='./a2c/logs/', log_path='./a2c/logs/', eval_freq=5000)

print("Training started...")
model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])
print("Training completed.")# Train the model

model.save("a2c/a2c_patient")
model = A2C.load("a2c/a2c_patient")

# Initialize CSV files
with open('a2c/patient_simulation.csv', mode='w', newline='') as file1, open('a2c/decisions_log.csv', mode='w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)

    writer1.writerow([
        'Patient', 'Step', 'Action', 'Performed', 'Age', 'Years_T2DM', 'Physical_Activity',
        'Motivation', 'Glucose_Level', 'Weight', 'Reward'
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

    # Age groups
    age_groups = {
        "18-30": [],
        "31-50": [],
        "51-70": [],
        "71-90": []
    }

    # Group patients by age
    for i, state in enumerate(env.patient_states):
        if 18 <= state['age'] <= 30:
            age_groups["18-30"].append(i)
        elif 31 <= state['age'] <= 50:
            age_groups["31-50"].append(i)
        elif 51 <= state['age'] <= 70:
            age_groups["51-70"].append(i)
        elif 71 <= state['age'] <= 90:
            age_groups["71-90"].append(i)

    # Simulate the environment for one patient per episode
    for episode in range(total_timesteps // num_steps):
        patient_id = episode % env.total_patients  # Cycle through patients
        print(f"Episode {episode + 1}/{total_timesteps // num_steps} - Patient {patient_id + 1}")
        observation, _ = env.reset(seed=123, patient_index=patient_id)
        episode_rewards = []

        for step in range(num_steps):
            action, _ = model.predict(observation, deterministic=False)  # Ensure exploration by using deterministic=False
            
            # Ensure action is hashable
            action = int(action) if isinstance(action, np.ndarray) and action.ndim == 0 else tuple(action)

            observation, reward, terminated, truncated, info = env.step(action)

            action_performed = info['action_performed']

            patient_rewards[patient_id].append(reward)
            glucose_levels[patient_id].append(env.patient_states[patient_id]['glucose_level'])
            motivation_levels[patient_id].append(env.patient_states[patient_id]['motivation'])
            action_summary[action] += 1

            writer1.writerow([
                patient_id + 1, step + 1, action, action_performed,
                env.patient_states[patient_id]['age'], env.patient_states[patient_id]['years_T2DM'],
                env.patient_states[patient_id]['physical_activity'], env.patient_states[patient_id]['motivation'],
                round(env.patient_states[patient_id]['glucose_level'], 2),
                round(env.patient_states[patient_id]['weight'], 1), reward
            ])
            writer2.writerow([
                patient_id + 1, step + 1, action, observation.tolist(), reward
            ])

            if terminated or truncated:
                break

            episode_rewards.append(reward)

        # Calculate and store average glucose and motivation levels
        avg_glucose = np.mean([state['glucose_level'] for state in env.patient_states])
        avg_motivation = np.mean([state['motivation'] for state in env.patient_states])
        all_glucose_levels.append(avg_glucose)
        all_motivation_levels.append(avg_motivation)

        all_rewards.append(np.mean(episode_rewards))

print("Simulation completed, results saved in 'a2c/patient_simulation.csv' and 'a2c/decisions_log.csv'.")

# Save action summary to CSV
with open('a2c/action_summary.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Action', 'Count'])
    for action, count in action_summary.items():
        writer.writerow([action, count])
    
def plot_average_metric(metric_data, metric_name):
    plt.figure(figsize=(12, 8))
    plt.plot(metric_data)
    plt.xlabel('Episode')
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Average {metric_name} Over Time')
    plt.savefig(f'a2c/average_{metric_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_average_metric_by_group(age_groups, metric_data, metric_name):
    plt.figure(figsize=(12, 8))
    for group_name, patients in age_groups.items():
        group_data = [metric_data[patient] for patient in patients if len(metric_data[patient]) > 0]
        
        if len(group_data) == 0:
            continue  # Skip if there's no data for this group
        
        # Find the length of the longest sequence
        max_length = max(len(data) for data in group_data)
        
        # Pad sequences with NaN values to make them of uniform length
        padded_data = [np.pad(data, (0, max_length - len(data)), constant_values=np.nan) for data in group_data]
        
        avg_metric = np.nanmean(padded_data, axis=0)
        plt.plot(avg_metric, label=f'Average {metric_name} ({group_name})')
    
    plt.xlabel('Step')
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Average {metric_name} Over Time for Age Groups')
    plt.legend()
    plt.savefig(f'average_{metric_name.lower().replace(" ", "_")}_age_groups.png')
    plt.close()

# Plotting overall average glucose and motivation
plot_average_metric(all_glucose_levels, 'Glucose Level')
plot_average_metric(all_motivation_levels, 'Motivation')

# Plotting metrics by age group
plot_average_metric_by_group(age_groups, glucose_levels, 'Glucose Level')
plot_average_metric_by_group(age_groups, motivation_levels, 'Motivation')

# Plotting the action summary
plt.figure(figsize=(12, 8))
actions = list(action_summary.keys())
counts = list(action_summary.values())
plt.bar(actions, counts)
plt.xlabel('Action')
plt.ylabel('Count')
plt.title('Action Summary')
plt.savefig('a2c/action_summary.png')
plt.close()

# Learning curve
plt.figure(figsize=(12, 8))
for patient_index, rewards in patient_rewards.items():
    plt.plot(rewards, label=f'Patient {patient_index}')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Learning Curve of the Agent per Patient')
plt.legend()
plt.savefig('a2c/learning_curve_per_patient.png')
plt.close()

# Plot overall learning curve
plot_average_metric(all_rewards, 'Reward')
