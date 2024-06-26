from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from Patient_env import PatientEnvironment
import csv
import numpy as np
from Patient_data import generate_random_patient, Patient, set_seed
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import create_dataset


env = PatientEnvironment(data_file="patients_data_grouped_one.csv")
check_env(env, warn=True)

total_timesteps = 100_00_00
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1       
)

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
    for episode in range(total_timesteps // num_steps):
        observation, _ = env.reset(seed=123)
        episode_rewards = []

        for step in range(num_steps):
            action, _ = model.predict(observation, deterministic=False)  # Ensure exploration by using deterministic=False
            
            # Ensure action is hashable
            if isinstance(action, np.ndarray) and action.ndim == 0:
                action = int(action)  # Convert 0-d array to scalar
            else:
                action = tuple(action)  # Convert array to tuple for hashability

            observation, reward, terminated, truncated, info = env.step(action)

            patient_id = env.current_patient_index - 1 if env.current_patient_index != 0 else env.total_patients - 1
            action_performed = info['action_performed']

            patient_rewards[patient_id].append(reward)
            patient_actions[patient_id].append(action)
            patient_glucose_levels[patient_id].append(env.patient_states[patient_id]['glucose_level'])
            patient_motivation_levels[patient_id].append(env.patient_states[patient_id]['motivation'])
            patient_stress_levels[patient_id].append(env.patient_states[patient_id]['stress_level'])
            action_summary[action] += 1

            writer1.writerow([
                patient_id + 1, step + 1, action, action_performed,
                env.patient_states[patient_id]['age'], env.patient_states[patient_id]['years_T2DM'], env.patient_states[patient_id]['physical_activity'],
                env.patient_states[patient_id]['motivation'],
                round(env.patient_states[patient_id]['glucose_level'], 2),
                round(env.patient_states[patient_id]['weight'], 1),
                round(env.patient_states[patient_id]['stress_level'], 2),
                reward
                ])
            writer2.writerow([
                patient_id + 1, step + 1, action, observation.tolist(), reward
                ])

            if terminated or truncated:
                break

            episode_rewards.append(reward)

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

# Function to plot average metric over time for different groups
# def plot_average_metric_by_group(patients, metric_data, metric_name, group_name):
#     max_length = max(len(metric_data[patient]) for patient in patients)
#     padded_metric_data = [
#         np.pad(metric_data[patient], (0, max_length - len(metric_data[patient])), 'constant', constant_values=np.nan)
#         for patient in patients
#     ]
#     avg_metric = np.nanmean(padded_metric_data, axis=0)
#     plt.figure(figsize=(12, 8))
#     plt.plot(avg_metric, label=f'Average {metric_name} ({group_name})')
#     plt.xlabel('Step')
#     plt.ylabel(f'Average {metric_name}')
#     plt.title(f'Average {metric_name} Over Time for {group_name} Patients')
#     plt.legend()
#     plt.savefig(f'average_{metric_name.lower()}_{group_name.lower()}.png')
#     plt.close()

# age_groups = {
#     "18-30": [i for i in range(15)],
#     "31-50": [i for i in range(15, 30)],
#     "51-70": [i for i in range(30, 45)],
#     "71-90": [i for i in range(45, 50)]
# }

# for group_name, patients in age_groups.items():
#     plot_average_metric_by_group(patients, patient_glucose_levels, 'Glucose Level', group_name)
#     plot_average_metric_by_group(patients, patient_motivation_levels, 'Motivation', group_name)
#     plot_average_metric_by_group(patients, patient_stress_levels, 'Stress Level', group_name)

def plot_average_metric(metric_data, metric_name):
    plt.figure(figsize=(12, 8))
    plt.plot(metric_data)
    plt.xlabel('Step')
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Average {metric_name} Over Time')
    plt.savefig(f'average_{metric_name.lower()}.png')
    plt.close()

# Plotting metrics
for patient_id, glucose_levels in patient_glucose_levels.items():
    plot_average_metric(glucose_levels, f'Glucose Level Patient {patient_id}')
for patient_id, motivation_levels in patient_motivation_levels.items():
    plot_average_metric(motivation_levels, f'Motivation Patient {patient_id}')
for patient_id, stress_levels in patient_stress_levels.items():
    plot_average_metric(stress_levels, f'Stress Level Patient {patient_id}')

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

#learning curve
plt.figure(figsize=(12, 8))
for patient_index, rewards in patient_rewards.items():
    plt.plot(rewards, label=f'Patient {patient_index}')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Learning Curve of the Agent per Patient')
plt.legend()
plt.savefig('learning_curve_per_patient.png')
plt.close()