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


env = PatientEnvironment()
check_env(env, warn=True)

total_timesteps = 365_00_00
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1,
    learning_rate=1e-3,            
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,   
    exploration_fraction=0.2       
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
    num_steps = 365  # one year
    for patient_index in range(env.total_patients):
        observation, _ = env.reset(seed=10, patient_index=patient_index)
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
            patient_glucose_levels[patient_index].append(env.state['glucose_level'])
            patient_motivation_levels[patient_index].append(env.state['motivation'])
            patient_stress_levels[patient_index].append(env.state['stress_level'])
            action_summary[action] += 1
            patient_rewards[patient_index].append(reward)
            patient_actions[patient_index].append(action)
            writer1.writerow([
                patient_index + 1, step + 1, action, action_performed,
                env.state['age'], env.state['years_T2DM'], env.state['physical_activity'],
                env.state['motivation'],
                round(env.state['glucose_level'], 2),
                round(env.state['weight'], 1),
                round(env.state['stress_level'], 2),
                reward
            ])
            writer2.writerow([
                patient_index + 1, step + 1, action, observation.tolist(), reward
            ])
            if terminated or truncated:
                break
        all_rewards.append(np.mean(episode_rewards))
        env.next_patient()  # Move to the next patient

        if patient_index % 10 == 0:
                print(f"Processed patient {patient_index + 1}/{env.total_patients}")


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
def plot_average_metric_by_group(patients, metric_data, metric_name, group_name):
    max_length = max(len(metric_data[patient]) for patient in patients)
    padded_metric_data = [
        np.pad(metric_data[patient], (0, max_length - len(metric_data[patient])), 'constant', constant_values=np.nan)
        for patient in patients
    ]
    avg_metric = np.nanmean(padded_metric_data, axis=0)
    plt.figure(figsize=(12, 8))
    plt.plot(avg_metric, label=f'Average {metric_name} ({group_name})')
    plt.xlabel('Step')
    plt.ylabel(f'Average {metric_name}')
    plt.title(f'Average {metric_name} Over Time for {group_name} Patients')
    plt.legend()
    plt.savefig(f'average_{metric_name.lower()}_{group_name.lower()}.png')
    plt.close()


# def plot_metric_for_patient_group(patient_group, metric_data, metric_name, group_name):
#     plt.figure(figsize=(12, 8))
#     for patient_index in patient_group:
#         plt.plot(metric_data[patient_index], label=f'Patient {patient_index}')
#     plt.xlabel('Step')
#     plt.ylabel(metric_name)
#     plt.title(f'{metric_name} Over Time for {group_name}')
#     plt.legend()
#     plt.savefig(f'{metric_name.lower()}_{group_name.lower()}.png')
#     plt.close()

age_groups = {
    "18-30": [i for i in range(15)],
    "31-50": [i for i in range(15, 30)],
    "51-70": [i for i in range(30, 45)],
    "71-90": [i for i in range(45, 50)]
}

for group_name, patients in age_groups.items():
    plot_average_metric_by_group(patients, patient_glucose_levels, 'Glucose Level', group_name)
    plot_average_metric_by_group(patients, patient_motivation_levels, 'Motivation', group_name)
    plot_average_metric_by_group(patients, patient_stress_levels, 'Stress Level', group_name)

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