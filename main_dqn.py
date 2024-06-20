from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from Patient_env import PatientEnvironment
import csv
import numpy as np
from Patient_data import generate_random_patient, Patient, set_seed
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Ensure dataset exists
import create_dataset

# Create the environment
env = PatientEnvironment()

# Check if the environment follows the gymnasium interface
check_env(env, warn=True)

# Adjust total timesteps for training
total_timesteps = 365000

# Create the PPO model with adjusted parameters for more exploration
model = DQN("MlpPolicy", env, verbose =1)

# Train the agent
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("dqn_patient")

# Load the trained model
model = DQN.load("dqn_patient")

# Initialize CSV files
with open('patient_simulation.csv', mode='w', newline='') as file1, open('decisions_log.csv', mode='w', newline='') as file2:
    writer1 = csv.writer(file1)
    writer2 = csv.writer(file2)

    writer1.writerow([
        'Patient', 'Step', 'Action', 'Age', 'Years_T2DM', 'Physical_Activity',
        'Glucose_Level', 'Weight', 'Motivation', 'Reward'
    ])

    writer2.writerow([
        'Patient', 'Step', 'Action', 'Observation', 'Reward'
    ])

    all_rewards = []
    glucose_levels = []
    motivation_levels = []
    weights = []
    action_summary = {action: 0 for action in range(env.action_space.n)}
    patient_rewards = defaultdict(list)
    patient_actions = defaultdict(list)
    motivation_timelines = defaultdict(list)

    # Simulate the environment
    num_steps = 365  # one year
    for patient_index in range(env.total_patients):
        observation, _ = env.reset(seed=456, patient_index=patient_index)
        episode_rewards = []
        episode_glucose_levels = []
        episode_motivation_levels = []
        episode_weights = []
        initial_motivation = env.state['motivation']
        for step in range(num_steps):
            action, _ = model.predict(observation, deterministic=False)  # Ensure exploration by using deterministic=False
            
            # Ensure action is hashable
            if isinstance(action, np.ndarray) and action.ndim == 0:
                action = action.item()  # Convert 0-d array to scalar
            else:
                action = tuple(action)  # Convert array to tuple for hashability
            
            observation, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            episode_glucose_levels.append(env.state['glucose_level'])
            episode_motivation_levels.append(env.state['motivation'])
            episode_weights.append(env.state['weight'])
            action_summary[action] += 1
            patient_rewards[patient_index].append(reward)
            patient_actions[patient_index].append(action)
            motivation_timelines[initial_motivation].append((step, env.state['motivation']))
            writer1.writerow([
                patient_index + 1, step + 1, action,
                env.state['age'], env.state['years_T2DM'], env.state['physical_activity'],
                round(env.state['glucose_level'], 2),
                round(env.state['weight'], 1),
                env.state['motivation'],
                reward
            ])
            writer2.writerow([
                patient_index + 1, step + 1, action, observation.tolist(), reward
            ])
            if terminated or truncated:
                break
        all_rewards.append(np.mean(episode_rewards))
        glucose_levels.append(np.mean(episode_glucose_levels))
        motivation_levels.append(np.mean(episode_motivation_levels))
        weights.append(np.mean(episode_weights))
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

# Plotting the average glucose level per episode
plt.figure(figsize=(12, 8))
plt.plot(glucose_levels)
plt.xlabel('Episode')
plt.ylabel('Average Glucose Level')
plt.title('Average Glucose Level per Episode Over Time')
plt.savefig('average_glucose_level_per_episode.png')
plt.close()

# Plotting the average motivation level per episode
plt.figure(figsize=(12, 8))
plt.plot(motivation_levels)
plt.xlabel('Episode')
plt.ylabel('Average Motivation Level')
plt.title('Average Motivation Level per Episode Over Time')
plt.savefig('average_motivation_level_per_episode.png')
plt.close()

# # Plotting the average weight per episode
# plt.figure(figsize=(12, 8))
# plt.plot(weights)
# plt.xlabel('Episode')
# plt.ylabel('Average Weight')
# plt.title('Average Weight per Episode Over Time')
# plt.savefig('average_weight_per_episode.png')
# plt.close()

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

# Analyzing action diversity for each patient
action_diversity = {patient: len(set(actions)) for patient, actions in patient_actions.items()}

# Plotting action diversity per patient
palet = sns.color_palette("hsv", len(action_diversity))

plt.figure(figsize=(12, 8))
for idx, (patient_index, diversity) in enumerate(action_diversity.items()):
    plt.bar(patient_index, diversity, color=palet[idx], label=f'Patient {patient_index}', alpha=0.7)
plt.xlabel('Patient Index')
plt.ylabel('Action Diversity (Number of Unique Actions)')
plt.title('Action Diversity per Patient')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.savefig('action_diversity_per_patient.png')
plt.close()

# Analyzing and plotting actions for different patient clusters
patient_clusters = {'Young': [], 'Middle-aged': [], 'Old': []}
for patient_index in range(env.total_patients):
    patient_age = env.patient_data.iloc[patient_index]['age']
    if patient_age < 30:
        patient_clusters['Young'].extend(patient_actions[patient_index])
    elif 30 <= patient_age < 60:
        patient_clusters['Middle-aged'].extend(patient_actions[patient_index])
    else:
        patient_clusters['Old'].extend(patient_actions[patient_index])

plt.figure(figsize=(12, 8))
for cluster, actions in patient_clusters.items():
    action_counts = {action: actions.count(action) for action in set(actions)}
    plt.bar(action_counts.keys(), action_counts.values(), alpha=0.5, label=cluster)

plt.xlabel('Action')
plt.ylabel('Count')
plt.title('Action Distribution by Patient Age Cluster')
plt.legend()
plt.savefig('action_distribution_by_age_cluster.png')
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

# Plotting the chosen action per patient
palette = sns.color_palette("hsv", len(patient_actions))

plt.figure(figsize=(12, 8))
for idx, (patient_index, actions) in enumerate(patient_actions.items()):
    action_counts = {action: actions.count(action) for action in set(actions)}
    actions_sorted = sorted(action_counts.keys())
    counts_sorted = [action_counts[action] for action in actions_sorted]
    plt.bar(actions_sorted, counts_sorted, label=f'Patient {patient_index}', alpha=0.5, color=palette[idx])
plt.xlabel('Action')
plt.ylabel('Count')
plt.title('Chosen Action per Patient')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.savefig('chosen_action_per_patient.png')
plt.close()

# Plotting the motivation timeline clustered by their starting value
plt.figure(figsize=(12, 8))
for initial_motivation, timeline in motivation_timelines.items():
    steps, motivations = zip(*timeline)
    plt.plot(steps, motivations, label=f'Initial Motivation {initial_motivation}')
plt.xlabel('Step')
plt.ylabel('Motivation')
plt.title('Motivation Timeline Clustered by Initial Value')
plt.legend()
plt.savefig('motivation_timeline_clustered.png')
plt.close()