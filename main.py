from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from env_no_motiv import PatientEnvironment
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_path = "single_young_421.csv" 
patient_data = pd.read_csv(file_path)

print("Initializing environment...")
env = PatientEnvironment(data_file=file_path)

total_timesteps = 1_000_000
model = DQN("MlpPolicy", env, verbose=1,
            learning_rate=0.0001,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.1,
            exploration_fraction=0.5)

os.makedirs("dqn/single_patient/10mln/", exist_ok=True)

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Step: {self.n_calls}, Time elapsed: {self.model.num_timesteps}")
        return True

progress_callback = ProgressCallback(check_freq=10000)

# Train
model.learn(total_timesteps=total_timesteps, callback=progress_callback)
print("Model learning completed.")
model.save("dqn/single_patient/10mln/dqn_patient_model_young_421NM")
print("Model saved.")
print("Loading model...")
model = DQN.load("dqn/single_patient/10mln/dqn_patient_model_young_421NM")
print("Model loaded.")


min_reward = -2190
max_reward = 365

def normalize_rewards(rewards, min_reward, max_reward):
    if max_reward == min_reward:
        return [0.5 for _ in rewards]
    normalized_rewards = [(reward - min_reward) / (max_reward - min_reward) for reward in rewards]
    return normalized_rewards

def run_simulation(agent, env, patient_data, csv_filename1, csv_filename2, num_episodes=30, num_steps=365, num_runs=5):
    all_episode_rewards = []
    all_episode_glucose_levels = []
    all_episode_motivation_levels = []
    all_step_rewards = np.zeros((num_runs, num_steps))
    all_step_glucose_levels = np.zeros((num_runs, num_steps))
    all_step_motivation_levels = np.zeros((num_runs, num_steps))
    terminated_counts = []

    for run in range(num_runs):
        episode_rewards = []
        episode_glucose_levels = []
        episode_motivation_levels = []
        step_rewards = np.zeros((num_steps,))
        step_glucose_levels = np.zeros((num_steps,))
        step_motivation_levels = np.zeros((num_steps,))
        terminated_count = 0

        with open(csv_filename1, mode='w', newline='') as file1, open(csv_filename2, mode='w', newline='') as file2:
            writer1 = csv.writer(file1)
            writer2 = csv.writer(file2)

            writer1.writerow([
                'Episode', 'Step', 'Action', 'Performed', 'Age', 'Years_T2DM', 'Physical_Activity',
                'Motivation', 'Glucose_Level', 'Reward'
            ])

            writer2.writerow([
                'Episode', 'Step', 'Action', 'Observation', 'Reward'
            ])

            for episode in range(num_episodes):
                env.reset_with_patient_data(patient_data)
                observation, _ = env.reset()
                total_reward = 0
                glucose_levels = []
                motivation_levels = []
                for step in range(num_steps):
                    if agent:
                        action, _ = agent.predict(observation, deterministic=False)
                        if isinstance(action, np.ndarray):
                            action = action.item()
                    else:
                        action = env.action_space.sample()

                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    glucose_levels.append(env.state['glucose_level'])
                    motivation_levels.append(env.state['motivation'])

                    writer1.writerow([
                        episode, step, action, info['action_performed'], env.state['age'], env.state['years_T2DM'],
                        env.state['physical_activity'], f"{env.state['motivation']:.1f}", f"{env.state['glucose_level']:.2f}",
                        reward
                    ])

                    writer2.writerow([
                        episode, step, action, observation, reward
                    ])

                    step_rewards[step] += reward
                    step_glucose_levels[step] += env.state['glucose_level']
                    step_motivation_levels[step] += env.state['motivation']
                    
                    if terminated or truncated:
                        terminated_count += 1
                        break

                episode_rewards.append(total_reward)
                episode_glucose_levels.append(np.mean(glucose_levels))
                episode_motivation_levels.append(np.mean(motivation_levels))

        step_rewards /= num_episodes
        step_glucose_levels /= num_episodes
        step_motivation_levels /= num_episodes

        all_episode_rewards.append(normalize_rewards(episode_rewards, min_reward, max_reward))
        all_episode_glucose_levels.append(episode_glucose_levels)
        all_episode_motivation_levels.append(episode_motivation_levels)
        all_step_rewards[run] = step_rewards
        all_step_glucose_levels[run] = step_glucose_levels
        all_step_motivation_levels[run] = step_motivation_levels
        terminated_counts.append(terminated_count)

    mean_rewards = np.mean(all_episode_rewards, axis=0)
    std_rewards = np.std(all_episode_rewards, axis=0)
    mean_glucose = np.mean(all_episode_glucose_levels, axis=0)
    mean_motivation = np.mean(all_episode_motivation_levels, axis=0)
    mean_step_rewards = np.mean(all_step_rewards, axis=0)
    mean_step_glucose = np.mean(all_step_glucose_levels, axis=0)
    mean_step_motivation = np.mean(all_step_motivation_levels, axis=0)
    
    return mean_rewards, std_rewards, mean_glucose, mean_motivation, mean_step_rewards, mean_step_glucose, mean_step_motivation, np.mean(terminated_counts)

#  save logs
dqn_csv_filename1 = 'dqn/single_patient/10mln/patient_simulation_dqn_young_421NM.csv'
dqn_csv_filename2 = 'dqn/single_patient/10mln/decisions_log_dqn_young_421NM.csv'
random_csv_filename1 = 'dqn/single_patient/10mln/patient_simulation_random_young_421NM.csv'
random_csv_filename2 = 'dqn/single_patient/10mln/decisions_log_random_young_421NM.csv'

# Evaluate
dqn_mean_rewards, dqn_std_rewards, dqn_mean_glucose, dqn_mean_motivation, dqn_mean_step_rewards, dqn_mean_step_glucose, dqn_mean_step_motivation, dqn_terminated_count = run_simulation(
    model, env, patient_data, dqn_csv_filename1, dqn_csv_filename2, num_episodes=30, num_steps=365, num_runs=5)
random_mean_rewards, random_std_rewards, random_mean_glucose, random_mean_motivation, random_mean_step_rewards, random_mean_step_glucose, random_mean_step_motivation, random_terminated_count = run_simulation(
    None, env, patient_data, random_csv_filename1, random_csv_filename2, num_episodes=30, num_steps=365, num_runs=5)

def plot_results(dqn_mean, dqn_std, random_mean, random_std, title, ylabel, filename):
    plt.figure(figsize=(14, 6))
    episodes = range(len(dqn_mean))
    
    plt.errorbar(episodes, dqn_mean, yerr=dqn_std, label='DQN Agent', fmt='-o', capsize=5)
    plt.errorbar(episodes, random_mean, yerr=random_std, label='Random Agent', fmt='-s', capsize=5)
    
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

plot_results(dqn_mean_rewards, dqn_std_rewards, random_mean_rewards, random_std_rewards, 'Normalized Average Reward per Episode', 'Reward', 'dqn/single_patient/10mln/rewards_comparison.png')
plot_results(dqn_mean_glucose, np.zeros_like(dqn_mean_glucose), random_mean_glucose, np.zeros_like(random_mean_glucose), 'Average Glucose Level per Episode', 'Glucose Level', 'dqn/single_patient/10mln/glucose_comparison.png')
plot_results(dqn_mean_motivation, np.zeros_like(dqn_mean_motivation), random_mean_motivation, np.zeros_like(random_mean_motivation), 'Average Motivation Level per Episode', 'Motivation Level', 'dqn/single_patient/10mln/motivation_comparison.png')

def plot_step_results(dqn_mean, dqn_std, random_mean, random_std, title, ylabel, filename):
    plt.figure(figsize=(14, 6))
    steps = range(len(dqn_mean))
    
    plt.errorbar(steps, dqn_mean, yerr=dqn_std, label='DQN Agent', fmt='-o', capsize=5)
    plt.errorbar(steps, random_mean, yerr=random_std, label='Random Agent', fmt='-s', capsize=5)
    
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

plot_step_results(dqn_mean_step_rewards, np.zeros_like(dqn_mean_step_rewards), random_mean_step_rewards, np.zeros_like(random_mean_step_rewards),
'Average Reward per Step', 'Reward', 'dqn/single_patient/10mln/step_rewards_comparison.png')
plot_step_results(dqn_mean_step_glucose, np.zeros_like(dqn_mean_step_glucose), random_mean_step_glucose, np.zeros_like(random_mean_step_glucose),
'Average Glucose Level per Step', 'Glucose Level', 'dqn/single_patient/10mln/step_glucose_comparison.png')
plot_step_results(dqn_mean_step_motivation, np.zeros_like(dqn_mean_step_motivation), random_mean_step_motivation, np.zeros_like(random_mean_step_motivation),
'Average Motivation Level per Step', 'Motivation Level', 'dqn/single_patient/10mln/step_motivation_comparison.png')