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
import matplotlib
import seaborn as sns

file_path = "patients.csv"
patients_df = pd.read_csv(file_path)

def categorize_age(age):
    if age < 30:
        return 'Young'
    elif 30 <= age < 60:
        return 'Middle'
    else:
        return 'Senior'

patients_df['AgeGroup'] = patients_df['age'].apply(categorize_age)

print("Initializing environment...")
env = PatientEnvironment(data_file=file_path)

total_timesteps = 10_000_000
model = DQN("MlpPolicy", env, verbose=1,
    learning_rate=0.0001,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.1,
    exploration_fraction=0.5
)

os.makedirs("dqn/age_group/10mln/NM/", exist_ok=True)
print("Learning model...")

class ProgressCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Step: {self.n_calls}, Time elapsed: {self.model.num_timesteps}")
        return True

progress_callback = ProgressCallback(check_freq=10000)

#Training Phase
model.learn(total_timesteps=total_timesteps, callback=progress_callback)
print("Model learning completed.")
model.save("dqn/age_group/10mln/M/dqn_patient_model")
print("Model saved.")
print("Loading model...")
model = DQN.load("dqn/age_group/10mln/NM/dqn_patient_model")
print("Model loaded.")

min_reward = -2190  
max_reward = 365  

def normalize_rewards(rewards, min_reward, max_reward):
    if max_reward == min_reward:
        return [0.5 for _ in rewards] 
    normalized_rewards = [(reward - min_reward) / (max_reward - min_reward) for reward in rewards]
    return normalized_rewards

def evaluate_agent(agent, env, patient_data, episodes=30, max_steps=365, num_runs=5):
    all_episode_rewards = []
    all_episode_glucose_levels = []
    all_episode_motivation_levels = []
    all_step_rewards = np.zeros((num_runs, max_steps))
    all_step_glucose_levels = np.zeros((num_runs, max_steps))
    all_step_motivation_levels = np.zeros((num_runs, max_steps))
    
    for run in range(num_runs):
        episode_rewards = []
        episode_glucose_levels = []
        episode_motivation_levels = []
        step_rewards = np.zeros((max_steps,))
        step_glucose_levels = np.zeros((max_steps,))
        step_motivation_levels = np.zeros((max_steps,))
        
        for episode in range(episodes):
            env.reset_with_patient_data(patient_data)
            obs, _ = env.reset()
            total_reward = 0
            total_glucose = 0
            total_motivation = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                if agent:
                    action, _ = agent.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                if isinstance(action, np.ndarray):
                    action = action.item()  
                elif not isinstance(action, (tuple, list)):
                    action = int(action) 

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                total_glucose += env.state['glucose_level']
                total_motivation += env.state['motivation']
                step_rewards[steps] += reward
                step_glucose_levels[steps] += env.state['glucose_level']
                step_motivation_levels[steps] += env.state['motivation']
                steps += 1
                
            episode_rewards.append(total_reward)
            episode_glucose_levels.append(total_glucose / steps)
            episode_motivation_levels.append(total_motivation / steps)
        
        all_episode_rewards.append(normalize_rewards(episode_rewards, min_reward, max_reward))
        all_episode_glucose_levels.append(episode_glucose_levels)
        all_episode_motivation_levels.append(episode_motivation_levels)
        all_step_rewards[run] = step_rewards / episodes
        all_step_glucose_levels[run] = step_glucose_levels / episodes
        all_step_motivation_levels[run] = step_motivation_levels / episodes

    mean_rewards = np.mean(all_episode_rewards, axis=0)
    std_rewards = np.std(all_episode_rewards, axis=0)
    mean_glucose = np.mean(all_episode_glucose_levels, axis=0)
    mean_motivation = np.mean(all_episode_motivation_levels, axis=0)
    mean_step_rewards = np.mean(all_step_rewards, axis=0)
    mean_step_glucose = np.mean(all_step_glucose_levels, axis=0)
    mean_step_motivation = np.mean(all_step_motivation_levels, axis=0)
    
    return mean_rewards, std_rewards, mean_glucose, mean_motivation, mean_step_rewards, mean_step_glucose, mean_step_motivation

age_groups = ['Young', 'Middle', 'Senior']
results = {
    group: {
        'dqn_rewards': [], 'random_rewards': [], 'dqn_glucose_levels': [], 'random_glucose_levels': [], 
        'dqn_motivation_levels': [], 'random_motivation_levels': [], 'dqn_step_rewards': [], 'random_step_rewards': [], 
        'dqn_step_glucose': [], 'random_step_glucose': [], 'dqn_step_motivation': [], 'random_step_motivation': []} 
    for group in age_groups}


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
        action_summary = {action: 0 for action in range(env.action_space.n)}
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
                    
                    action_summary[action] += 1

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

for i, patient_data in patients_df.iterrows():
    age_group = patient_data['AgeGroup']
    print(f"Running simulation for patient {i+1}/{len(patients_df)} in age group {age_group}...")

    # save logs
    dqn_csv_filename1 = f'dqn/age_group/10mln/NM/patient_simulation_dqn_{age_group}_{i+1}.csv'
    dqn_csv_filename2 = f'dqn/age_group/10mln/NM/decisions_log_dqn_{age_group}_{i+1}.csv'
    random_csv_filename1 = f'dqn/age_group/10mln/NM/patient_simulation_random_{age_group}_{i+1}.csv'
    random_csv_filename2 = f'dqn/age_group/10mln/NM/decisions_log_random_{age_group}_{i+1}.csv'
    
    # evaluate
    dqn_mean_rewards, dqn_std_rewards, dqn_mean_glucose, dqn_mean_motivation, dqn_mean_step_rewards, dqn_mean_step_glucose, dqn_mean_step_motivation, dqn_terminated_count = run_simulation(
        model, env, patient_data, dqn_csv_filename1, dqn_csv_filename2, num_episodes=30, num_steps=365, num_runs=5)
    results[age_group]['dqn_rewards'].append(dqn_mean_rewards)
    results[age_group]['dqn_glucose_levels'].append(dqn_mean_glucose)
    results[age_group]['dqn_motivation_levels'].append(dqn_mean_motivation)
    results[age_group]['dqn_step_rewards'].append(dqn_mean_step_rewards)
    results[age_group]['dqn_step_glucose'].append(dqn_mean_step_glucose)
    results[age_group]['dqn_step_motivation'].append(dqn_mean_step_motivation)
    
    random_mean_rewards, random_std_rewards, random_mean_glucose, random_mean_motivation, random_mean_step_rewards, random_mean_step_glucose, random_mean_step_motivation, random_terminated_count = run_simulation(
        None, env, patient_data, random_csv_filename1, random_csv_filename2, num_episodes=30, num_steps=365, num_runs=5)
    results[age_group]['random_rewards'].append(random_mean_rewards)
    results[age_group]['random_glucose_levels'].append(random_mean_glucose)
    results[age_group]['random_motivation_levels'].append(random_mean_motivation)
    results[age_group]['random_step_rewards'].append(random_mean_step_rewards)
    results[age_group]['random_step_glucose'].append(random_mean_step_glucose)
    results[age_group]['random_step_motivation'].append(random_mean_step_motivation)

def set_size(width=345, fraction=0.5, subplots=(1,1)):
    fig_width_pt = width * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)

def plot_results(results, episode_metric, step_metric, title, ylabel_episode, ylabel_step, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(fraction=1, subplots=(2,1)))
    
    # episode
    for agent_type in ['dqn', 'random']:
        for age_group in age_groups:
            avg_episode_values = np.mean(results[age_group][f'{agent_type}_{episode_metric}'], axis=0)
            std_episode_values = np.std(results[age_group][f'{agent_type}_{episode_metric}'], axis=0)
            ax1.plot(range(len(avg_episode_values)), avg_episode_values, label=f'{age_group} - {agent_type.upper()}')
            ax1.fill_between(range(len(avg_episode_values)), 
                             avg_episode_values - std_episode_values, 
                             avg_episode_values + std_episode_values, 
                             alpha=0.3)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel(ylabel_episode)
    ax1.set_title(f'{title} per Episode')

    # step
    for agent_type in ['dqn', 'random']:
        for age_group in age_groups:
            avg_step_values = np.mean(results[age_group][f'{agent_type}_{step_metric}'], axis=0)
            std_step_values = np.std(results[age_group][f'{agent_type}_{step_metric}'], axis=0)
            ax2.plot(range(len(avg_step_values)), avg_step_values, label=f'{age_group} - {agent_type.upper()}')
            ax2.fill_between(range(len(avg_step_values)), avg_step_values - std_step_values, avg_step_values + std_step_values, alpha=0.3)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel(ylabel_step)
    ax2.set_title(f'{title} per Step')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()


plot_results(results, 'glucose_levels', 'step_glucose','Average Glucose Level', 'Glucose Level (mg/dL)','Glucose Level (mg/dL)','dqn/age_group/10mln/NM/plot/glucose_pool.png')
plot_results(results, 'rewards', 'step_rewards', 'Average Reward', 'Reward', 'Reward','dqn/age_group/10mln/NM/plot/reward_pool.png')

def plot_box_plots(results, metric, title, ylabel, filename):
    data = []
    for agent_type in ['dqn', 'random']:
        for age_group in age_groups:
            flat_data = [item for sublist in results[age_group][f'{agent_type}_{metric}'] for item in sublist]
            data.extend([(agent_type.upper(), age_group, value) for value in flat_data])
    
    df = pd.DataFrame(data, columns=['Agent Type', 'Age Group', 'Value'])
    
    plt.figure(figsize=set_size(fraction=1))
    
    sns.boxplot(x='Age Group', y='Value', hue='Agent Type', data=df,
            palette='Set2', linewidth=1.5,
            boxprops={'edgecolor': 'k', 'alpha': 0.5},
            medianprops={'color': 'black', 'linewidth': 1},
            whiskerprops={'linewidth': 1.5},
            capprops={'linewidth': 1.5})

    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

plot_box_plots(results, 'rewards', 'Average Reward Distribution', 'Reward', 'dqn/age_group/10mln/NM/plot/boxplot_reward.png')
plot_box_plots(results, 'motivation_levels', 'Average Motivation Level Distribution', 'Motivation Level', 'dqn/age_group/10mln/NM/plot/boxplot_motivation.png')
