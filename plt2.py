import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns


dqn_pattern = 'dqn/single_patient/10mln/NM/*dqn*.csv'
random_pattern = 'dqn/single_patient/10mln/NM/*random*.csv'

dqn_files = glob.glob(dqn_pattern)
random_files = glob.glob(random_pattern)

def load_data(files, agent_type):
    all_data = []
    for file in files:
        data = pd.read_csv(file)
        data['Agent_Type'] = agent_type
        all_data.append(data)
    return pd.concat(all_data, ignore_index=True)


dqn_data = load_data(dqn_files, 'DQN')
random_data = load_data(random_files, 'Random')
combined_data = pd.concat([dqn_data, random_data], ignore_index=True)
age_bins = [0, 35, 60, 100]
age_labels = ['Young', 'Middle', 'Senior']
combined_data['Age_Group'] = pd.cut(combined_data['Age'], bins=age_bins, labels=age_labels, right=False)

def print_aggregated_metrics(data, metric, title):
    print(f"\n{title}:\n")
    grouped = data.groupby(['Agent_Type', 'Age_Group']).agg(
        Mean=(metric, 'mean'),
        Std=(metric, 'std'),
        Min=(metric, 'min'),
        Max=(metric, 'max')
    ).reset_index()
    print(grouped.to_string(index=False))


print_aggregated_metrics(combined_data, 'Reward', 'Reward Metrics')
print_aggregated_metrics(combined_data, 'Glucose_Level', 'Glucose Level Metrics')
print_aggregated_metrics(combined_data, 'Motivation', 'Motivation Metrics')

def plot_metric_combined(metric, ylabel, title_prefix, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    for agent_type in combined_data['Agent_Type'].unique():
        for age_group in age_labels:
            agent_data = combined_data[(combined_data['Agent_Type'] == agent_type) & (combined_data['Age_Group'] == age_group)]
            
            # Per Episode
            episode_means = agent_data.groupby('Episode')[metric].mean()
            episode_stds = agent_data.groupby('Episode')[metric].std()
            ax1.plot(episode_means.index, episode_means.values, label=f'{agent_type} {age_group} (Episode)')
            ax1.fill_between(episode_means.index, episode_means - episode_stds, episode_means + episode_stds, alpha=0.3)

            # Per Step
            step_means = agent_data.groupby('Step')[metric].mean()
            step_stds = agent_data.groupby('Step')[metric].std()
            ax2.plot(step_means.index, step_means.values, label=f'{agent_type} {age_group} (Step)')
            ax2.fill_between(step_means.index, step_means - step_stds, step_means + step_stds, alpha=0.3)

    ax1.set_title(f'{title_prefix} per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel(ylabel)
    ax1.legend()

    ax2.set_title(f'{title_prefix} per Step')
    ax2.set_xlabel('Step')
    ax2.set_ylabel(ylabel)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_metric_combined('Reward', 'Average Reward', 'Average Reward', 'dqn/single_patient/10mln/NM/average_reward.png')
plot_metric_combined('Glucose_Level', 'Average Glucose Level (mg/dL)', 'Average Glucose Level', 'dqn/single_patient/10mln/NM/average_glucose.png')
plot_metric_combined('Motivation', 'Average Motivation Level', 'Average Motivation Level', 'dqn/single_patient/10mln/NM/average_motivation.png')

sns.boxplot(data=combined_data, x='Age_Group', y='Motivation', hue='Agent_Type', fliersize=0,
            palette='Set2', linewidth=1.5,
            boxprops={'edgecolor': 'k', 'alpha': 0.5},
            medianprops={'color': 'black', 'linewidth': 2},
            whiskerprops={'color': 'black', 'linewidth': 1.5})


plt.title('Average Motivation Distribution')
plt.xlabel('Age Group')
plt.ylabel('Motivation level')
plt.legend(title='Agent Type')
plt.tight_layout()
plt.savefig('dqn/single_patient/10mln/NM/motivation_boxplot_outline.png', dpi=600)
plt.close()
