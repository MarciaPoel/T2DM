import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns

dqn_pattern = 'dqn/age_group/10mln/NM/plot/*dqn*.csv'
random_pattern = 'dqn/age_group/10mln/NM/plot/*random*.csv'

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

def plot_metric_combined(metric, ylabel, title_prefix, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=set_size(fraction=1, subplots=(2,1)))

    for agent_type in combined_data['Agent_Type'].unique():
        for age_group in age_labels:
            agent_data = combined_data[(combined_data['Agent_Type'] == agent_type) & (combined_data['Age_Group'] == age_group)]
            
            # episode
            episode_means = agent_data.groupby('Episode')[metric].mean()
            episode_stds = agent_data.groupby('Episode')[metric].std()
            ax1.plot(episode_means.index, episode_means.values, label=f'{agent_type} {age_group}')
            ax1.fill_between(episode_means.index, episode_means - episode_stds, episode_means + episode_stds, alpha=0.3)

            # step
            step_means = agent_data.groupby('Step')[metric].mean()
            step_stds = agent_data.groupby('Step')[metric].std()
            ax2.plot(step_means.index, step_means.values, label=f'{agent_type} {age_group}')
            ax2.fill_between(step_means.index, step_means - step_stds, step_means + step_stds, alpha=0.3)

    ax1.set_title(f'{title_prefix} per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel(ylabel)

    ax2.set_title(f'{title_prefix} per Step')
    ax2.set_xlabel('Step')
    ax2.set_ylabel(ylabel)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    plt.close()

plot_metric_combined('Reward', 'Reward', 'Average Reward', 'dqn/age_group/10mln/NM/plot/average_rewardNM.png')
plot_metric_combined('Glucose_Level', 'Glucose Level (mg/dL)', 'Average Glucose Level', 'dqn/age_group/10mln/NM/plot/average_glucose.png')

sns.boxplot(data=combined_data, x='Age_Group', y='Motivation', hue='Agent_Type',
            palette='Set2', linewidth=1.5,
            boxprops={'edgecolor': 'k', 'alpha': 0.5},
            medianprops={'color': 'black', 'linewidth': 2},
            whiskerprops={'color': 'black', 'linewidth': 1.5})

plt.title('Average Motivation Distribution')
plt.xlabel('Age Group')
plt.ylabel('Motivation level')
plt.legend(title='Agent Type')
plt.tight_layout()
plt.savefig('dqn/age_group/10mln/NM/plot/motivation_boxplot.png', dpi=600)
plt.close()
