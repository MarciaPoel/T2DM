import matplotlib.pyplot as plt
from PIL import Image

# Load the images
glucose_img = Image.open("dqn/age_group/10mln/aggregated_glucose_comparison.png")
glucoseNM_img = Image.open("dqn/age_group/10mln/NM/aggregated_glucose_comparison.png")
glucose_step = Image.open("dqn/age_group/10mln/aggregated_step_glucose_comparison.png")
glucoseNM_step = Image.open("dqn/age_group/10mln/NM/aggregated_step_glucose_comparison.png")


motivation_box_img = Image.open("dqn/age_group/10mln/boxplot_motivation_comparison.png")
motivationNM_box_img = Image.open("dqn/age_group/10mln/NM/boxplot_motivation_comparison.png")

# reward_box_img = Image.open("dqn/age_group/10mln/boxplot_rewards_comparison.png")
# rewardNM_box_img = Image.open("dqn/age_group/10mln/NM/boxplot_rewards_comparison.png")

medium_font = 12
large_font = 14

# Create a new figure
fig, axs = plt.subplots(2, 1, figsize=(6, 5))

plt.rc('font', size=medium_font)
plt.rc('axes', labelsize=large_font)
plt.rc('xtick', labelsize=large_font)
plt.rc('ytick', labelsize=large_font)
# Plot the images in the respective positions
axs[0].imshow(motivation_box_img)
axs[0].axis('off')

# axs[0, 1].imshow(glucoseNM_img)
# axs[0, 1].axis('off')

axs[1].imshow(motivationNM_box_img)
axs[1].axis('off')
#axs[0, 2].set_title("Average Reward per Episode")

# axs[1, 1].imshow(glucoseNM_step)
# axs[1, 1].axis('off')
#axs[1, 2].set_title("Average Reward per Step")

# Adjust the layout, using more precise spacing
plt.subplots_adjust(wspace=0.1, hspace=0.2)
plt.tight_layout()
plt.savefig("dqn/age_group/10mln/NM/box_motivation.png", dpi=600)
plt.show()
