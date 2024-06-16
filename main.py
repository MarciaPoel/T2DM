from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from Patient_env import PatientEnvironment
import csv
import numpy as np
from Patient_data import generate_random_patient, Patient
import matplotlib.pyplot as plt

np.random.seed(123)

def get_patients(number_patients):
    return [generate_random_patient() for _ in range(number_patients)]

number_patients = 10
patients = get_patients(number_patients)

for i, patient in enumerate(patients):
    print(f'Patient {i+1}: Age={patient.age}, Years_T2DM={patient.years_T2DM}, '
          f'Physical Activity={patient.physical_activity}, Glucose Level={patient.glucose_level}, '
          f'Weight={patient.weight}, Motivation={patient.motivation}')

# Create the environment
env = PatientEnvironment()

# Check if the environment follows the gymnasium interface
check_env(env, warn=True)

# Create the PPO model with adjusted parameters for more exploration
model = PPO("MlpPolicy", env, verbose=1, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, learning_rate=3e-4)

# Train the agent
model.learn(total_timesteps=10000)

# Save the model
model.save("ppo_patient")

# Load the trained model
model = PPO.load("ppo_patient")

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

    # Simulate the environment
    num_steps = 1000
    for patient_index, patient in enumerate(patients):
        observation, _ = env.reset(patient=patient)
        episode_rewards = []
        for step in range(num_steps):
            action, _ = model.predict(observation, deterministic=False)  # Ensure exploration by using deterministic=False
            observation, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
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

print("Simulation completed, results saved in 'patient_simulation.csv' and 'decisions_log.csv'.")

# Plotting the average reward per episode
plt.plot(all_rewards)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward per Episode Over Time')
plt.savefig('learning_curve.png')
plt.show()
