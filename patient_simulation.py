from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from Patient_env import PatientEnvironment
import csv
import numpy as np
from Patient_data import generate_random_patient, Patient

np.random.seed(123)

def get_patients(number_patients):
    return [generate_random_patient() for _ in range(number_patients)]

number_patients = 5
patients = get_patients(number_patients)

for i, patient in enumerate(patients):
    print(f'Patient {i+1}: Age={patient.age}, Years_T2DM={patient.years_T2DM}, '
          f'Physical Activity={patient.physical_activity}, Glucose Level={patient.glucose_level}, '
          f'Weight={patient.weight}, Motivation={patient.motivation}')

# Create the environment
env = PatientEnvironment()

# Check if the environment follows the gymnasium interface
check_env(env, warn=True)

# Create the PPO model
model = PPO("MlpPolicy", env, verbose=1)

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
        'Step', 'Action', 'Age', 'Years_T2DM', 'Physical_Activity',
        'Glucose_Level', 'Weight', 'Motivation', 'Reward'
    ])

    writer2.writerow([
        'Step', 'Action', 'Observation', 'Reward'
    ])

    # Simulate the environment
    num_steps = 10
    for patient in patients:
        observation, _ = env.reset(patient)
        for step in range(num_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            writer1.writerow([
                step + 1, action,
                env.state['age'], env.state['years_T2DM'], env.state['physical_activity'],
                round(env.state['glucose_level'], 2),
                round(env.state['weight'], 1),
                env.state['motivation'],
                reward
            ])
            writer2.writerow([
                step + 1, action, observation.tolist(), reward
            ])

print("Simulation completed, results saved in 'patient_simulation.csv' and 'decisions_log.csv'.")
