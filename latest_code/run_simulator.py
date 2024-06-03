import csv
import numpy as np
from Patient_data import generate_random_patient, Patient
from Patient_env import PatientEnvironment

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

# Initialize CSV file
with open('patient_simulation.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Step', 'Action', 'Age', 'Years_T2DM', 'Physical_Activity', 
        'Glucose_Level', 'Weight', 'Motivation', 'Reward'
    ])

    # Simulate the environment
    num_steps = 10
    for patient in patients:
        observation = env.reset(patient)
        for step in range(num_steps):
            action = env.action_space.sample()  # Random
            observation, reward, done, info = env.step(action)
        # to csv
            writer.writerow([
                step + 1, action, 
                env.state['age'], env.state['years_T2DM'], env.state['physical_activity'],
                round(env.state['glucose_level'], 2),  
                round(env.state['weight'], 1),  
                env.state['motivation'],
                reward
                ])

print("Simulation completed, results saved in file'.")
