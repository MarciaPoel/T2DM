import csv
from Patient_env import PatientEnvironment

# Create the environment
env = PatientEnvironment()

# Initialize CSV file
with open('patient_simulation.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow([
        'Step', 'Action', 'Age', 'Years_T2DM', 'Physical_Activity', 
        'Glucose_Level', 'Weight', 'Motivation', 'Reward'
    ])

    # Reset the environment to get the initial state
    observation = env.reset()

    # Simulate the environment
    num_steps = 10
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
