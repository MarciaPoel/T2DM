import pandas as pd
import numpy as np
import random

def generate_patient(age_group, base_values, noise_level=0.05):
    patient = {
        'age': np.random.randint(age_group[0], age_group[1] + 1),
        'years_T2DM': max(0, int(10 * (1 + np.random.uniform(-noise_level, noise_level)))),
        'physical_activity': max(0, min(5, int(np.random.uniform(0, 5) * (1 + np.random.uniform(-noise_level, noise_level))))),
        'glucose_level': round(max(100, min(355, 150 * (1 + np.random.uniform(-noise_level, noise_level)))), 2),
        'weight': round(max(100, int(110 * (1 + np.random.uniform(-noise_level, noise_level)))), 2),
        'motivation': round(max(0, min(4, np.random.uniform(0, 4) * (1 + np.random.uniform(-noise_level, noise_level)))), 2),
        'stress_level': round(max(0, min(1, 0.5 * (1 + np.random.uniform(-noise_level, noise_level)))), 2)
    }
    return patient

def create_dataset(seed=421):
    np.random.seed(seed)
    random.seed(seed)

    age_groups = [(18, 30), (31, 50), (51, 70), (71, 90)]
    base_values = {
        'years_T2DM': 10,
        'physical_activity': 3,
        'glucose_level': 150,
        'weight': 110,
        'motivation': 2,
        'stress_level': 0.5
    }

    patients = []
    for age_group in age_groups:
        for _ in range(1):
            patients.append(generate_patient(age_group, base_values))

    patient_df = pd.DataFrame(patients)
    patient_df.to_csv("patients_data_grouped_one.csv", index=False)

if __name__ == "__main__":
    create_dataset()