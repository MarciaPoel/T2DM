import pandas as pd
import numpy as np
import random

def generate_patient(age_group, base_values, glucose_variation='normal', noise_level=0.10):
    glucose_factor = {
        'lower': 0.8,
        'normal': 1.0,
        'higher': 1.2
    }[glucose_variation]

    patient = {
        'age': np.random.randint(age_group[0], age_group[1] + 1),
        'years_T2DM': max(0, int(base_values['years_T2DM'] * (1 + np.random.uniform(-noise_level, noise_level)))),
        'physical_activity': max(0, min(5, int(np.random.uniform(0, 5) * (1 + np.random.uniform(-noise_level, noise_level))))),
        'glucose_level': round(max(100, min(355, base_values['glucose_level'] * glucose_factor * (1 + np.random.uniform(-noise_level, noise_level)))), 2),
        'weight': round(max(100, base_values['weight'] * (1 + np.random.uniform(-noise_level, noise_level))), 2),
        'motivation': round(max(0, min(4, np.random.uniform(0, 4) * (1 + np.random.uniform(-noise_level, noise_level)))), 2),
    }
    return patient

def create_dataset(seed=421):
    np.random.seed(seed)
    random.seed(seed)

    age_groups = [(18, 30), (31, 60), (65, 85)]
    base_values = {
        'years_T2DM': 10,
        'physical_activity': 3,
        'glucose_level': 180,
        'weight': 110,
        'motivation': 2,
    }

    patients = []
    glucose_variations = ['normal', 'lower', 'higher']
    for age_group in age_groups:
        for variation in glucose_variations:
            for _ in range(5):
                patients.append(generate_patient(age_group, base_values, glucose_variation=variation))

    patient_df = pd.DataFrame(patients)
    patient_df.to_csv("patients2.csv", index=False)
    return patient_df

if __name__ == "__main__":
    patient_df = create_dataset()
    print(patient_df.head())
