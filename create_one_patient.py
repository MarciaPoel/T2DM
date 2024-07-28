import pandas as pd
import numpy as np
import random

def generate_patient(age_group, base_values, noise_level=0.05):
    patient = {
        'age': np.random.randint(age_group[0], age_group[1] + 1),
        'years_T2DM': max(0, int(10 * (1 + np.random.uniform(-noise_level, noise_level)))),
        'physical_activity': max(0, min(5, int(np.random.uniform(0, 5) * (1 + np.random.uniform(-noise_level, noise_level))))),
        'glucose_level': round(max(100, min(355, 180 * (1 + np.random.uniform(-noise_level, noise_level)))), 2),
        'weight': round(max(100, int(110 * (1 + np.random.uniform(-noise_level, noise_level)))), 2),
        'motivation': round(max(0, min(4, np.random.uniform(0, 4) * (1 + np.random.uniform(-noise_level, noise_level)))), 2),
    }
    return patient

def single_patient(seed=650, age_group=(65,90)):
    np.random.seed(seed)
    random.seed(seed)

    base_values = {
        'years_T2DM': 9,
        'physical_activity': 3,
        'glucose_level': 180,
        'weight': 110,
        'motivation': 2,
    }

    patient = generate_patient(age_group, base_values)
    patient_df = pd.DataFrame([patient])
    patient_df.to_csv("single_old_600.csv", index=False)

if __name__ == "__main__":
    single_patient()
