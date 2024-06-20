from Patient_data import generate_random_patient, set_seed
import pandas as pd
import os

def get_patients(number_patients, seed=456):
    print(f"Getting {number_patients} patients with seed: {seed}")
    set_seed(seed)
    return [generate_random_patient() for _ in range(number_patients)]

number_patients = 50

if not os.path.exists('patients_data_50.csv'):
    patients = get_patients(number_patients)

    # Convert patient data to DataFrame
    patient_data = {
        "age": [patient.age for patient in patients],
        "years_T2DM": [patient.years_T2DM for patient in patients],
        "physical_activity": [patient.physical_activity for patient in patients],
        "glucose_level": [patient.glucose_level for patient in patients],
        "weight": [patient.weight for patient in patients],
        "motivation": [patient.motivation for patient in patients],
        "stress_level": [patient.stress_level for patient in patients]
    }

    df = pd.DataFrame(patient_data)

    # Save DataFrame to CSV
    df.to_csv("patients_data_50.csv", index=False)

    print("Patient data saved to patients_data_50.csv")
else:
    print("Dataset already exists.")
    df = pd.read_csv("patients_data_50.csv")
    print(df)
