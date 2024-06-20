import random

class Patient:
    def __init__(self, age, years_T2DM, physical_activity, glucose_level, weight, motivation, stress_level):
        self.age = age
        self.years_T2DM = years_T2DM
        self.physical_activity = physical_activity
        self.glucose_level = glucose_level
        self.weight = weight
        self.motivation = motivation
        self.stress_level = stress_level

def generate_random_patient():
    age = random.randint(18, 90)
    years_T2DM = random.randint(1, 20)  # Assumption
    physical_activity = random.randint(0, 5)  # None to high
    glucose_level = round(random.uniform(100, 355),1)
    weight = round(random.uniform(50, 200), 1)
    motivation = random.randint(0, 5)  # TTM framework stages
    stress_level = random.uniform(0, 1)  # Random stress level
    return Patient(age, years_T2DM, physical_activity, glucose_level, weight, motivation, stress_level)

def set_seed(seed):
    random.seed(seed)