import random
import datetime

class Patient:
    def __init__(self, age, years_T2DM, physical_activity, glucose_level, weight, motivation):
        self.age = age
        self.years_T2DM = years_T2DM
        self.physical_activity = physical_activity
        self.glucose_level = glucose_level
        self.weight = weight
        self.motivation = motivation
        self.log_entries = []

    def receive_advice(self):
        date = datetime.datetime.now()
        for _ in range(7):  # Simulate for a week
            for _ in range(3):  # Three times a day
                date += datetime.timedelta(hours=6) #every 6 hours - now random start time
                advice_response = self.evaluate_advice()
                self.log_entries.append({
                    "date_time": date,
                    "glucose_level": self.glucose_level,
                    "physical_activity": self.physical_activity,
                    "motivation": self.motivation,
                    "weight": self.weight,
                    "advice_response": advice_response
                })
                self.adjust_advice(advice_response)

        # response is always true -> motivation go up
        if all(entry["advice_response"] for entry in self.log_entries):
            self.motivation += 1
            self.motivation = min(self.motivation, 5) 
    
    
    def evaluate_advice(self):
        # Response patient def - for now random.
        return random.choice([True, False])

    def adjust_advice(self, advice_response):
        if advice_response:
            # If advice is effective, decrease glucose level
            self.glucose_level -= random.uniform(5, 15)
        else:
            # If advice is not effective, increase glucose level
            self.glucose_level += random.uniform(0, 10)
        self.glucose_level = max(70, min(self.glucose_level, 300))

    def write_log_to_file(self, file_name):
        with open(file_name, 'w') as file:
            for entry in self.log_entries:
                file.write(f"Date/Time: {entry['date_time'].strftime('%Y-%m-%d %H:%M')}, Glucose Level: {entry['glucose_level']}, Weight: {entry['weight']}, Physical Activity: {entry['physical_activity']}, Motivation: {entry['motivation']}, Advice Response: {entry['advice_response']}\n")

def generate_random_patient():
    age = random.randint(18, 90)
    years_T2DM = random.randint(1, 20)  # Assumption
    physical_activity = random.randint(0, 5)  # None to high
    glucose_level = random.uniform(70, 355)
    weight = round(random.uniform(50, 150), 1)
    motivation = random.randint(0, 5)  # TTM framework stages
    return Patient(age, years_T2DM, physical_activity, glucose_level, weight, motivation)


# Generate randomized patients
patients = [generate_random_patient() for _ in range(2)]

for i, patient in enumerate(patients):
    patient.receive_advice()
    file_name = f"patient_{i+1}_log.txt"
    patient.write_log_to_file(file_name)