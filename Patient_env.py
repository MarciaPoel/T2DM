import gymnasium as gym
import numpy as np
import pandas as pd
import random
from gymnasium import spaces
from Patient_data import generate_random_patient, Patient

class PatientEnvironment(gym.Env):
    def __init__(self):
        super(PatientEnvironment, self).__init__()

        self.observation_space = spaces.Box(low=np.array([18, 70, 0, 0]), high=np.array([90, 355, 5, 5]), dtype=np.float32)
        self.action_space = spaces.Discrete(8)  

        self.state = None
        self.patient_index = 0

        self.patient_data = pd.read_csv("patients_data_50.csv")
        self.total_patients = len(self.patient_data)

        self.actions_performed = {action: 0 for action in range(self.action_space.n)}
        self.previous_action = None
        self.advice_in_row = 0

    def reset(self, seed=None, patient_index=0):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.patient_index = patient_index
        self.state = self._get_patient_state(self.patient_index)
        self.actions_performed = {action: 0 for action in range(self.action_space.n)}
        return self.get_observation(self.state), {}

    def _get_patient_state(self, index):
        patient = self.patient_data.iloc[index]
        return {
            'age': patient.age,
            'years_T2DM': patient.years_T2DM,
            'physical_activity': patient.physical_activity,
            'glucose_level': patient.glucose_level,
            'weight': patient.weight,
            'motivation': patient.motivation
        }

    def next_patient(self):
        self.patient_index = (self.patient_index + 1) % self.total_patients
        self.state = self._get_patient_state(self.patient_index)
        self.actions_performed = {action: 0 for action in range(self.action_space.n)}
        self.previous_action = None
        self.advice_in_row = 0
        return self.get_observation(self.state), {}

    def get_observation(self, state):
        glucose_meter_noise = random.uniform(0, 1)
        observation = np.array([
            state['age'], 
            state['glucose_level'] + glucose_meter_noise, 
            state['physical_activity'],
            state['motivation']
        ], dtype=np.float32)
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        return observation

    def next_glucose(self, state, effect, follow_advice):
        base_fluctuations = random.uniform(-10, 10)  # Daily variations
        age_factor = 5 if state['age'] < 30 else (10 if 30 <= state['age'] < 65 else 7)
        weight_factor = 1.2 if state['weight'] > 100 else 1.0

        improvement_factor = 1.0
        if state['age'] > 65:
            improvement_factor *= 0.7
        if state['weight'] > 100:
            improvement_factor *= 0.8

        if follow_advice:
            return state['glucose_level'] + (base_fluctuations + age_factor) * weight_factor * improvement_factor + effect
        else:
            return state['glucose_level'] + 1.5 * ((base_fluctuations + age_factor) * weight_factor * improvement_factor + effect)

    def update_motivation(self, state, coach_action):
        motivation_increase = 0
        if coach_action in [0, 1, 2, 3, 4]:
            motivation_increase = 0.1
        elif coach_action == 6:  
            motivation_increase = 0.1 if state['motivation'] < 1 else 0.05
        else:
            if state['physical_activity'] <= 1 and state['age'] > 60:
                motivation_increase = -0.05

        noise = random.uniform(-0.1, 0.1)
        state['motivation'] = min(max(state['motivation'] + motivation_increase + noise, 0), 5)

    def next_state(self, coach_action):
        effect = 0
        follow_advice = False

        # follow advice yes or no
        if coach_action in [1, 2, 3]:  # Physical activities
            if self.state['physical_activity'] <= 1 or self.state['motivation'] < 2:
                follow_advice = False
            else:
                follow_advice = True
        elif coach_action in [0, 4, 5, 6, 7]:  # Other actions
            if self.state['motivation'] < 2 and coach_action != 6:
                follow_advice = False
            else:
                follow_advice = True

        # Differences age
        age_factor = 1.0
        if self.state['age'] < 30:
            age_factor = 1.5
        elif 30 <= self.state['age'] < 65:
            age_factor = 1.2
        else:
            age_factor = 1.0

        # Effects per advice
        if follow_advice:
            if coach_action == self.previous_action:
                self.advice_in_row += 1
            else:
                self.advice_in_row = 1
                self.previous_action = coach_action

            if coach_action == 0:  # manage_diet
                self.actions_performed[coach_action] += 1
                if self.advice_in_row == 3:
                    self.state['weight'] -= 0.1
                effect = -5 * age_factor
            elif coach_action == 1:  # short_walk
                self.actions_performed[coach_action] += 1
                if self.advice_in_row == 3:
                    self.state['weight'] -= 0.1
                effect = -5 * age_factor
            elif coach_action == 2:  # long_walk
                self.actions_performed[coach_action] += 1
                if self.advice_in_row == 3:
                    self.state['weight'] -= 0.2
                effect = -7 * age_factor
            elif coach_action == 3:  # gym_time
                self.actions_performed[coach_action] += 1
                if self.advice_in_row == 3:
                    self.state['weight'] -= 0.4
                effect = -10 * age_factor
            elif coach_action == 4:  # medication
                effect = -12 * age_factor
            elif coach_action == 5:  # reduce_sugar
                effect = -5 * age_factor
            elif coach_action == 6:  # motivational_message
                effect = 0 
                self.state['motivation'] += 0.1
            elif coach_action == 7:  # hydration
                self.actions_performed[coach_action] += 1
                if self.advice_in_row == 3:
                    self.state['weight'] -= 0.1
                self.state['glucose_level'] -= 1 * age_factor
                self.state['motivation'] += 0.1

 #      print(f"Action: {coach_action}, Action Performed: {follow_advice}, Effect: {effect}")

        self.state['glucose_level'] = self.next_glucose(self.state, effect, follow_advice)

        self.state['glucose_level'] = max(0, min(self.state['glucose_level'], 330))
        self.state['motivation'] = max(0, min(self.state['motivation'], 5))
        self.state['weight'] = max(0, self.state['weight'])

        self.update_motivation(self.state, coach_action)

    def get_reward(self, state, coach_action):
        reward = 0
        glucose_in_range = 100 <= state['glucose_level'] <= 180
        glucose_critical_low = state['glucose_level'] < 70
        glucose_critical_high = state['glucose_level'] > 300

        # glucose
        if glucose_in_range:
            reward += 5
        else:
            reward -= 5

        if glucose_critical_low or glucose_critical_high:
            reward -= 10

        # motivation
        if state['motivation'] > 4:
            reward += 1
        elif state['motivation'] < 2:
            reward -= 1  

        # 
        if state['motivation'] > 1 and coach_action == 6:
            reward -= 10
        elif state['motivation'] < 0 and coach_action == 6:
            reward += 10

        # weight
        if 70 <= state['weight'] <= 100:
            reward += 1
        elif state['weight'] < 50 or state['weight'] > 120:
            reward -= 1

        # good health
        if state['motivation'] >= 4 and glucose_in_range:
            reward += 1

        # extra
        sustained_action_bonus = 0
        for action, count in self.actions_performed.items():
            if count > 0 and count % 3 == 0:
                sustained_action_bonus += 0.5

        reward += sustained_action_bonus

        return reward

    def step(self, coach_action):
        self.next_state(coach_action)
        reward = self.get_reward(self.state, coach_action)
        observation = self.get_observation(self.state)
        terminated = False
        truncated = False

        # Terminate if glucose levels become critical
        if self.state['glucose_level'] < 70 or self.state['glucose_level'] > 300:
            terminated = True

        info = {}
        return observation, reward, terminated, truncated, info