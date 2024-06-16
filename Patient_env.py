import gymnasium as gym
import numpy as np
import pandas as pd
import random
from gymnasium import spaces
from Patient_data import generate_random_patient, Patient

class PatientEnvironment(gym.Env):
    def __init__(self):
        super(PatientEnvironment, self).__init__()

        self.observation_space = spaces.Box(low=np.array([18, 70, 0]), high=np.array([90, 355, 5]), dtype=np.float32)
        self.action_space = spaces.Discrete(14)

        self.state = None
        self.patient_index = 0

        self.patient_data = pd.read_csv("patients_data.csv")
        self.total_patients = len(self.patient_data)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.patient_index = 0
        self.state = self._get_patient_state(self.patient_index)
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
        return self.get_observation(self.state), {}

    def get_observation(self, state):
        glucose_meter_noise = random.uniform(0, 1)
        observation = np.array([state['age'], state['glucose_level'] + glucose_meter_noise, state['physical_activity']], dtype=np.float32)
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        return observation

    def next_glucose(self, state, effect):
        base_fluctuations = 0
        if state['age'] < 30:
            base_fluctuations = 5
        elif 30 <= state['age'] < 65:
            base_fluctuations = 10
        else:
            base_fluctuations = 7

        weight_factor = 1.2 if state['weight'] > 100 else 1.0
        return state['glucose_level'] + base_fluctuations * weight_factor + effect

    def update_motivation(self, state, coach_action):
        motivation_increase = 0
        if coach_action in [0, 2, 3, 4, 10, 11]:
            motivation_increase = 0.1
        elif coach_action == 6:
            if state['motivation'] < 1:
                motivation_increase = 0.2
        
        noise = random.uniform(-0.1, 0.1)
        state['motivation'] = min(max(state['motivation'] + motivation_increase + noise, 0), 5)

    def next_state(self, coach_action):
        effect = 0
        
        if coach_action == 0:
            self.state['weight'] -= 0.1
            effect = -5
        elif coach_action == 1:
            if self.state['motivation'] > 0:
                effect = -10
                self.state['motivation'] -= 0.2
        elif coach_action == 2:
            effect = -3
            self.state['weight'] -= 0.2
        elif coach_action == 3:
            effect = -7
            self.state['weight'] -= 0.3
        elif coach_action == 4:
            effect = -12
            self.state['weight'] -= 0.5
        elif coach_action == 5:
            effect = -15
        elif coach_action == 6:
            if self.state['motivation'] < 1:
                self.state['motivation'] += 0.2
        elif coach_action == 7:
            self.state['weight'] -= 0.1
            effect = -3
        elif coach_action == 8:
            effect = -5
        elif coach_action == 9:
            self.state['glucose_level'] -= 1
            self.state['motivation'] += 0.1
        elif coach_action == 10:
            effect = -4
            self.state['motivation'] += 0.2
        elif coach_action == 11:
            self.state['weight'] -= 0.1
            effect = -4
        elif coach_action == 12:
            self.state['motivation'] += 0.2
        elif coach_action == 13:
            self.state['weight'] -= 0.2
            effect = -3

        self.state['glucose_level'] = self.next_glucose(self.state, effect)
        
        self.state['glucose_level'] = max(0, self.state['glucose_level'])
        self.state['motivation'] = max(0, min(self.state['motivation'], 5))
        self.state['weight'] = max(0, self.state['weight'])
        
        self.update_motivation(self.state, coach_action)

    def get_reward(self, state):
        reward = 0
        glucose_in_range = 100 < state['glucose_level'] < 180 

        if glucose_in_range and state['motivation'] > 3:
            reward += 2
        elif glucose_in_range:
            reward += 1
        else:
            reward -= 1

        if state['motivation'] < 2:
            reward -= 1
        elif state['motivation'] > 4:
            reward += 1

        if 70 <= state['weight'] <= 100:
            reward += 1

        if state['motivation'] >= 4 and glucose_in_range:
            reward += 1

        return reward

    def step(self, coach_action):
        self.next_state(coach_action)
        reward = self.get_reward(self.state)
        observation = self.get_observation(self.state)
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info