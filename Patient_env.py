import gymnasium as gym
import numpy as np
import pandas as pd
import random
from gymnasium import spaces

class PatientEnvironment(gym.Env):
    def __init__(self, data_file="patients_data_grouped_one.csv"):
        super(PatientEnvironment, self).__init__()

        self.observation_space = spaces.Box(
            low=np.array([18, 0, 0, 100, 80, 0]), 
            high=np.array([90, 5, 4, 355, 200, 1]), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)

        self.patient_data = pd.read_csv(data_file)
        self.total_patients = len(self.patient_data)
        self.patient_states = [self._get_patient_state(i) for i in range(self.total_patients)]
        self.current_patient_index = 0
        self.current_step = 0
        self.actions_in_row = {i: None for i in range(self.total_patients)}
        self.count_in_row = {i: 0 for i in range(self.total_patients)}

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.current_patient_index = 0
        self.current_step = 0
        self.patient_states = [self._get_patient_state(i) for i in range(self.total_patients)]
        self.actions_in_row = {i: None for i in range(self.total_patients)}
        self.count_in_row = {i: 0 for i in range(self.total_patients)}
        observation = self.get_observation(self.patient_states[self.current_patient_index])
        return observation, {}

    def _get_patient_state(self, index):
        patient = self.patient_data.iloc[index]
        return {
            'age': patient.age,
            'years_T2DM': patient.years_T2DM,
            'physical_activity': patient.physical_activity,
            'motivation': patient.motivation,
            'glucose_level': patient.glucose_level,
            'weight': patient.weight,
            'stress_level': patient.stress_level
        }

    def get_observation(self, state):
        glucose_meter_noise = random.uniform(0, 1)
        observation = np.array([
            state['age'], 
            state['physical_activity'],
            state['motivation'],
            state['glucose_level'] + glucose_meter_noise, 
            state['weight'],
            state['stress_level']
        ], dtype=np.float32)
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        return observation

    def next_glucose(self, state, effect, action_performed):
        base_fluctuations = random.uniform(-5, 5) 
        age_factor = 5 if state['age'] < 30 else (10 if 30 <= state['age'] < 65 else 7)
        weight_factor = 1.2 if state['weight'] > 100 else 1.0
        improvement_factor = 1.0

        if state['age'] > 65:
            improvement_factor *= 0.7
        if state['weight'] > 100:
            improvement_factor *= 0.8

        if action_performed:
            return state['glucose_level'] + (base_fluctuations + age_factor) * weight_factor * improvement_factor + effect
        else:
            return state['glucose_level'] + 1.2 * ((base_fluctuations + age_factor) * weight_factor * improvement_factor + effect)

    def next_state(self, state, coach_action, patient_id):
        effect = 0
        action_performed = False

        if coach_action in [1, 2, 3]:  # Physical activities
            if state['physical_activity'] <= 2 or state['motivation'] < 2:
                action_performed = False
            else:
                action_performed = True
        elif coach_action in [0, 4, 5, 7]:  # Other actions
            if state['motivation'] < 2:
                action_performed = False
            else:
                action_performed = True
        elif coach_action == 6:
            if state['motivation'] < 2:
                action_performed = True

        age_factor = 1.0
        if state['age'] < 30:
            age_factor = 1.5
        elif 30 <= state['age'] < 65:
            age_factor = 1.2
        

        if action_performed:
            if coach_action == self.actions_in_row[patient_id]:
                self.count_in_row[patient_id] += 1
            else:
                self.count_in_row[patient_id] = 1
                self.actions_in_row[patient_id] = coach_action

            if coach_action == 0: #diet
                if self.count_in_row[patient_id] >= 5:
                    state['motivation'] = min(state['motivation'] + 0.1, 4)
                effect = -2 * age_factor
            elif coach_action == 1: #short walk
                if self.count_in_row[patient_id] >= 5:
                    state['motivation'] = min(state['motivation'] + 0.1, 4)
                effect = -2 * age_factor
            elif coach_action == 2: #long walk
                if self.count_in_row[patient_id] >= 5:
                    state['motivation'] = min(state['motivation'] + 0.1, 4)
                effect = -4 * age_factor
            elif coach_action == 3: #gym_time
                if self.count_in_row[patient_id] >= 5:
                    state['motivation'] = min(state['motivation'] + 0.1, 4)
                effect = -5 * age_factor
            elif coach_action == 4: #medication
                if self.count_in_row[patient_id] >= 5:
                    state['motivation'] = max(state['motivation'] - 0.1, 4)
                effect = -6 * age_factor
            elif coach_action == 5: #less sugar
                effect = -2 * age_factor
            elif coach_action == 6: #motivational
                effect = 0 
                state['motivation'] += 0.1
            elif coach_action == 7: #relieve stress
                state['stress_level'] -= 0.1

        state['glucose_level'] = max(100, self.next_glucose(state, effect, action_performed))
        state['motivation'] = max(0, min(state['motivation'], 4))
        state['weight'] = max(80, state['weight'])
        state['stress_level'] = min(max(state['stress_level'], 0), 1)
        return action_performed, state

    def get_reward(self, state, coach_action, action_performed):
        reward = 0
        if action_performed:
            if 100 <= state['glucose_level'] <= 125:
                reward += 1
            else:
                reward -= 1

            if state['motivation'] > 2:
                reward += 0.2
            else:
                reward -= 0.2

            if state['motivation'] > 2 and coach_action == 6:
                reward -= 1
            elif state['motivation'] < 2 and coach_action == 6:
                reward += 1

            if state['stress_level'] > 0.7 and coach_action == 7:
                reward += 0.5
            elif state['stress_level'] < 0.5 and coach_action == 7:
                reward -= 0.5
        else:
            reward -= 1

        return reward

    def step(self, coach_action):
        state = self.patient_states[self.current_patient_index]
        action_performed, new_state = self.next_state(state, coach_action, self.current_patient_index)
        self.patient_states[self.current_patient_index] = new_state
        reward = self.get_reward(new_state, coach_action, action_performed)
        observation = self.get_observation(new_state)

        terminated = False
        truncated = False
        if new_state['glucose_level'] < 100 or new_state['glucose_level'] > 300:
            terminated = True

        info = {'action_performed': action_performed}
        
        self.current_patient_index = (self.current_patient_index + 1) % self.total_patients
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info