import gymnasium as gym
import numpy as np
import pandas as pd
import random
from gymnasium import spaces

class PatientEnvironment(gym.Env):
    def __init__(self, data_file="patients_data_grouped.csv"):
        super(PatientEnvironment, self).__init__()

        self.observation_space = spaces.Box(
            low=np.array([18, 0, 100, 80, 0]), 
            high=np.array([90, 5, 355, 200, 4]), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)

        self.patient_data = pd.read_csv(data_file)
        self.total_patients = len(self.patient_data)
        self.patient_states = [self._get_patient_state(i) for i in range(self.total_patients)]
        self.current_patient_index = 0
        self.current_step = 0
        self.actions_in_row = {i: None for i in range(self.total_patients)}
        self.count_in_row = {i: 0 for i in range(self.total_patients)}

    def reset(self, seed=None, patient_index=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        if patient_index is not None:
            self.current_patient_index = patient_index
        else:
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
            'glucose_level': patient.glucose_level,
            'weight': patient.weight,
            'motivation': patient.motivation,
        }

    def get_observation(self, state):
        glucose_meter_noise = random.uniform(0, 1)
        observation = np.array([
            state['age'], 
            state['physical_activity'],
            state['glucose_level'] + glucose_meter_noise, 
            state['weight'],
            state['motivation'],
        ], dtype=np.float32)

        #print(f"Observation: Age: {state['age']}, Physical Activity: {state['physical_activity']}, Motivation: {state['motivation']}, Glucose Level: {state['glucose_level'] + glucose_meter_noise}, Weight: {state['weight']}")
        
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)

        return observation

    def next_glucose(self, state, effect, action_performed):
        base_fluctuations = random.uniform(-5, 5)
        age_factor = 5 if state['age'] < 30 else (10 if 30 <= state['age'] < 65 else 7)
        improvement_factor = 1.0

        if state['age'] > 65:
            improvement_factor *= 0.7

        if action_performed:
            new_glucose_level = state['glucose_level'] + (base_fluctuations + age_factor) * improvement_factor + effect
        else:
            new_glucose_level = state['glucose_level'] + 1.2 * ((base_fluctuations + age_factor) * improvement_factor + effect)
        
        # Debugging new glucose level
        #print(f"Calculated new glucose level: {new_glucose_level}")

        return new_glucose_level

    def next_state(self, state, coach_action, patient_id):
        effect = 0
        action_performed = False

           # follow advice yes or no
        if coach_action in [1, 2]:  # Physical activities
            if state['physical_activity'] <= 2 or state['motivation'] < 2:
                action_performed = False
            elif state['age'] > 70 and coach_action == 2:
                action_performed = False
            else:
                action_performed = True
        elif coach_action in [0, 4]:  # Other actions
            if state['motivation'] < 2:
                action_performed = False
            elif state['physical_activity'] > 2:
                action_performed = False
            else:
                action_performed = True
        elif coach_action == 5:
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

            if coach_action == 0: #balance meals
                if self.count_in_row[patient_id] >= 5:
                    state['motivation'] = min(state['motivation'] + 0.1, 4)
                effect = -2 * age_factor
            elif coach_action == 1: #short walk
                if self.count_in_row[patient_id] >= 5:
                    state['motivation'] = min(state['motivation'] + 0.1, 4)
                effect = -3 * age_factor
            elif coach_action == 2: #long walk
                if self.count_in_row[patient_id] >= 5:
                    state['motivation'] = min(state['motivation'] + 0.1, 4)
                effect = -4 * age_factor
            elif coach_action == 3: #medication
                if self.count_in_row[patient_id] >= 5:
                    state['motivation'] = min(state['motivation'] - 0.1, 4)
                effect = -5 * age_factor
            elif coach_action == 4: #reduce sugar
                effect = -2 * age_factor
            elif coach_action == 5: #motivational
                effect = 0 
                state['motivation'] += 0.1
            elif coach_action == 6: #keep it up
                effect = 0

        state['glucose_level'] = max(100, self.next_glucose(state, effect, action_performed))
        state['motivation'] = max(0, min(state['motivation'], 4))
        state['weight'] = max(80, state['weight'])
        
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
            
            if coach_action == 5:
                if state['motivation'] < 2:
                    reward += 1
                else:
                    reward -= 1
            
            if coach_action == 6:
                if 100 <= state['glucose_level'] <= 125:
                    reward += 0.5
                else:
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
        
        
        self.current_step += 1
        
        return observation, reward, terminated, truncated, info

    def set_patient_index(self, patient_index):
        self.current_patient_index = patient_index