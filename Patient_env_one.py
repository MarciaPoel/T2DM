import gymnasium as gym
import numpy as np
import pandas as pd
import random
from gymnasium import spaces

class PatientEnvironment(gym.Env):
    def __init__(self, data_file="single_patient_middle.csv", log_file="simulation_log.csv"):
        super(PatientEnvironment, self).__init__()

        self.observation_space = spaces.Box(low=np.array([18, 0, 0, 90]), high=np.array([90, 5, 4, 355]), dtype=np.float32)
        self.action_space = spaces.Discrete(7)  
        self.state = None
        self.patient_index = 0
        self.patient_data = pd.read_csv(data_file)
        self.total_patients = len(self.patient_data)
        self.actions_performed = {action: 0 for action in range(self.action_space.n)}
        self.previous_action = None
        self.advice_in_row = 0
        self.log_file = log_file
        self.logs = []


    def reset(self,seed = None, patient_index=0):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.patient_index = 0
        self.state = self._get_patient_state(self.patient_index)
        self.actions_performed = {action: 0 for action in range(self.action_space.n)}
        return self.get_observation(self.state), {}

    def _get_patient_state(self, index):
        patient = self.patient_data.iloc[index]
        return {
            'age': patient.age,
            'years_T2DM': patient.years_T2DM,
            'physical_activity': patient.physical_activity,
            'motivation': patient.motivation,
            'glucose_level': patient.glucose_level,
        }

    def get_observation(self, state):
        glucose_meter_noise = random.uniform(0, 1)
        observation = np.array([
            state['age'], 
            state['physical_activity'],
            state['motivation'],
            state['glucose_level'] + glucose_meter_noise, 
        ], dtype=np.float32)
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        return observation

    def next_glucose(self, state, effect, action_performed):
        base_fluctuations = random.uniform(-5, 5)  # Daily variations
        age_factor = 2 if state['age'] < 30 else (5 if 30 <= state['age'] < 65 else 3)

        improvement_factor = 1.0
        if state['age'] > 65:
            improvement_factor *= 0.7

        if action_performed:
            return state['glucose_level'] + (base_fluctuations + age_factor) * improvement_factor + effect
        else:
            return state['glucose_level'] + 1.2 * ((base_fluctuations + age_factor) * improvement_factor + effect)

    def next_state(self, coach_action):
        effect = 0
        action_performed = False

        # follow advice yes or no
        if coach_action in [1, 2]:  # Physical activities
            if self.state['physical_activity'] <= 2 or self.state['motivation'] < 2:
                action_performed = False
            elif self.state['age'] > 70 and coach_action == 2:
                action_performed = False
            else:
                action_performed = True
        elif coach_action in [0, 4, 5, 7]:  # Other actions
            if self.state['motivation'] < 2:
                action_performed = False
            elif self.state['physical_activity'] > 2:
                action_performed = False
            else:
                action_performed = True
        elif coach_action == 6:
            if self.state['motivation'] < 2:
                action_performed = True

        # Differences age
        age_factor = 1.0
        if self.state['age'] < 30:
            age_factor = 1.5
        elif 30 <= self.state['age'] < 65:
            age_factor = 1.2
        else:
            age_factor = 1.0

        # Effects per advice
        if action_performed:
            if coach_action == self.previous_action:
                self.advice_in_row += 1
            else:
                self.advice_in_row = 1
                self.previous_action = coach_action

            if coach_action == 0:  # balance meals
                effect = -2 * age_factor
            elif coach_action == 1:  # short walk
                effect = -3 * age_factor
            elif coach_action == 2:  # long walk
                effect = -4 * age_factor
            elif coach_action == 3:  # medication
                self.actions_performed[coach_action] += 1
                if self.advice_in_row == 3:
                    self.state['motivation'] -= 0.1
                effect = -6 * age_factor
            elif coach_action == 4:  # reduce sugar
                effect = -2 * age_factor
            elif coach_action == 5:  # motivational message
                effect = 0 
                self.state['motivation'] += 0.1
            elif coach_action == 6: #keep it up
                effect = 0
                                        
        if coach_action in [1, 2] and self.advice_in_row == 3:
            self.state['motivation'] = min(max(self.state['motivation'] + 0.1 + random.uniform(-0.1, 0.1), 0), 4)
        elif coach_action == 6 and self.advice_in_row == 5:
            self.state['motivation'] = min(max(self.state['motivation'] + 0.1 + random.uniform(-0.1, 0.1), 0), 4)


 #      print(f"Action: {coach_action}, Action Performed: {action_performed}, Effect: {effect}")

        self.state['glucose_level'] = self.next_glucose(self.state, effect, action_performed)

        self.state['glucose_level'] = max(100, self.state['glucose_level'])
        self.state['motivation'] = max(0, min(self.state['motivation'], 4))
        
        self.action_performed = action_performed
        return action_performed
        
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
        
            if state['motivation'] > 2 and coach_action == 5:
                reward -= 1
            elif state['motivation'] < 2 and coach_action == 5:
                reward += 1
            
            if 100 <= state['glucose_level'] <= 125 and coach_action == 6:
                reward += 0.5
            if 100 <= state['glucose_level'] <= 125 and coach_action == 6:
                reward -= 0.5

            # # extra
            # sustained_action_bonus = sum(0.5 for count in self.actions_performed.values() if count > 0 and count % 4 == 0)
            # reward += sustained_action_bonus
        else:
            reward -= 1

        return reward

    def step(self, coach_action):
        action_performed = self.next_state(coach_action)
        self.next_state(coach_action)
        reward = self.get_reward(self.state, coach_action, action_performed)
        observation = self.get_observation(self.state)
        truncated = False
        if self.state['glucose_level'] < 100 or self.state['glucose_level'] > 300:
            terminated = True
        else:
            terminated = False

        info = {'action_performed': self.action_performed}
        return observation, reward, terminated, truncated, info
