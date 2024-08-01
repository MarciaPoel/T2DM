import gymnasium as gym
import numpy as np
import pandas as pd
import random
from gymnasium import spaces

class PatientEnvironment(gym.Env):
    def __init__(self, data_file="single_young_421.csv", log_file="log_single_young_421.csv"):
        super(PatientEnvironment, self).__init__()

        self.observation_space = spaces.Box(low=np.array([18, 0, 90, 80]), high=np.array([90, 5, 355, 200]), dtype=np.float32)
        self.action_space = spaces.Discrete(6)  
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

    def reset_with_patient_data(self, patient_data):
        self.state = {
            'age': patient_data['age'],
            'years_T2DM': patient_data['years_T2DM'],
            'physical_activity': patient_data['physical_activity'],
            'motivation': patient_data['motivation'],
            'glucose_level': patient_data['glucose_level'],
            'weight': patient_data['weight']
        }
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
            'weight': patient.weight
        }

    def get_observation(self, state):
        glucose_meter_noise = random.uniform(0, 1)
        observation = np.array([
            state['age'], 
            state['physical_activity'],
            state['glucose_level'] + glucose_meter_noise,
            state['weight'], 
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
            return state['glucose_level'] + 1.2 * ((random.uniform(1, 4) + age_factor) * improvement_factor)

    def next_state(self, coach_action):
        effect = 0
        action_performed = False

        # Determine if the advice is followed based on the current state
        if coach_action in [1, 2]:  # Physical activities
            if self.state['physical_activity'] <= 2:
                action_performed = False
            elif self.state['age'] > 70 and coach_action == 2:
                action_performed = False
            else:
                action_performed = True
        elif coach_action in [0, 3, 4]:  # Other actions
            if self.state['physical_activity'] > 2:
                action_performed = False
            else:
                action_performed = True
        # elif coach_action == 5:
        #     if self.state['motivation'] < 2:
        #         action_performed = True

        # Handle consecutive advice
        if action_performed:
            if coach_action == self.previous_action:
                self.advice_in_row += 1
            else:
                self.advice_in_row = 1
                self.previous_action = coach_action

            # Calculate the effect based on the action
            age_factor = 1.0
            if self.state['age'] < 30:
                age_factor = 1.5
            elif 30 <= self.state['age'] < 65:
                age_factor = 1.2
            else:
                age_factor = 1.0

            if coach_action == 0:  # balance meals
                effect = -3 * age_factor
            elif coach_action == 1:  # short walk
                effect = -4 * age_factor
            elif coach_action == 2:  # long walk
                effect = -6 * age_factor
            elif coach_action == 3:  # medication
                self.actions_performed[coach_action] += 1
                if self.advice_in_row == 4:
                    self.state['motivation'] -= 0.1
                effect = -7 * age_factor
            elif coach_action == 4:  # reduce sugar
                effect = -3 * age_factor
            elif coach_action == 5:  # motivational message
                effect = 0
                self.state['motivation'] += 0.1

            # Additional motivation adjustments
            if coach_action in [1, 2] and self.advice_in_row == 4:
                self.state['motivation'] = min(max(self.state['motivation'] + 0.1 + random.uniform(-0.1, 0.1), 0), 4)

        # Update glucose level based on the action performed
        self.state['glucose_level'] = self.next_glucose(self.state, effect, action_performed)
        self.state['glucose_level'] = max(100, self.state['glucose_level'])
        self.state['motivation'] = max(0, min(self.state['motivation'], 4))

        self.action_performed = action_performed
        return action_performed
   


    # def potential(self, state):
    #     # Potential function based on how close the glucose level is to the target range
    #     # Normalizing the penalty to keep the values small
    #     if 100 <= state['glucose_level'] <= 125:
    #         return 0  # Best case, in the target range
    #     elif state['glucose_level'] < 100:
    #         return (100 - state['glucose_level']) / 100  # Normalized penalty for being too low
    #     else:
    #         return (state['glucose_level'] - 125) / 100  # Normalized penalty for being too high


    # def get_reward(self, state, coach_action, action_performed):
    #     reward = 0
    #     if 100 <= state['glucose_level'] <= 125:
    #         reward += 1
    #     else:
    #         reward -= 1

    #     # Original reward
    #     original_reward = reward

    #     # Potential-based reward shaping
    #     gamma = 0.99  # Discount factor, typically same as used in the ppo
    #     next_state = self.state.copy()
    #     next_state['glucose_level'] = self.next_glucose(state, action_performed, coach_action)

    #     # Compute potential for current and next state
    #     potential_current = self.potential(state)
    #     potential_next = self.potential(next_state)

    #     # Augment reward with potential-based shaping
    #     reward = original_reward + gamma * potential_next - potential_current
        
    #     return reward

    def get_reward(self, state, coach_action, action_performed):
        reward = 0
        if action_performed:
            reward +=1

            if 100 <= state['glucose_level'] <= 125:
                reward += 0.5
            else:
                reward -= 0.5
        else:
            reward -= 1
        
        if state['glucose_level'] < 100 or state['glucose_level'] > 300:
            reward -= 5

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