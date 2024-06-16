import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from Patient_data import generate_random_patient, Patient

class PatientEnvironment(gym.Env):
    def __init__(self):
        super(PatientEnvironment, self).__init__()

        self.observation_space = spaces.Box(low=np.array([18, 70, 0]), high=np.array([90, 355, 5]), dtype=np.float32)
        self.action_space = spaces.Discrete(14) 

        self.state = None
        self.seed_value = None

    def seed(self, seed=None):
        """Sets the seed for the environment's random number generator."""
        self.seed_value = seed
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, patient=None, seed=None, options=None):
        """Resets the environment to an initial state and returns an initial observation."""
        self.seed(seed)
        
        if patient is None:
            self.patient = generate_random_patient()
        else:
            self.patient = patient

        self.state = {
            'age': self.patient.age,
            'years_T2DM': self.patient.years_T2DM,
            'physical_activity': self.patient.physical_activity,
            'glucose_level': self.patient.glucose_level,
            'weight': self.patient.weight,
            'motivation': self.patient.motivation
        }
        return self.get_observation(self.state), {}

    def get_observation(self, state):
        """Generates partial observations with potential noise."""
        glucose_meter_noise = random.uniform(0, 1)
        observation = np.array([state['age'], state['glucose_level'] + glucose_meter_noise, state['physical_activity']], dtype=np.float32)
        # Clip the observation to be within the bounds
        observation = np.clip(observation, self.observation_space.low, self.observation_space.high)
        return observation

    def next_glucose(self, state, effect):
        base_fluctuations = 0
        if state['age'] < 30:
            base_fluctuations = 2
        elif 30 <= state['age'] < 65:
            base_fluctuations = 5
        else:
            base_fluctuations = 3

        weight_factor = 1.2 if state['weight'] > 100 else 1.0
        return state['glucose_level'] + base_fluctuations * weight_factor + effect

    def update_motivation(self, state, coach_action):
            motivation_increase = 0
            if action in [0, 2, 3, 4, 10, 11]:
                motivation_increase = 0.1
            elif action == 6:  
                if state['motivation'] < 1:
                    motivation_increase = 0.2
            
            # Motivation noise
            noise = random.uniform(-0.1, 0.1)
            state['motivation'] = min(max(state['motivation'] + motivation_increase + noise, 0), 5)

    def next_state(self, coach_action):
        effect = 0
        
        if coach_action == 0:  # manage_diet
            self.state['weight'] -= 0.1
            effect = -5
        elif coach_action == 1:  # glucose_down
            if self.state['motivation'] > 0:
                effect = -10
                self.state['motivation'] -= 0.2
        elif coach_action == 2:  # short_walk
            effect = -3
            self.state['weight'] -= 0.2
        elif coach_action == 3:  # long_walk
            effect = -7
            self.state['weight'] -= 0.3
        elif coach_action == 4:  # gym_time
            effect = -12
            self.state['weight'] -= 0.5
        elif coach_action == 5:  # medication
            effect = -15
        elif coach_action == 6:  # motivational_message
            if self.state['motivation'] < 1:
                self.state['motivation'] += 0.2
        elif coach_action == 7:  # increase_fiber
            self.state['weight'] -= 0.1
            effect = -3
        elif coach_action == 8:  # reduce_sugar
            effect = -5
        elif coach_action == 9:  # hydration
            self.state['glucose_level'] -= 1
            self.state['motivation'] += 0.1
        elif coach_action == 10:  # yoga
            effect = -4
            self.state['motivation'] += 0.2
        elif coach_action == 11:  # meal_planning
            self.state['weight'] -= 0.1
            effect = -4
        elif coach_action == 12:  # stress_relief_techniques
            self.state['motivation'] += 0.2
        elif coach_action == 13:  # increase_fruit_vegetables
            self.state['weight'] -= 0.2
            effect = -3

        self.state['glucose_level'] = self.next_glucose(self.state, effect)
        
        #Local bounds
        self.state['glucose_level'] = max(0, self.state['glucose_level'])
        self.state['motivation'] = max(0, min(self.state['motivation'], 5))
        self.state['weight'] = max(0, self.state['weight'])
        
        self.update_motivation(self.state, coach_action)

    def get_reward(self, state):
        reward = 0
        glucose_in_range = 100 < state['glucose_level'] < 180 

        if glucose_in_range and state['motivation'] > 3:
            reward += 2  # Good glucose and high motivation
        elif glucose_in_range:
            reward += 1  # Only glucose is good
        else:
            reward -= 1  # Bad glucose level

        if state['motivation'] < 2:
            reward -= 1  # Low motivation is bad
        elif state['motivation'] > 4:
            reward += 1  # High motivation is good

        if 70 <= state['weight'] <= 100:
            reward += 1  # Reward for maintaining a healthy weight

        # Reward adherence to good habits
        if state['motivation'] >= 4 and glucose_in_range:
            reward += 1  # Extra reward for high motivation and good glucose control

        return reward

    def step(self, coach_action):
        self.next_state(coach_action)
        reward = self.get_reward(self.state)
        observation = self.get_observation(self.state)
        terminated = False  # Define your termination condition if any
        truncated = False   # Define your truncation condition if any
        info = {}
        return observation, reward, terminated, truncated, info