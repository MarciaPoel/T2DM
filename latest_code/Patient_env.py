import gym
import numpy as np
import random
from gym import spaces
from Patient_data import generate_random_patient, Patient

class PatientEnvironment(gym.Env):
    def __init__(self):
        super(PatientEnvironment, self).__init__()

        self.observation_space = spaces.Dict({
            'glucose_level': spaces.Box(low=70, high=300, shape=(1,), dtype=float),
            'weight': spaces.Box(low=50, high=150, shape=(1,), dtype=float),
            'age': spaces.Box(low=18, high=90, shape=(1,), dtype=np.int64),
            'physical_activity': spaces.Discrete(6),
            'motivation': spaces.Discrete(6)
        })

        self.action_space = spaces.Discrete(3)  # manage_diet, hit_the_gym, glucose_down

        self.reset()  # Initialize state

    def reset(self):
        """Resets the environment to an initial state (toestand patient) and returns an initial observation (stukje patient-actie daarop uitvoeren)."""
        self.patient = generate_random_patient()
        self.state = {
            'age': self.patient.age,
            'years_T2DM': self.patient.years_T2DM,
            'physical_activity': self.patient.physical_activity,
            'glucose_level': self.patient.glucose_level,
            'weight': self.patient.weight,
            'motivation': self.patient.motivation
        }
        return self.get_observation(self.state)

    def get_observation(self, state):
        """Generates partial observations with potential noise."""
        agent_observation = ['age', 'glucose_level', 'physical_activity']
        observation = {}
        for variable in agent_observation:
            if variable == 'glucose_level':
                glucose_meter_noise = random.uniform(0, 1)  # does noise impact learning performace
                observation[variable] = state[variable] + glucose_meter_noise
            else:
                observation[variable] = state[variable]
        return observation

    def next_glucose(self, state):
        if state['age'] < 30:
            return state['glucose_level'] + 5
        elif 30 < state['age'] < 60:
            return state['glucose_level'] + 10
        else:
            return state['glucose_level'] + 5

    def next_state(self, state, coach_action):
        if coach_action == 0:  # manage_diet
            state['motivation'] -= 0.3
            state['glucose_level'] = self.next_glucose(state)
            state['weight'] += 1
            
        elif coach_action == 1:  # glucose_down
            if state['motivation'] > 0:
                state['glucose_level'] -= 10 
                state['motivation'] -= 0.2 
            else:   #hit_the_gym
                if state['physical_activity'] > 3:
                    state['glucose_level'] = self.next_glucose(state)
                    state['weight'] -= 0.2 

    def get_reward(self, state, agent_action):
        if 70 < state['glucose_level'] < 100:
            return 1
        elif state['motivation'] > 0:
            return 0.5
        else:
            return -1

    def step(self, coach_action):
        self.next_state(self.state, coach_action)
        reward = self.get_reward(self.state, coach_action)
        observation = self.get_observation(self.state)
        done = False  
        info = {}
        return observation, reward, done, info
