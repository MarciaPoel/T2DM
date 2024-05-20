import gym
from gym import spaces
from patient import generate_random_patient

class EnvironmentPatient(gym.Env):
    def __init__(self):
        super(EnvironmentPatient, self).__init__()

        self.observation_space = spaces.Dict({
            'glucose_level': spaces.Box(low=70, high=300, shape=(1,), dtype=float),
            'weight': spaces.Box(low=50, high=150, shape=(1,), dtype=float),
            'physical_activity': spaces.Discrete(6),
            'motivation': spaces.Discrete(6)
        })

        self.action_space = spaces.Discrete(2)  # hoeveel acties - nu 2 acties (advies 1 & 2)

        # Initialize patient
        self.patient = generate_random_patient()

    def step(self, action):
        new_patient = {
            'glucose_level': self.patient.glucose_level,
            'weight': self.patient.weight,
            'physical_activity': self.patient.physical_activity,
            'motivation': self.patient.motivation
        }
        reward = self.get_reward(action)
        done = False  # condition for when the action is done
        info = {}  # Additional info
        return new_patient, reward, done, info

    def reset(self):
        self.patient = generate_random_patient()
        return {
            'glucose_level': self.patient.glucose_level,
            'weight': self.patient.weight,
            'physical_activity': self.patient.physical_activity,
            'motivation': self.patient.motivation
        }

    def get_reward(self, action):
        # Calculate reward based on action -give advice - and patient's response
        # positive reward for effective advice, 
        # negative reward for ineffective advice
        return 1
