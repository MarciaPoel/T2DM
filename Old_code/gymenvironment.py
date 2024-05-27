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

        self.action_space = spaces.Discrete(3)  # 3 actions for now: walk, gym, diet

        # Initialize patient
        self.patient = generate_random_patient()

    def step(self, action):
        #take action, return new state with reward, update patient state - on action and return observation
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

    def send_advice(self, action):
        if action == 0:
            return "Take 1h walk"
        elif action == 1:
            return "Hit the gym"
        elif action == 2:
            return "Think about the diet"

    def reset(self):
        self.patient = generate_random_patient()
        return {
            'glucose_level': self.patient.glucose_level,
            'weight': self.patient.weight,
            'physical_activity': self.patient.physical_activity,
            'motivation': self.patient.motivation
        }

    def get_reward(self, action):
        # quantify effectiveness advice - on patient response & long-term outcome
        # positive reward for effective advice, 
        # negative reward for ineffective advice
        if patient.glucose_level <= 100:
            return reward = 1
        elif patient.glucose_level >= 126:
            return reward = -1
