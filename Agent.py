import numpy as np
from Parameters import *

class Agent:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        # Random initialization of velocity and initializing acceleration to null
        self.acceleration = np.zeros(2)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]

        self.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)       
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    def update(self, action):
        self.acceleration += action
        
        self.acceleration=np.clip(self.acceleration, -(SimulationVariables["AccelerationUpperLimit"]), SimulationVariables["AccelerationUpperLimit"])

        self.velocity += self.acceleration * SimulationVariables["dt"]
        vel = np.linalg.norm(self.velocity)
        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity    
        self.position += self.velocity * SimulationVariables["dt"]

        return self.position, self.velocity
