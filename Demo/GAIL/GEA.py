from commons.DataInfo import DataInfo
from GAIL.Generator import Generator
from torch.distributions import Normal
import gym, cv2, torch
import numpy as np
class GEA():
    def __init__(self, scores, rewards, dones)-> None:
        self.delta = 0
        self.discountFactor = 0.99
        self.smoothing = 0.95
        self.gea = 0
        self.returns = []
        self.scores = scores
        self.rewards = rewards
        self.dones = dones
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def getAdavantage(self):
        orders = reversed(range(len(self.scores)))
        for i in orders:
            delta = self.rewards[i] + self.discountFactor * self.scores[i]*self.dones[i]- self.scores[i]
            self.gea = delta + self.smoothing * self.discountFactor * self.dones[i] *self.gea
            self.returns.insert(0, self.gea + self.scores[i])
        advantages = np.array(self.returns) - self.scores[:]
        #advantages = (advantages-np.mean(advantages))/np.std(advantages) #normolised advantages
        return advantages, self.returns







