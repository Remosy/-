#//////////////#####///////////////
#
# ANU u6325688 Yangyang Xu
# Supervisor: Dr.Penny Kyburz
#//////////////#####///////////////
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
        self.scores = scores
        self.rewards = rewards
        self.dones = dones
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def getAdavantage(self):
        recordSize = len(self.scores)
        orders = reversed(range(recordSize))
        returns = np.empty(shape=(recordSize))
        gae = 0
        for i in orders:
            i = i-1
            delta = self.rewards[i] + self.discountFactor * self.scores[i+1]*self.dones[i]- self.scores[i]
            #current-gea = current delta + discounted old-gea
            gae = self.smoothing * self.discountFactor * self.dones[i] * gae + delta

            returns[i] = gae + self.scores[i]
        advantages = returns - self.scores

        #Normalise
        advantages = advantages - advantages.mean()
        advantages = advantages / (advantages.std() + 1e-8) #avoid 0
        return advantages, returns







