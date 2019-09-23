from commons.DataInfo import DataInfo
from GAIL.Discriminator import Discriminator
class PPO():
    def __init__(self, dataInfo:DataInfo, oldAgent)-> None:
        self.dataInfo = dataInfo
        self.epsolone = 0.2
        self.accumReward = 0
        self.bias = 0
        self.oldAgent = oldAgent
        self.actor = None
        self.clip = 0
        self.advantage = 0

    def setAdvantage(self, valueFunc:Discriminator):
        #fit value function model

        self.advantage = 0






