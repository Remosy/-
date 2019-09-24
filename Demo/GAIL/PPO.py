from commons.DataInfo import DataInfo
from GAIL.Generator import Generator
from torch.distributions import Normal
import gym, cv2, torch
import numpy as np
class PPO():
    def __init__(self, generator:Generator)-> None:
        self.epsilon = 0.2
        self.accumReward = 0
        self.bias = 0
        self.clip = 0
        self.advantage = 0
        self.env = gym.make("IceHockey-v0")
        self.actions = []
        self.states = []
        self.distribution = []
        self.rewards = []
        self.actor = generator
        self.actor
        self.gameframe = 200
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def setAdavantage(self, returns, values):
        self.advantage = returns - values


    #Collect a bunch of samples from enviroment
    def tryEnvironment(self):
        state = self.env.reset()
        for t in range(self.gameframe):
            self.env.render()
            tmpImg = np.asarray(state)
            cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            state = np.rollaxis(tmpImg, 2, 0)
            state = (torch.from_numpy(state / 255)).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0).to(self.device)  # => (n,3,210,160)
            actorDis = self.actor(state)
            action = (actorDis).argmax(1)

            self.states.append(state)
            self.actions.append(action)
            actorDist = Normal(actorDis, self.actor.std)
            self.distribution.append(actorDist)

            state, reward, done, _ = self.env.step(action)


    def optimiseGenerator(self,optimiser:torch.optim):
        dataRange = len(self.states)
        for i in range(dataRange):
            oldDistribution = self.distribution[i]
            newActDis = self.actor(self.states[i])
            self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
            std = self.log_std.exp().expand_as(newActDis)
            newDistribution = Normal(newActDis, std)
            # --------------------------------
            ratio = torch.exp(newDistribution-oldDistribution)
            clipResult = 0
            # CLIP
            if ratio < 1 - self.epsilon:
                clipResult = 1 - self.epsilon
            elif ratio > 1 - self.epsilon:
                clipResult = 1 + self.epsilon
            else:
                clipResult = ratio
                # ToDo:~~!!!!
            loss = min((ratio, clipResult)*self.advantage)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        return optimiser







