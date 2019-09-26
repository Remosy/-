from commons.DataInfo import DataInfo
from GAIL.Generator import Generator
from GAIL.GEA import GEA
from torch.distributions import Normal
import gym, cv2, torch
import numpy as np
class PPO():
    def __init__(self, generator:Generator)-> None:
        self.epsilon = 0.2
        self.accumReward = 0
        self.bias = 0
        self.clip = 0
        self.advantages = []
        self.env = gym.make("IceHockey-v0")
        self.actions = []
        self.states = []
        self.scores = []
        self.distribution = []
        self.rewards = []
        self.dones = []
        self.returns = []
        self.actor = generator
        self.entropyBeta = 0.001
        self.gameframe = 3
        self.criticDiscount = 0.5

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            score = self.actor.criticScore

            self.states.append(state)
            self.scores.append(score)
            self.actions.append(action)
            actorDist = Normal(actorDis, self.actor.std)
            self.distribution.append(actorDist)

            state, reward, done, _ = self.env.step(action)
            self.rewards.append(reward)
            self.dones.append(not done)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                self.env.reset()

        self.env.close()


    def optimiseGenerator(self,actorOptimiser):
        dataRange = len(self.states)
        gea = GEA(self.scores, self.rewards, self.dones)
        self.advantages, self.returns = gea.getAdavantage()
        for i in range(dataRange):
            oldDistribution = self.distribution[i]
            newActDis = self.actor(self.states[i])
            entropy = 0.5
            #entropy = newActDis.entropy().mean() #ToDo

            newDistribution = Normal(newActDis, self.actor.std)
            # --------------------------------
            ratio = torch.exp(newDistribution._natural_params - oldDistribution._natural_params) #ToDo
            clipResult = 0
            # CLIP
            if ratio < 1 - self.epsilon:
                clipResult = 1 - self.epsilon
            elif ratio > 1 - self.epsilon:
                clipResult = 1 + self.epsilon
            else:
                clipResult = ratio

            actorloss = min((ratio, clipResult)*self.advantages[i]).mean()
            criticloss = (self.rewards-self.scores).pow(2).mean()
            loss = self.criticDiscount*criticloss+actorloss-entropy*self.entropyBeta

            actorOptimiser.zero_grad()
            loss.backward()
            actorOptimiser.step()
        return actorOptimiser







