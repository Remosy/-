from commons.DataInfo import DataInfo
from GAIL.Generator import Generator
from GAIL.GEA import GEA
from torch.distributions import Normal
from scipy.stats import entropy
from StateClassifier import darknet
import gym, cv2, torch
import numpy as np
class PPO():
    def __init__(self, generator:Generator,lr)-> None:
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
        self.actorOptim = torch.optim.Adam(generator.parameters(), lr)
        self.entropyBeta = 0.001
        self.gameframe = 3
        self.criticDiscount = 0.5


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epoch = 5

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
            policyDist, action, _ = self.actor(state)
            score = self.actor.criticScore

            self.states.append(state)
            self.scores.append(score.detach())
            state, reward, done, _ = self.env.step(action)
            action = action.type(torch.FloatTensor)
            self.actions.append(action)
            self.distribution.append(policyDist)
            self.rewards.append(reward)
            self.dones.append(not done)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                self.env.reset()

        self.env.close()

    def tryEnvironment1D(self, sampleState):
        state = self.env.reset()
        for t in range(self.gameframe):
            self.env.render()
            tmpImg = np.asarray(state)
            cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            #Detect by YOLO
            state = darknet.getState(tmpImg, sampleState)
            policyDist, action, _ = self.actor(state)
            score = self.actor.criticScore

            self.states.append(state)
            self.scores.append(score.detach())
            state, reward, done, _ = self.env.step(action)
            action = action.type(torch.FloatTensor)
            self.actions.append(action)
            self.distribution.append(policyDist)
            self.rewards.append(reward)
            self.dones.append(not done)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                self.env.reset()

        self.env.close()

    def importExpertData(self,states,actions,rewards,scores,dones,dists):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.scores = scores
        self.distribution = dists
        self.dones = dones



    def optimiseGenerator(self,lr):
        dataRange = len(self.states)
        gea = GEA(self.scores, self.rewards, self.dones)
        self.advantages, self.returns = gea.getAdavantage()
        for ei in range(self.epoch):
            for i in range(dataRange):
                oldPolicyDist = self.distribution[i]
                newPolicyDist, newAction, actEntropy  = self.actor(self.states[i])
                # --------------------------------
                ratio = torch.exp(newPolicyDist - oldPolicyDist).type(torch.FloatTensor).to(self.device)
                clipResult = 0
                # CLIP
                if ratio < 1 - self.epsilon:
                    clipResult = 1 - self.epsilon
                elif ratio > 1 + self.epsilon:
                    clipResult = 1 + self.epsilon
                else:
                    clipResult = ratio

                #LOSS
                actorloss = min((ratio*self.advantages[i]).mean(), (clipResult*self.advantages[i]).mean())
                criticloss = (self.returns[i]-self.scores[i]).pow(2).mean()
                loss = self.criticDiscount*criticloss+actorloss-actEntropy*self.entropyBeta

                self.actorOptim.zero_grad()
                loss.backward(retain_graph=True)
                self.actorOptim.step()
        return self.actor, self.actorOptim







