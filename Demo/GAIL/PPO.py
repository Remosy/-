from commons.DataInfo import DataInfo
from GAIL.Generator import Generator
from GAIL.GEA import GEA
from torch.autograd import Variable
from StateClassifier import darknet
import gym, cv2, torch, os,shutil
import numpy as np
from sklearn.preprocessing import normalize
TMP = "StateClassifier/tmp"
class PPO():
    def __init__(self, generator:Generator,generatorOptim)-> None:
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
        self.actorOptim = generatorOptim
        self.entropyBeta = 0.001
        self.gameframe = 4000
        self.criticDiscount = 0.3

        self.totalReward = 0

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epoch = 3
        self.interval = 10
        self.interval1D = 30

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
            policyDist, action, _, _ = self.actor(state)
            score = self.actor.criticScore
            """
            state = (Variable(state.detach()).data).cpu().numpy()
            self.states.append(state)
            score = (Variable(score.detach()).data).cpu().numpy()
            self.scores.append(score)
            action = (Variable(action.detach()).data).cpu().numpy()
            self.actions.append(action)
            """
            state, reward, done, _ = self.env.step(action)
            self.totalReward += reward
            """
            self.distribution.append(policyDist)
            self.rewards.append(reward)
            self.dones.append(not done)
            """
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                self.env.reset()
        self.env.close()

    def tryEnvironment1D(self):
        state = self.env.reset()
        if os.path.isdir(TMP):
            shutil.rmtree(TMP)
        os.mkdir(TMP)
        for t in range(self.gameframe):
            self.env.render()
            tmpImg = np.asarray(state)
            cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            #Detect by YOLO
            imgpath = TMP+"/"+str(t)+".jpg"
            cv2.imwrite(imgpath, tmpImg)
            state = darknet.getState(imgpath)
            self.states.append(np.asarray(state))
            state = torch.FloatTensor(state).to(self.device)
            state = torch.unsqueeze(state, 0).to(self.device)
            policyDist, action, _ = self.actor(state)

            score = self.actor.criticScore
            score = (Variable(score.detach()).data).cpu().numpy()
            self.scores.append(score)
            state, reward, done, _ = self.env.step(action)
            self.totalReward += reward
            action = action.type(torch.FloatTensor)
            self.actions.append(action.detach())
            self.distribution.append(policyDist.detach())
            self.rewards.append(reward)
            self.dones.append(not done)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                self.env.reset()
        shutil.rmtree("StateClassifier/tmp")
        self.env.close()
        self.distribution = torch.FloatTensor(self.distribution).to(self.device)
        self.scores = np.asarray(self.scores)


    def importExpertData(self,states,actions,rewards,scores,dones,dists):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.scores = scores
        self.distribution = dists
        self.dones = dones


    def optimiseGenerator(self):
        #dataRange = len(self.states)
        dataRange = list(range(self.interval, len(self.states), self.interval))
        gea = GEA(self.scores, self.rewards, self.dones)
        self.advantages, self.returns = gea.getAdavantage()
        loss = 0
        for ei in range(self.epoch):
            for i in dataRange:
                oldPolicyDist = self.distribution[i-self.interval:i]
                tmpState = torch.from_numpy(self.states[i-self.interval:i]).type(torch.FloatTensor).to(self.device)
                newPolicyDist, newAction, actEntropy,_ = self.actor(tmpState)
                actEntropy = actEntropy.mean()
                # --------------CLIP------------------
                ratio = torch.mean(torch.exp(newPolicyDist - oldPolicyDist)).type(torch.FloatTensor).to(self.device)
                clipResult = 0
                # CLIP
                if ratio < 1 - self.epsilon:
                    clipResult = 1 - self.epsilon
                elif ratio > 1 + self.epsilon:
                    clipResult = 1 + self.epsilon
                else:
                    clipResult = ratio

                # LOSS
                adva = torch.from_numpy(self.advantages[i-self.interval:i]).type(torch.FloatTensor).to(self.device)
                adva = torch.mean(adva)
                actorloss = -min((ratio * adva).mean(), (clipResult * adva).mean())
                criticloss = np.mean(np.power((self.returns[i-self.interval:i]).mean() - (self.scores[i-self.interval:i]).mean(), 2))
                # -(Loss[clip] - Loss[crit]+ H)
                loss = -actorloss + self.criticDiscount * criticloss - self.entropyBeta * actEntropy
                self.actorOptim.zero_grad()
                loss.backward(retain_graph=True)
                self.actorOptim.step()
        """
        for ei in range(self.epoch):
            for i in range(dataRange):
                #print("--Epoch {}--{}".format(str(ei),str(i)))
                oldPolicyDist = self.distribution[i]
                tmpState = torch.from_numpy(self.states[i]).type(torch.FloatTensor).to(self.device)
                if len(tmpState.shape) < 4:
                    tmpState = torch.unsqueeze(tmpState, 0).to(self.device)  # => (n,3,210,160)
                newPolicyDist, newAction, actEntropy,_ = self.actor(tmpState)
                # --------------CLIP------------------
                ratio = torch.exp(newPolicyDist - oldPolicyDist).type(torch.FloatTensor).to(self.device)
                clipResult = 0
                # CLIP
                if ratio < 1 - self.epsilon:
                    clipResult = 1 - self.epsilon
                elif ratio > 1 + self.epsilon:
                    clipResult = 1 + self.epsilon
                else:
                    clipResult = ratio

                #----------------LOSS----------------------------
                adva = torch.from_numpy(self.advantages[i]).type(torch.FloatTensor).to(self.device)
                actorloss = min((ratio * adva).mean(), (clipResult * adva).mean())

                #SquareErrorLoss
                criticloss = np.mean(np.power(self.returns[i]-self.scores[i],2))

                #-(Loss[clip] - Loss[crit]+ H)
                loss = -actorloss+self.criticDiscount*criticloss-self.entropyBeta*actEntropy
                # ----------------LOSS----------------------------

                self.actorOptim.zero_grad()
                loss.backward(retain_graph=True)
                self.actorOptim.step()
        """
        if type(loss)!=int:
            loss = loss.detach()
        return self.actor.state_dict(), loss, actEntropy.detach()

    def optimiseGenerator1D(self):
        dataRange = list(range(self.interval1D, len(self.states), self.interval1D))
        gea = GEA(self.scores, self.rewards, self.dones)
        self.advantages, self.returns = gea.getAdavantage()
        self.states = normalize(self.states)
        loss = 0

        for ei in range(self.epoch):
            for i in dataRange:
                oldPolicyDist = self.distribution[i-self.interval1D:i]
                tmpState = torch.from_numpy(self.states[i-self.interval1D:i]).type(torch.FloatTensor).to(self.device)
                #if len(tmpState.shape) < 4:
                    #tmpState = torch.unsqueeze(tmpState, 0).to(self.device)  # => (n,20)
                newPolicyDist, newAction, actEntropy = self.actor(tmpState)
                if torch.isnan(actEntropy):
                    continue
                # --------------CLIP------------------
                ratio = torch.mean(torch.exp(newPolicyDist - oldPolicyDist)).type(torch.FloatTensor).to(self.device)
                clipResult = 0
                # CLIP
                if ratio < 1 - self.epsilon:
                    clipResult = 1 - self.epsilon
                elif ratio > 1 + self.epsilon:
                    clipResult = 1 + self.epsilon
                else:
                    clipResult = ratio

                # LOSS
                adva = torch.from_numpy(self.advantages[i-self.interval1D:i]).type(torch.FloatTensor).to(self.device)
                adva = torch.mean(adva)
                actorloss = -min((ratio * adva).mean(), (clipResult * adva).mean())
                criticloss = np.mean(np.power((self.returns[i-self.interval1D:i]).mean() - (self.scores[i-self.interval1D:i]).mean(), 2))
                # -(Loss[clip] - Loss[crit]+ H)
                loss = -actorloss + self.criticDiscount * criticloss - self.entropyBeta * actEntropy
                self.actorOptim.zero_grad()
                loss.backward(retain_graph=True)
                self.actorOptim.step()
        """
        for ei in range(self.epoch):
            for i in range(dataRange):
                oldPolicyDist = self.distribution[i]
                tmpState = torch.from_numpy(self.states[i]).type(torch.FloatTensor).to(self.device)
                if len(tmpState.shape) < 4:
                    tmpState = torch.unsqueeze(tmpState, 0).to(self.device)  # => (n,3,210,160)
                newPolicyDist, newAction, actEntropy = self.actor(tmpState)
                if torch.isnan(actEntropy):
                    continue
                # --------------CLIP------------------
                ratio = torch.exp(newPolicyDist - oldPolicyDist).type(torch.FloatTensor).to(self.device)
                clipResult = 0
                # CLIP
                if ratio < 1 - self.epsilon:
                    clipResult = 1 - self.epsilon
                elif ratio > 1 + self.epsilon:
                    clipResult = 1 + self.epsilon
                else:
                    clipResult = ratio

                # LOSS
                adva = torch.from_numpy(self.advantages[i]).type(torch.FloatTensor).to(self.device)
                actorloss = -min((ratio * adva).mean(), (clipResult * adva).mean())
                criticloss = np.mean(np.power(self.returns[i] - self.scores[i], 2))
                # -(Loss[clip] - Loss[crit]+ H)
                loss = -actorloss + self.criticDiscount * criticloss - self.entropyBeta * actEntropy
                self.actorOptim.zero_grad()
                loss.backward(retain_graph=True)
                self.actorOptim.step()
        """
        if type(loss)!=int:
            loss = loss.detach()
        return self.actor.state_dict(), loss, actEntropy.detach()




