from GAIL.gail import GAIL
from commons.DataInfo import DataInfo
import matplotlib.pyplot as plt
import gym
import torch
import numpy as np

env = gym.make("IceHockey-v0")
gameInfo = DataInfo("IceHockey-v0")
gameInfo.loadData("/DropTheGame/Demo/Stage1/openai.gym.1566264389.031848.82365","/Users/u6325688/DropTheGame/Demo/resources/openai.gym.1566264389.031848.82365")
gameInfo.sampleData()
gail = GAIL(gameInfo)
gail.setUpGail()

epoch = 5
plotEpoch = []
plotReward = []

for ep in range(epoch):
    #gail.train(1) #init index is 0
    totalReawrd = 0
    for i_episode in range(1):
        state = env.reset()
        for t in range(200):
            env.render()
            state = np.rollaxis(state, 2, 0)
            state = (torch.from_numpy(state)).type(torch.FloatTensor)
            state = torch.unsqueeze(state,0)
            actionDis = gail.generator(state)
            action = (actionDis).argmax(1)
            state, reward, done, _ = env.step(action)
            totalReawrd += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    plotEpoch.append(ep)
    plotReward.append(totalReawrd/(i_episode+1))

    print("Epoch: {}\t Avg Reward: {}".format(ep, totalReawrd/(i_episode+1)))

gail.save("resources")
del gail

plt.plot(plotEpoch,plotReward, marker="X")
plt.xlabel("Epochs")
plt.ylabel("Rewards")
plt.title("{} {} {}".format("IceHockey","0.005","ImageState"))
env.close()