from GAIL.gail import GAIL
from commons.DataInfo import DataInfo
import matplotlib.pyplot as plt

import Demo_gym as gym
env = gym.make("IceHockey-v0")
gameInfo = DataInfo("IceHockey-v0")
gameInfo.loadData("/DropTheGame/Demo/Stage1/openai.gym.1566264389.031848.82365","/Users/u6325688/DropTheGame/Demo/resources/openai.gym.1566264389.031848.82365")
gameInfo.sampleData()
gail = GAIL(gameInfo)
gail.setUpGail()

epoch = 5
plotEpoch = []
plotReward = []

for epoch in range(epoch):
    gail.train(10, 0) #init index is 0
    totalReawrd = 0
    for i_episode in range(20):
        state = env.reset()
        for t in range(100):
            env.render()
            action = gail.getAction(state)
            observation, reward, done, info = env.step(action)
            totalReawrd += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    plotEpoch.append(epoch)
    plotReward.append(reward)

    print("Epoch: {}\t Reward: {}".format(epoch, totalReawrd/i_episode))

plt.plot(plotEpoch,plotReward)
plt.xlabel("Epochs")
plt.ylabel("Rewards")
plt.title("{} {} {}".format(("IceHockey","0.005","ImageState")))
env.close()