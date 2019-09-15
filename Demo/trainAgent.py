from GAIL.gail import GAIL
from commons.DataInfo import DataInfo
import matplotlib.pyplot as plt
import Demo_gym as gym
import torch, gc
import numpy as np

env = gym.make("IceHockey-v0")
gameInfo = DataInfo("IceHockey-v0")
gameInfo.loadData("Stage1/openai.gym.1568127083.838687.41524","resources")
gameInfo.sampleData()
gail = GAIL(gameInfo)
gail.setUpGail()

epoch = 1
plotEpoch = []
plotReward = []
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())
    except:
        pass

for ep in range(epoch):
    gail.train(1) #init index is 0
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