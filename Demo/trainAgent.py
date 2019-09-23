from GAIL.gail import GAIL
from commons.DataInfo import DataInfo
import matplotlib.pyplot as plt
import Demo_gym as gym
import torch, gc, cv2, sys
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make("IceHockey-v0")
gameInfo = DataInfo("IceHockey-v0")
gameInfo.loadData("Stage1/openai.gym.1568127083.838687.41524","resources")
gameInfo.displayActionDis()
gail = GAIL(gameInfo)
gail.setUpGail()
epoch = 2
iteration = 1
episode = 1
frame = 4000
plotEpoch = []
plotReward = []
#for obj in gc.get_objects():
    #try:
        #if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #print(type(obj), obj.size())
    #except:
        #pass

for ep in range(epoch):
    print("********Epoch {}".format(str(ep)))
    gail.train(iteration) #init index is 0
    totalReawrd = 0
    for i_episode in range(episode):
        state = env.reset()
        for t in range(frame):
            env.render()
            tmpImg = np.asarray(state)
            cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            state = np.rollaxis(tmpImg, 2, 0)
            state = (torch.from_numpy(state/255)).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0).to(device) #=> (n,3,210,160)
            actionDis = gail.generator(state)
            #print(actionDis)
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
#plt.figure(0)
plt.plot(plotEpoch,plotReward, marker="X")
plt.xlabel("Epochs")
plt.ylabel("Rewards")
plt.title("{} {} {}".format("IceHockey","0.005","ImageState"))
plt.savefig("TrainResult.png")
env.close()
sys.exit(0)