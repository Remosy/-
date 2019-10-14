from GAIL.gail import GAIL
from commons.DataInfo import DataInfo
import matplotlib.pyplot as plt
import Demo_gym as gym
import torch, sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make("IceHockey-v0")
gameInfo = DataInfo("IceHockey-v0")
gameInfo.loadData("Stage1/openai.gym.1568127083.838687.41524","resources","img")
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


gail.train(iteration) #init index is 0
gail.save("resources")
del gail
print("END")
sys.exit(0)

