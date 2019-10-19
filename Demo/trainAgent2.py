from GAIL.gail1D import GAIL
from commons.DataInfo import DataInfo
import  sys

gameInfo = DataInfo("IceHockey-v0")
gameInfo.loadData("Stage1/openai.gym.1568127083.838687.41524","resources","loc")
gameInfo.displayActionDis()
gail = GAIL(gameInfo)
gail.setUpGail()
iteration = 100
gail.train(iteration,True) #init index is 0
gail.save("resources","locOff")
del gail
print("END")
sys.exit(0)