from GAIL.gail import GAIL
from commons.DataInfo import DataInfo
import sys

gameInfo = DataInfo("IceHockey-v0")
gameInfo.loadData("Stage1/openai.gym.1568127083.838687.41524","resources","img")
gameInfo.displayActionDis()
gail = GAIL(gameInfo)
gail.setUpGail()
iteration = 10
gail.train(iteration,False)
gail.save("resources","rgbFalse")
del gail
print("END")
sys.exit(0)

