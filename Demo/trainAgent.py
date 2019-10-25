#//////////////#####///////////////
#
# ANU u6325688 Yangyang Xu
# Supervisor: Dr.Penny Kyburz
#//////////////#####///////////////
from GAIL.gail import GAIL
from GAIL.gail1D import GAIL as GAIL1D
from commons.DataInfo import DataInfo
import sys

#######################################################################
ENVNAME = "IceHockey-v0"
expertPath = "commons/openai.gym.1568127083.838687.41524"
resourcePath = "resources"
resultPath = "result"
iteration = 1
enableOnPolicy = True
type="img"
########################################################################

gameInfo = DataInfo(ENVNAME)
gameInfo.loadData(expertPath, resourcePath, type)
gameInfo.displayActionDis()

if type == "loc":
    gail = GAIL1D(gameInfo, resultPath)
    gail.setUpGail()
    gail.train(iteration, enableOnPolicy)  # init index is 0
    gail.save(resourcePath, type + str(enableOnPolicy))
    del gail
else:
    gail = GAIL(gameInfo, resultPath)
    gail.setUpGail()
    gail.train(iteration, enableOnPolicy)  # init index is 0
    gail.save(resourcePath, type + str(enableOnPolicy))
    del gail

print("-----------------------------------TRAINING IS END-----------------------------------------------")
sys.exit(0)

