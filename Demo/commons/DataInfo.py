
import sys, numpy
from sympy.ntheory import factorint
import shutil, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Stage1.getVideoWAction import GetVideoWAction
from collections import Counter
from StateClassifier import darknet

class DataInfo():
    def __init__(self, gameName)-> None:
        self.gameName = gameName
        self.gameFrame = 100
        self.epsolone = 0.2

        self.expertState = []
        self.expertAction = []
        self.expertReward = []
        self.expertLocation = []
        self.rawState = []
        self.rawAction = []
        self.rawReward = []
        self.rawLocation = []
        self.locationSample = []
        self.numEntity = 0

        self.stateShape = []
        self.actionShape = []
        self.locateShape = []
        self.maxAction = 1
        self.numActPState = 1 #number of action per state
        self.miniBatchDivider = 2
        self.batchDivider = 100
        self.stateTensorShape = 0
        self.stateTensorShape = 0

        self.generatorIn = 0
        self.generatorOut = 18 #num of actions [0 - 17]
        self.generatorKernel = 2

        self.discriminatorIn = 0
        self.discriminatorOut = 1 #fixed
        self.discriminatorKernel = 2 #fixed

        self.midLayerShape = (0,0)



    def loadData(self, folder, targetFolder, type):
        tmp1 = folder.split("/")
        ipath = targetFolder+"/"+tmp1[-1]
        #load images
        if not os.path.isdir(ipath):
            print("Collecting data")
            expertData = GetVideoWAction(self.gameName, 3, True)
            dataName = expertData.replay(folder, targetFolder)
        else:
            dataName = ipath

        # Read Action
        self.rawAction = np.load(dataName+"/action.npy")
        self.rawLocation = np.load(dataName+"/location.npy")
        self.maxAction = max(self.rawAction)
        self.numEntity = len(self.rawAction)
        # Read Reward
        self.rawReward = np.load(dataName+"/reward.npy")
        # Read State
        shutil.unpack_archive(dataName + "/state.zip", dataName + "/state")
        for ii in range(0, self.numEntity):
            ii += 1
            self.rawState.append(dataName + "/state/"+str(ii)+".jpg")

        imgSample = cv2.imread(self.rawState[0])

        self.actionShape = self.rawAction[0].size
        self.stateShape = imgSample.shape
        self.locateShape = self.rawLocation[0].size

        if type == "loc":
            self.generatorIn = self.locateShape  #dimension = 20
            self.discriminatorIn = self.locateShape + 1
        else:
            self.generatorIn = imgSample.shape[-1]  # dimension = 3
            self.discriminatorIn = self.generatorOut + 1

        del imgSample

        if self.generatorIn == 3: #use the least common divisor of input's w & h as the kernel
            factora = set(factorint(self.stateShape[0]).keys())
            factorb = set(factorint(self.stateShape[1]).keys())
            factors = factora.union(factorb)
            self.generatorKernel = min(factors)



    def shuffle(self):
        self.rawState, self.rawAction, self.rawReward = shuffle(self.rawState, self.rawAction, self.rawReward)

    def sampleData(self):
        print("--Total img data {}--".format(str(len(self.rawLocation))))
        self.expertAction = np.array_split(self.rawAction, self.batchDivider)
        self.expertReward = np.array_split(self.rawReward, self.batchDivider)
        self.expertState = np.array_split(self.rawState, self.batchDivider)
        self.expertLocation = np.array_split(self.rawLocation, self.batchDivider)
        print("--devided into {} batches".format(str(len(self.expertAction))))

    def displayActionDis(self):
        x = Counter(self.rawAction).keys() # equals to list(set(words))
        y = Counter(self.rawAction).values()  # counts the elements' frequency
        y_pos = np.arange(len(x))
        plt.bar(y_pos, y, align='center')
        plt.xticks(y_pos, x)
        plt.savefig("RAWaction.png")
        plt.close()

    def defineGame(self):
        self.actionShape = self.env.action_space.shape
        self.stateShape = self.env.observation_space.shape

        if "Tuple" in self.env.action_space:
            self.generatorOut = len(self.env.action_space.spaces)
        elif "Box" in self.env.action_space:
            self.generatorOut = self.actionShape[-1]
            self.maxAction = self.env.action_space.high[0]
        elif "Discrete" in self.env.action_space:
            self.generatorOut = 1
            self.maxAction = self.env.action_space.n - 1
        else:
            sys.exit("This game is too new")

        if "Tuple" in self.env.observation_space:
            self.discriminatorIn = 1
        elif "Box" in self.env.observation_space:
            self.generatorIn = self.stateShape.shape[-1]
            if len(self.stateShape.shape) > 1:
               factora = set(factorint(self.stateShape[0]).keys())
               factorb = set(factorint(self.stateShape[1]).keys())
               factors = factora.union(factorb)
            else:
                factors = set(factorint(self.stateShape[0]).keys())
            self.generatorKernel = min(factors)
            self.discriminatorIn = numpy.prod(self.stateShape.shape) + self.generatorOut #ToDo
        elif "Discrete" in self.env.observation_space:
            self.generatorIn = self.stateShape.shape[-1]
        else:
            sys.exit("This game is too new")









