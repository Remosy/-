
import sys, numpy
from sympy.ntheory import factorint
import shutil, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from Stage1.getVideoWAction import GetVideoWAction
from collections import Counter

class DataInfo():
    def __init__(self, gameName)-> None:
        self.gameName = gameName
        self.gameFrame = 100
        self.epsolone = 0.2

        self.expertState = []
        self.expertAction = []
        self.expertReward = []
        self.rawState = []
        self.rawAction = []
        self.rawReward = []
        self.numEntity = 0

        self.stateShape = []
        self.actionShape = []
        self.maxAction = 1
        self.numActPState = 1 #number of action per state
        self.miniBatchDivider = 2
        self.batchDivider = 5 #ToDo
        self.stateTensorShape = 0
        self.stateTensorShape = 0

        self.generatorIn = 0
        self.generatorOut = 18 #num of actions [0 - 17]
        self.generatorKernel = 2

        self.discriminatorIn = 0
        self.discriminatorOut = 1 #fixed
        self.discriminatorKernel = 2 #fixed

    def loadData(self, folder, targetFolder):
        tmp1 = folder.split("/")
        ipath = targetFolder+"/"+tmp1[-1]
        #load images
        if not os.path.isdir(ipath):
            expertData = GetVideoWAction(self.gameName, 3, True)
            dataName = expertData.replay(folder, targetFolder)
        else:
            dataName = ipath
        #/Users/remosy/Desktop/DropTheGame/Demo/resources/openai.gym.1566264389.031848.82365"

        # Read Action
        self.rawAction = np.load(dataName+"/action.npy")
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
        self.generatorIn = imgSample.shape[-1]
        #self.generatorOut = self.expertAction[0].size

        self.actionShape = self.rawAction[0].size #ToDo:
        self.stateShape = imgSample.shape
        if self.generatorIn == 3: #use the least common divisor of input's w & h as the kernel
            factora = set(factorint(self.stateShape[0]).keys())
            factorb = set(factorint(self.stateShape[1]).keys())
            factors = factora.union(factorb)
            self.generatorKernel = min(factors)

        #self.discriminatorIn = np.prod(self.stateShape) + self.numActPState
        self.discriminatorIn = 1

    def shuffle(self):
        self.rawState, self.rawAction, self.rawReward\
        = shuffle(self.rawState, self.rawAction, self.rawReward )

    def sampleData(self):
        print("--Total img data {}--".format(str(len(self.rawAction))))
        self.expertAction = np.array_split(self.rawAction, self.batchDivider)
        self.expertReward = np.array_split(self.rawReward, self.batchDivider)
        self.expertState = np.array_split(self.rawState, self.batchDivider)
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









