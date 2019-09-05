
import gym,sys, numpy, math
from sympy.ntheory import factorint
import shutil
import numpy as np

class DataInfo():
    def __init__(self, gameName)-> None:
        self.gameName = gameName
        self.expertState = []
        self.expertAction = []
        self.expertReward = []
        self.numEntity = 0

        self.stateShape = []
        self.actionShape = []
        self.maxAction = 1
        self.miniBatch = 2
        self.batch = 2

        self.generatorIn = 0
        self.generatorOut = 0
        self.generatorKernel = 0

        self.discriminatorIn = 0
        self.discriminatorOut = 0
        self.discriminatorKernel = 2 #fixed

    def loadData(self, folder, targetFolder):
        #load images
        #expertData = GetVideoWAction(self.gameInfo.gameName, 3, True)
        #dataName = expertData.replay(folder, targetFolder)

        dataName ="resources/openai.gym.1566264389.031848.82365"
        # Read Action
        self.expertAction = np.load(dataName+"/action.npy")
        self.maxAction = max(self.expertAction)
        self.numEntity = len(self.expertAction)
        # Read Reward
        self.expertReward = np.load(dataName+"/reward.npy")
        # Read State
        shutil.unpack_archive(dataName + "/state.zip", dataName + "/state")
        for ii in range(0, self.numEntity):
            ii += 1
            self.expertState.append(dataName + "/state/"+str(ii)+".jpg")

        self.generatorIn = self.expertState.shape[-1]
        self.generatorOut = self.expertAction.shape[-1]
        self.stateShape = self.expertState[0].shape()
        if self.generatorIn == 3:
            factora = set(factorint(self.stateShape[0]).keys())
            factorb = set(factorint(self.stateShape[1]).keys())
            factors = factora.union(factorb)
            self.generatorKernel = min(factors)

        self.discriminatorIn = numpy.prod(self.stateShape.shape) + self.expertAction.shape[-1]
        self.discriminatorOut = 1 * self.batch



    def sampleData(self):
        actionChunks = np.array_split(self.expertAction, self.batch)
        rewardChunks = np.array_split(self.expertReward, self.batch)
        stateChunks = np.array_split(self.expertState, self.batch)
        return actionChunks, rewardChunks, stateChunks

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
            self.discriminatorIn = numpy.prod(self.stateShape.shape) + self.generatorOut
        elif "Discrete" in self.env.observation_space:
            self.generatorIn = self.stateShape.shape[-1]
        else:
            sys.exit("This game is too new")









