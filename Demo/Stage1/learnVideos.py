import numpy as np

class LearnVideos:
    def __init__(self):
        self.trainData = np.load("videoFrames.npy")
        self.train()


if __name__ == "__main__":

    x = LearnVideos()
    def train(self):
        learningRate = 0.0005
        trainingEpoch = 5
        batchsize = 256
        display_step = 1
        examples_to_show = 10
