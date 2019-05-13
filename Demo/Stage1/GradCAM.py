import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import utils
from torch.autograd import Variable

import numpy as np
import cv2

from typing import NewType
GradCAM = NewType('GradCAM', int)
LAYER_NAME = "35"

inWidth = 160
inHeight = 210
inDepth = 3 #RGB

use_gpu = torch.cuda.is_available()

class Gradient():
    def __init__(self,model:GradCAM,whichLayer):
        self.model = model
        self.gradLayer = whichLayer
        self.gradients = []

    def updateGradeint(self,grad):
        self.gradients.append(grad)

    def __call__(self,img):
        targetAct = []
        for name, module in self.model._modules.items():
            out = module(img)
            if name == self.gradLayer:
                out.register_hook(self.updateGradeint)
                targetAct += [out]
        return targetAct, out


class GradCAM:
    def __init__(self,input):
        super(GradCAM, self).__init__()
        self.data = np.load(input)
        self.modle = models.vgg16(pretrained=True)
        self.model.eval()
        if use_gpu:
            print("Using GPU")
            self.modle= self.modle.cuda()
        self.gradient = Gradient(self.modle,LAYER_NAME)


    def train(self, inImgs):
        return self.modle(inImgs)

    def getCAM(self):
        if use_gpu:
            targetActivation, output = self.gradient(self.data.cuda())
        targetActivation, output = self.gradient()
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        index = np.argmax(output.cpu().data.numpy())

        encode = np.zeros((1, output.size()[-1]), dtype=np.float32)
        encode[0][index] = 1
        encode = Variable(torch.from_numpy(encode), requires_grad=True)
        if use_gpu:
            encode = torch.sum(encode.cuda() * output)
        else:
            encode = torch.sum(encode * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        encode.backward(retain_variables=True)

        grads_val = self.gradient.gradients[-1].cpu().data.numpy()

        target = targetActivation[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (inWidth, inHeight))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

if __name__ == '__main__':
    img = cv2.imread("")
    gradcam = GradCAM(img)
    cam = gradcam.getCAM() #mask
