import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
from torch.autograd import Variable

import numpy as np
import cv2

from typing import NewType
GradCAM = NewType('GradCAM', int)
LAYER_NAME = "35"

inWidth = 100
inHeight = 190
inDepth = 3 #RGB

use_gpu = torch.cuda.is_available()
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def preprocess_image(img):
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[: , :, ::-1]
	for i in range(3):
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = Variable(preprocessed_img, requires_grad = True)
	return input

def show_cam_on_image(img, mask):
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite("cam2.jpg", np.uint8(255 * cam))
	return cam

class Gradient():
    def __init__(self,model,whichLayer):
        self.model = model
        self.gradLayer = whichLayer
        self.gradients = []

    def updateGradeint(self,grad):
        self.gradients.append(grad)

    def __call__(self,img):
        targetAct = []
        self.gradients = []
        for name, module in self.model._modules.items():
            img = module(img)
            if name == self.gradLayer:
                img.register_hook(self.updateGradeint)
                targetAct += [img]
        return targetAct, img


class GradCAM:
    def __init__(self,input):
        self.data = input
        self.model = models.vgg19(pretrained=True)
        self.model.eval()
        if use_gpu:
            print("Using GPU")
            self.model= self.model.cuda()
        self.gradient = Gradient(self.model.features, LAYER_NAME)


    def train(self, inImgs):
        return self.model(inImgs)

    def getCAM(self):
        if use_gpu:
            targetActivation, output = self.gradient(self.data.cuda())
        else:
            targetActivation, output = self.gradient(self.data)
        output = output.view(output.size(0), -1)
        self.model.classifier[0] = nn.Linear(7680, 4096) #modify for 210*160
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
        encode.backward()

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

    def display(self, img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cv2.imwrite("cam2.jpg", np.uint8(255 * cam))
        return cam

if __name__ == '__main__':
    img = cv2.imread("/Users/remosy/Desktop/test1.png",1)
    img = np.float32(cv2.resize(img, (inWidth,inHeight))) / 255
    out = preprocess_image(img)

    gradcam = GradCAM(out)
    cam = gradcam.getCAM() #mask
    gradcam.display(img,cam)