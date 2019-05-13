import torch
import numpy as np
import cv2


inWidth = 160
inHeight = 210
inDepth = 3 #RGB

use_gpu = torch.cuda.is_available()

class Shim():
    def __init__(self):
        self.MLS = 0


if __name__ == '__main__':
    img = cv2.imread("")

