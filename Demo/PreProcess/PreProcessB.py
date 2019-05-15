import torch
import numpy as np
import cv2


inWidth = 160
inHeight = 210
inDepth = 3 #RGB

use_gpu = torch.cuda.is_available()

class PreProcessB():
    def __init__(self,data):
        self.imgdata = data

    """
    
    """
    def crop(self):
        return self.imgdata[0:190,30:130]

    """
    Resize video clip frames [w,h]
    """
    def resize(self):
        return cv2.resize(self.imgdata, (80,105))


    """
    Normolise video clip frames
    """
    def normolise(self):
        return self.imgdata


if __name__ == '__main__':
    img = cv2.imread("")

