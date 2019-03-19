from os.path import join

import cv2
import numpy as np
from os import listdir


class GetVideos():
    """
    Classifier network used for both TDC and CMC.

    Parameters
    ----------
    in_channels : int
        Number of features in the input layer. Should match the output
        of TDC and CMC. Defaults to 1024.
    out_channels : int
        Number of features in the output layer. Should match the number
        of classes. Defaults to 6.

    """

    def __init__(self, method:int,path:str,v_num:int):
        if method==0:
            self.savedVideo(path,v_num)
        else:
            self.liveVideo()


    def savedVideo(self, folderPath, n):
        collection = np.empty(shape=(n, 3))
        for vFile in listdir(folderPath):
            if vFile.endswith(".mp4") | vFile.endswith(".mov"):
                path = join(folderPath, vFile)
                collection += self.convertToframes(path)
                print(collection.__len__())

        return collection

     def convertToframes(self,vPath):
         frames = []
         return frames


     #def liveVideo(self):




