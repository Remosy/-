from os.path import join

import cv2
import numpy as np
from os import listdir


class GetVideos:

    def __init__(self, method:int,path:str,v_num:int)-> None:
        self.videoClips = self.savedVideo(path, v_num)

    def savedVideo(self, folderPath, n) -> np.numarray:
        folderPath = "/Users/remosy/Desktop/DropTheGame/Demo/resources"
        collection = []
        for vFile in listdir(folderPath):
            if vFile.endswith(".mp4") | vFile.endswith(".mov"):
                path = join(folderPath, vFile)
                print(path)
                collection.append(self.convertToframes(path))
                print(collection.__len__())
        return collection

    def convertToframes(self,vPath)-> np.numarray:
        video = cv2.VideoCapture(vPath) #load video
        frameNum = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        videoWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        videoHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((frameNum,videoHeight,videoWidth, 3), np.dtype('uint8'))
        #video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        #video.set(cv2.CAP_PROP_FPS, fps)
        fc = 0
        ret = True
        while (fc < frameNum and ret):
            ret, frames[fc] = video.read()
            #frames[fc] = frame
            fc += 1
        video.release()
        cv2.destroyAllWindows()
        print("FrameNum ="+str(frameNum)+"; VideoWidth ="+str(videoWidth)+"; VideoHeight="+str(videoHeight))
        return frames


    def liveVideo(self):
        return ""

if __name__ == "__main__":
    x = GetVideos(0, "", 0)
    np.save("videoFrames", x.videoClips)
    print("Done: convert videos to frames in type:"+str(type(x.videoClips[0])))