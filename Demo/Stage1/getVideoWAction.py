import os
import time

import Demo_gym
from Demo_gym.utils.play import play
import gym_recording.playback
import numpy as np
import shutil
import os, logging, time, tkinter,cv2
from StateClassifier import darknet
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)

class GetVideoWAction():
    def __init__(self, gameName:str,zoomIndex:int,transposable:bool)-> None:
        self.gameName = gameName
        self.env = Demo_gym.make(gameName)
        self.zoom = zoomIndex
        self.transposable = transposable
        self.videoFrames = []
        self.video_size = (0,0)
        self.fps = 35
        self.out = 0
        self.framId = 0
        self.plyReward = 0
        self.actions = []
        self.recordName = ""
        self.expertState = []
        self.expertAction = []
        self.expertReward = []
        self.expertLocation = []

    def playNrecord(self):
        human_agent_action = 0
        human_wants_restart = False
        human_sets_pause = False
        KEYWORD_TO_KEY = {
            'FIRE': ord(' '),
            'UP': ord('w'),
            'DOWN': ord('s'),
            'LEFT': ord('a'),
            'RIGHT': ord('d')
        }
        play(self.env, zoom=self.zoom)

    def display_trainingData(sellf,path):
        def handle_ep(observations, actions, rewards):
            tmpImg = np.asarray(observations[0])
            cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            # cv2.imshow("",tmpImg[0:190,30:130]) #non-original size
            # tmpImg = cv2.resize(tmpImg,(130*5, 190*5), interpolation = cv2.INTER_CUBIC)
            cv2.imshow("", tmpImg)
            cv2.waitKey(1)
            print(str(actions[0]) + "-" + str(rewards[0]))

        gym_recording.playback.scan_recorded_traces(path, handle_ep)


    def replay(self, path, targetPath):
        self.recordName = path.split("/")[-1]
        newFolder = targetPath+"/"+self.recordName
        imgFolder = newFolder + "/img"
        if os.path.exists(newFolder):
            shutil.rmtree(newFolder)
        print("Svaed at:"+newFolder)
        os.mkdir(newFolder)
        os.mkdir(imgFolder)
        def handle_ep(observations, actions, rewards):
            h, w, _ = observations[0].shape
            tmpImg = np.asarray(observations[0])
            if self.framId > 1:
                cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)

            if rewards[0] >= 0:
                self.framId += 1
                self.expertAction.append(actions[0])
                self.expertReward.append(rewards[0])

                #self.plyReward += int(rewards[0])
                #self.actions.append(str(actions[0]))
                imgpath = imgFolder+"/"+str(self.framId)+".jpg"
                cv2.imwrite(imgpath, tmpImg)
                points = darknet.getState(imgpath)
                print(points)
                self.expertLocation.append(points)  # ToDo

            # cv2.imshow("",tmpImg[0:190,30:130]) #non-original size
            #tmpImg = cv2.resize(tmpImg,(130*5, 190*5), interpolation = cv2.INTER_CUBIC)
            #cv2.imshow("", tmpImg)
            #cv2.waitKey(1)
            print(str(actions[0])+"-"+str(rewards[0]))

        gym_recording.playback.scan_recorded_traces(path, handle_ep)
        print("Finished read")

        # save
        shutil.make_archive(newFolder + "/state", "zip", imgFolder)
        shutil.rmtree(imgFolder)
        print("Saved state")
        self.expertAction = np.asarray(self.expertAction)
        np.save(newFolder + "/action.npy", self.expertAction)
        print("Saved action")
        self.expertReward = np.asarray(self.expertReward)
        np.save(newFolder + "/reward.npy", self.expertReward)
        print("Saved reward")
        self.expertLocation = np.asarray(self.expertLocation)
        np.save(newFolder + "/location.npy", self.expertLocation)
        print("Saved location")

        return newFolder


if __name__ == "__main__":
    x = GetVideoWAction("IceHockey-v0",3,True)
    x.playNrecord()
    #x.replay("/Dro s pTheGame/Demo/Stage1/openai.gym.1566264389.031848.82365","../resources")
    #x = GetVideoWAction('CartPole-v0')

    #x.replay("/Users/remosy/Desktop/DropTheGame/Demo/Stage1/openai.gym.1563643050.743562.78175")

    #x.makeVideo()