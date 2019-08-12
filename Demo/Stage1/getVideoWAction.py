import os
import time

import Demo_gym
import cv2 as cv2
from Demo_gym.utils.play import play
#from gym_recording.wrappers import TraceRecordingWrapper
#from gym_recording import playback, storage_s3
import gym_recording.playback
import numpy as np
import queue
from time import gmtime, strftime


import os, logging, time, tkinter, cv2
from gym_recording.wrappers import TraceRecordingWrapper
from gym_recording.playback import scan_recorded_traces
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)
import pygame

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
        self.expertState = queue.Queue()
        self.expertAction = queue.Queue()
        self.expertReward = []
        #self.env = TraceRecordingWrapper(self.env)
        #self.recordPath = self.env.directory



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

    def display_arr(sellf,screen, arr, video_size, transpose):
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, video_size)
        screen.blit(pyg_img, (0, 0))

    def replay(self, path, targetPath):
        self.recordName = path.split("/")[-1]
        newFolder = targetPath+"/"+self.recordName
        print("Svaed at:"+newFolder)
        #os.mkdir(imageFolder)
        def handle_ep(observations, actions, rewards):
            self.framId += 1
            h, w, _ = observations[0].shape
            tmpImg = np.asarray(observations[0])
            cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            #self.expertState.put(tmpImg)
            self.expertAction.put(actions[0])
            self.expertReward.append(rewards[0])
            #self.videoFrames.append(tmpImg)
            #self.plyReward += int(rewards[0])
            #self.actions.append(str(actions[0]))

            cv2.imwrite(newFolder+"/"+str(self.framId)+".jpg", tmpImg)
            
            # cv2.imshow("",tmpImg[0:190,30:130]) #non-original size
            #tmpImg = cv2.resize(tmpImg,(130*5, 190*5), interpolation = cv2.INTER_CUBIC)
            #cv2.imshow("", tmpImg)
            #cv2.waitKey(1)
            print(str(actions[0])+"-"+str(rewards[0]))

        gym_recording.playback.scan_recorded_traces(path, handle_ep)
        cv2.destroyAllWindows()

        #save
        print("Finished Queue")
        self.expertAction = list(self.expertAction.queue)
        self.expertAction = np.asarray(self.expertAction)
        np.save(newFolder + "_action.npy", self.expertAction)
        print("Saved action")
        self.expertReward = np.asarray(self.expertReward)
        np.save(newFolder + "_reward.npy", self.expertReward)
        print("Saved reward")

        """
        
        file = open(targetPath+"/" + self.recordName + ".txt", "w+")
        file.write(frameLen + '\n')
        file.write(str(self.plyReward)+'\n')
        file.write(','.join(self.actions))
        file.close()
        print("saved video: "+self.recordName)
        exit(0)
        """
        return self.expertState, self.expertAction, self.expertReward


if __name__ == "__main__":
    x = GetVideoWAction("IceHockey-v0",3,True)
    #x.playNrecord()
    x.replay("openai.gym.1563812853.178726.40887","../resources")
    #x = GetVideoWAction('CartPole-v0')

    #x.replay("/Users/remosy/Desktop/DropTheGame/Demo/Stage1/openai.gym.1563643050.743562.78175")

    #x.makeVideo()