import os
import time

import Demo_gym
import cv2
from Demo_gym.utils.play import play
#from gym_recording.wrappers import TraceRecordingWrapper
#from gym_recording import playback, storage_s3
import gym_recording.playback
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

    def replay(self,path):
        #counts = [0, 0]
        if self.zoom is not None:
            rendered = self.env.render(mode='rgb_array')
            self.video_size = [rendered.shape[1], rendered.shape[0]]
            self.video_size = int(self.video_size[0] * self.zoom), int(self.video_size[1] * self.zoom)
            self.out = cv2.VideoWriter(strftime("%Y-%m-%d_%H_%M", gmtime()) + "_video.avi",
                                    cv2.VideoWriter_fourcc(*'DIVX'), self.fps, self.video_size)
        def handle_ep(observations, actions, rewards):
            tmpImg = observations[0]
            for i in range(0,len(observations)):
                self.out.write(observations[i])
            #if self.size == -1:
                #height, width, layers = self.observations[0].shape
                #size = (width, height)
            #RGB_obs = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2RGB)
            #self.videoFrames.append(tmpImg)
            cv2.imshow("",tmpImg[0:190,30:130]) #non-original size
            cv2.waitKey(1)
            print(str(actions[0:1])+"-"+str(rewards[0:1]))

        gym_recording.playback.scan_recorded_traces(path, handle_ep)
        cv2.destroyAllWindows()

    def makeVideo(self):
        self.out.release()
        self.videoFrames = []
        print("video is done")


if __name__ == "__main__":
    x = GetVideoWAction("IceHockey-v0",3,True)
    #x.playNrecord()
    #x = GetVideoWAction('CartPole-v0')

    x.replay("/Users/remosy/Dropbox/openai.gym.1557159104.119814.77942")
    x.makeVideo()