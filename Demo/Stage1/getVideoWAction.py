import os
import time

import Demo_gym
import cv2
from Demo_gym.utils.play import play
#from gym_recording.wrappers import TraceRecordingWrapper
#from gym_recording import playback, storage_s3
import gym_recording.playback


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
        play(self.env, self.zoom)

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
            video_size = [rendered.shape[1], rendered.shape[0]]
            video_size = int(video_size[0] * self.zoom), int(video_size[1] * self.zoom)
        def handle_ep(observations, actions, rewards):
            tmpImg = observations[0]
            cv2.imshow("",tmpImg[0:190,30:130])
            cv2.waitKey(1)
            print(str(actions[0:1])+"-"+str(rewards[0:1]))

        gym_recording.playback.scan_recorded_traces(path, handle_ep)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    x = GetVideoWAction("IceHockey-v0",3,True)
    #x.playNrecord()
    #x = GetVideoWAction('CartPole-v0')
    x.replay("/Users/remosy/Desktop/"+"openai.gym.1557159104.119814.77942")