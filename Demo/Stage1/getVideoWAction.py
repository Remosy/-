import os
import time

import gym
from gym.utils.play import play
from gym_recording.wrappers import TraceRecordingWrapper
from gym_recording import playback, storage_s3

class GetVideoWAction():
    def __init__(self, gameName:str)-> None:
        self.gameName = gameName
        self.playNrecord(self.gameName)


    def playNrecord(self, gameName):
        human_agent_action = 0
        human_wants_restart = False
        human_sets_pause = False
        KEYWORD_TO_KEY = {
            'NOOP':ord(''),
            'FIRE': ord(' '),
            'UP': ord('w'),
            'DOWN': ord('s'),
            'LEFT': ord('a'),
            'RIGHT': ord('d')
        }
        env = gym.make(self.gameName)



if __name__ == "__main__":
    x = GetVideoWAction("IceHockey-v0")
