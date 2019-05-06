import os
import time

import Demo_gym
from Demo_gym.utils.play import play
#from gym_recording.wrappers import TraceRecordingWrapper
#from gym_recording import playback, storage_s3

#import gym
#from gym.utils.play import play, PlayPlot
import os, logging, time, tkinter, cv2
from gym_recording.wrappers import TraceRecordingWrapper
from gym_recording.playback import scan_recorded_traces
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
logger.addHandler(console)

class GetVideoWAction():
    def __init__(self, gameName:str)-> None:
        self.gameName = gameName
        self.env = Demo_gym.make(gameName)
        self.env = TraceRecordingWrapper(self.env)
        self.recordPath = self.env.directory


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
        play(self.env, zoom=3)


    def replay(self,path):
        if path!="":
            originPath = self.recordPath
            self.recordPath = path

        #counts = [0, 0]
        def handle_ep(observations, actions, rewards):
            #counts[0] += 1
            #counts[1] += observations[0].shape
            #logger.debug('Observations.shape={}, actions.shape={}, rewards.shape={}', str(observations[0].shape), actions.shape,
                         #rewards.shape)
            print(str(observations.shape)+"\n")
        scan_recorded_traces(self.recordPath, handle_ep)
        self.recordPath = originPath

if __name__ == "__main__":
    x = GetVideoWAction("IceHockey-v0")
    #x.playNrecord()
    #x = GetVideoWAction('CartPole-v0')
    #x.replay("/Users/remosy/Desktop/"+"openai.Demo_gym.1557148506.295021.73364")