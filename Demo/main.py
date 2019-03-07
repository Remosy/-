# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym
from gym.utils.play import play, PlayPlot
from gym_recording.wrappers import TraceRecordingWrappe
from gym_recording import playback, storage_s3

import cv2 #openCV

import numpy



class IceHockey:
    def __init__(self):
        super().__init__()
        self.env0 = gym.make("IceHockey-v0")
        self.env0 = TraceRecordingWrapper(self.env0)
        s3url = storage_s3.upload_recording(self.env0.directory, 
                                             self.env0.spec.id, 
                                             'IceHockey_records')
         playback.scan_recorded_traces(storage_s3.download_recording(s3url))
        
        '''
        self.env4 = gym.make("IceHockey-v4")
    
        self.envR0 = gym.make("IceHockey-ram-v0")
        self.envR4 = gym.make("IceHockey-ram-v4") 
        
        self.envNF0 = gym.make("IceHockeyNoFrameskip-v0")
        self.envNF4 = gym.make("IceHockeyNoFrameskip-v4") 
        '''
        
    def getInfo(self,deEnv):
        print(deEnv.action_space)
        print(deEnv.unwrapped.get_action_meanings())
        print(deEnv.unwrapped.get_keys_to_action())
        print(deEnv.action_space.sample())
        
        print(deEnv.observation_space)
    
    def callback(self,obs_t, obs_tp1, rew, done, info):
        return [obs_t,]   
    
    def playGame(self,deEnv):
        env_plotter = EnvPlay.PlayPlot(callback, 30 * 5, ["reward"])
        play(deEnv,zoom=3,callback=)
        
        



