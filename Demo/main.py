# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym
from gym.utils.play import play, PlayPlot

import cv2 #openCV

import numpy



class IceHockey:
    def __init__(self):
        super().__init__()
        self.env0 = gym.make("IceHockey-v0")
        
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
        return [rew,]   
    
    def playGame(self,deEnv):
        env_plotter = EnvPlay.PlayPlot(callback, 30 * 5, ["reward"])
        play(deEnv,zoom=3,callback=env_plotter.callback)
        
        



