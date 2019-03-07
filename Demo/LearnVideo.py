# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2 as cv #penCV

import numpy as np



class LearnVideo:
    def __init__(self):
        super().__init__()
        
        
    def getInfo(self,deEnv):
        print(deEnv.action_space)
        print(deEnv.unwrapped.get_action_meanings())
        print(deEnv.unwrapped.get_keys_to_action())
        print(deEnv.action_space.sample())
        
        print(deEnv.observation_space)
    
   
        
        



