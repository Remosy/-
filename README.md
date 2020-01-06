### Supervisor: Dr. Penny Kyburz
This project allows users to use image-state in a specific game: Ice Hockey (Atari).

This project also combined the YOLO to extract the location as the state in a specific game: Ice Hockey (Atari).

Algorithms: GAIL, off-policy PPO
Youtube:https://www.youtube.com/watch?v=rgYOhhg0DUk

# Warnning
* This project only supports Mac user and Linux user
* This project uses Python3

# Step 0
## Install Anaconda
[Website](https://www.anaconda.com/distribution/#download-section) (Follow the instruction from this link)

Please ensure you have installed the anaconda environment.

## Install Pycharm
[Website](https://www.jetbrains.com/pycharm/download) (Follow the instruction from this link)
Please ensure you have installed the Pycharm environment

## Setup the insterpreter in Pycharm
Open our code with Pycharm
1. Click "Pycharm" (Top menu)
2. Click "Preference"
3. Find "Project Interpreter" under your project
4. Add "Conda Envionment" one by choosig the "Existing environment"
5. Add address like "/User/USERNAME/anaconda3/bin/python"
6. Finally click "Apply"

# Step 1

## Edit file before next step
1. Find file and open the file "DropTheGame/Demo/StateClassifier/darknet.py" 
2. Go to **line 48**. 
3. If you are using Linux: 
	```
	lib = CDLL("StateClassifier/libdarknet_linux.so",RTLD_GLOBAL)
	```
	
	If you are using Mac:
	```
	lib = CDLL("StateClassifier/libdarknet_mac.so",RTLD_GLOBAL)
	```
  
## pip
```
pip install -r /DropTheGame/Demo/requirement.txt
```
## Test our code on Pycharm
You can run "Start.py" directly. We have added some example codes under the main function. Uncomments any of them to try.

## Test our code on Terminal
Firstly, you have to make sure you are inside "DropTheGame/Demo"
```
python Start
```

## Test our code on Python Console
The functions listed in Start.py can be called as following

```
from Start import IceHockey
IH = IceHockey()
IH.AIplay(True,"loc")
```

## conda
```
conda install -c akode gym
conda install -c conda-forge opencv
```
## brew
```
brew install swig
```

# Check Gym
```
from gym import envs
gym.envs.registry
```

## Play Gym
[Gym **play** function](https://github.com/openai/gym/blob/master/gym/utils/play.py#L26)


## If you used Spyder
Restar Spyder every time once a new tool is installed by “pip install”

# Issues
```
print(deEnv.unwrapped.actions_space) #useless
```

Method 2:install via homebrew
https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/

# Acknowledgments
The GAIL and PPO codes are modified based on part of [OpenAI Baseline](https://github.com/openai/baselines)
