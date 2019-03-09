# Install Anaconda
[Website](https://www.anaconda.com/distribution/#download-section) (Follow the instruction from this link)

# Set Up via Anaconda
Open the "Environments", click the "Play" button, then click the  
## pip
```
pip install gym
Pip install gym_recording
pip install -e '.[atari]' 
Pip install pygame
pip install box2d
pip install box2d-py
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
## Test Gym on Console
```
import main as x
AI = x.IceHockey()
AI.get Info(AI.env4)
```
## Play Gym
[Gym **play** function](https://github.com/openai/gym/blob/master/gym/utils/play.py#L26)

# Issues
```
print(deEnv.unwrapped.actions_space) #useless
```



Method 2:install via homebrew
https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/


Alert
Restar Spyder every time once a new tool is installed by “pip install”