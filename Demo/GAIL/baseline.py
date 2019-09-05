
import Demo_gym
from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset

from stable_baselines import TRPO
from stable_baselines.gail import generate_expert_traj

# Generate expert trajectories (train expert)
#model = TRPO('MlpPolicy', 'IceHockey-v0', verbose=1)
# Train for 100 timesteps and record 10 trajectories
# run about 5 mins
# all the data will be saved in 'expert_pendulum.npz' file
#generate_expert_traj(model, 'expert_icehockey', n_timesteps=100, n_episodes=10)
# Load the expert dataset
dataset = ExpertDataset(expert_path='expert_icehockey.npz', traj_limitation=10, verbose=1)

model = GAIL("MlpPolicy", 'IceHockey-v0', dataset, verbose=1)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=1000)
model.save("gail_icehockey")
print("SAVED")
del model # remove to demonstrate saving and loading

model = GAIL.load("gail_icehockey")

env = Demo_gym.make('IceHockey-v0')
obs = env.reset()
while True:
  action, _states = model.predict(obs)
  print(action)
  obs, rewards, dones, info = env.step(action)
  env.render()
