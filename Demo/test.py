from gym.utils.play import play, PlayPlot

def callback(obs_t, obs_tp1, action, rew, done, info):
    return [rew,]

env_plotter = PlayPlot(callback, 30 * 5, ["reward"])

env = gym.make("PongNoFrameskip-v4")