try:
    import Box2D
    from Demo_gym.envs.box2d.lunar_lander import LunarLander
    from Demo_gym.envs.box2d.lunar_lander import LunarLanderContinuous
    from Demo_gym.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHardcore
    from Demo_gym.envs.box2d.car_racing import CarRacing
except ImportError:
    Box2D = None
