class Base:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def reset(self):
        raise NotImplementedError

    def get_obs(self, s):
        raise NotImplementedError

    def get_reward(self, s):
        raise NotImplementedError

    def get_done(self):
        return False

    @property
    def obs_shape(self):
        raise NotImplementedError
