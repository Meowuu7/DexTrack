import torch
import taichi as ti

from .base import Base


class MoveForward(Base):
    def __init__(self, env, config):
        super().__init__(env, config)

        self.config['reward_mode'] = self.config.get('reward_mode', 'per_step_velocity')
        assert self.config['reward_mode'] in ['final_step_position', 'per_step_velocity']
        self.config['forward_direction'] = self.config.get('forward_direction', [1., 0., 0.])
        self.max_episode_steps = self.config.get('max_episode_steps', self.env.max_steps)
        assert self.max_episode_steps <= self.env.max_steps
        # assert self.config['max_episode_steps'] != torch.inf, 'Maximal episode step is infinite'

    def reset(self):
        if self.config['reward_mode'] == 'final_step_position':
            s = 0
            x = self.env.design_space.get_x(s)
            self.x0_avg = x.mean(0).float()
        self.step_cnt = 0

    def get_obs(self, s):
        return None

    def get_reward(self, s):
        self.step_cnt += 1
        if self.config['reward_mode'] == 'per_step_velocity':
            v_avg = self.env.design_space.get_v_avg(s, mode=1)
            forward_dir = torch.tensor(self.config['forward_direction']).to(v_avg)
            rew = (v_avg * forward_dir).sum().item()
        else:
            if self.step_cnt >= self.max_episode_steps:
                x = self.env.design_space.get_x(s)
                x_avg = x.mean(0)
                forward_dir = torch.tensor(self.config['forward_direction']).to(x_avg)
                rew = ((x_avg - self.x0_avg) * forward_dir).sum().item()
            else:
                rew = 0.

        return rew

    def get_done(self):
        return not (self.step_cnt < self.max_episode_steps)

    @property
    def obs_shape(self):
        return None



