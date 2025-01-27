import torch
import taichi as ti

from .base import Base


class MoveInCircles(Base):
    def __init__(self, env, config):
        super().__init__(env, config)

        self.draw_v_avg = True
        self.draw_v_avg_proj = True
        self.up_direction = self.env.sim.device.tensor([0., 1., 0.])
        self.max_episode_steps = self.config.get('max_episode_steps', self.env.max_steps)
        assert self.max_episode_steps <= self.env.max_steps
        assert self.config['max_episode_steps'] != torch.inf, 'Maximal episode step is infinite'

    def reset(self):
        self.step_cnt = 0

    def get_obs(self, s):
        obs = torch.zeros((1,)) # TODO: perhaps sending inertia?!
        return obs

    def get_reward(self, s):
        self.step_cnt += 1
        x = self.env.design_space.get_x(s)
        x_avg = x.mean(0)
        x_centered = x - x_avg
        v_tan_dir = self.up_direction[None,:].cross(x_centered)
        v_tan_dir = v_tan_dir / (v_tan_dir.norm(dim=1) + 1e-8)[:,None]
        v = self.env.design_space.get_v(s)
        rew = (v_tan_dir * v).sum(1).mean()
        rew = rew.item()
        return rew

    def get_done(self):
        return not (self.step_cnt < self.max_episode_steps)

    @property
    def obs_shape(self):
        return (1,) # TODO: define shape

    def render(self):
        if not hasattr(self.env.renderer, 'scene'):
            return

        s = self.env.sim.solver.current_s

        x = self.env.design_space.get_x(s)
        x_avg = x.mean(0)

        if self.draw_v_avg:
            if not hasattr(self, 'line_points_v_avg'):
                self.line_points_v_avg = ti.Vector.field(self.env.sim.solver.dim, dtype=ti.f32, shape=2)

            v_avg = self.env.design_space.get_v_avg(s)
            self.line_points_v_avg[0] = x_avg
            self.line_points_v_avg[1] = x_avg + v_avg / v_avg.norm() * 0.1

            self.env.renderer.scene.lines(self.line_points_v_avg, color=(0.99, 0.28, 0.68), width=5.0)

        if self.draw_v_avg_proj: # projection on heading direction
            if not hasattr(self, 'line_points_v_avg_proj'):
                self.line_points_v_avg_proj = ti.Vector.field(self.env.sim.solver.dim, dtype=ti.f32, shape=2)
            v_avg_proj = self.env.design_space.get_v_avg(s, mode=1)
            self.line_points_v_avg_proj[0] = x_avg
            self.line_points_v_avg_proj[1] = x_avg + v_avg_proj / v_avg.norm() * 0.1

            self.env.renderer.scene.lines(self.line_points_v_avg_proj, color=(0.99, 0.99, 0.0), width=5.0)
