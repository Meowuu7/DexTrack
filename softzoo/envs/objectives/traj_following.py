import numpy as np
import torch
import taichi as ti

from .base import Base
from ...engine.renderer.ggui_renderer import GGUIRenderer


class TrajectoryFollowing(Base):
    def __init__(self, env, config):
        super().__init__(env, config)

        self.config['start_velocity'] = self.config.get('start_velocity', [0.1, 0., 0.])
        self.config['end_position'] = self.config.get('end_position', [0.5, 0., 0.])
        self.config['end_position_rand_range'] = self.config.get('end_position_rand_range', [0., 0., 0.])
        self.config['end_velocity'] = self.config.get('end_velocity', [0.1, 0., 0.])
        self.config['end_velocity_rand_range'] = self.config.get('end_velocity_rand_range', [0., 0., 0.])
        self.config['end_substep'] = self.config.get('end_substep', self.env.max_steps)
        self.config['weight_norm'] = self.config.get('weight_norm', 0.1)
        self.config['weight_direction'] = self.config.get('weight_direction', 0.9)
        self.config['sigma_norm'] = self.config.get('sigma_norm', 0.05)
        self.config['sigma_direction'] = self.config.get('sigma_direction', 0.1)
        self.config['reward_mode'] = self.config.get('reward_mode', 'velocity_separate')

        self.max_episode_steps = self.config.get('max_episode_steps', self.env.max_steps)
        assert self.max_episode_steps <= self.env.max_steps
        assert self.config['max_episode_steps'] != torch.inf, 'Maximal episode step is infinite'

        self.draw_x = False
        self.draw_v_avg = True
        self.draw_v_avg_raw = True
        self.draw_v_avg_proj = True

        if hasattr(self.env, 'renderer'):
            self.env.renderer.objective_render_init = self.objective_render_init
            self.env.renderer.objective_render = self.objective_render

    def reset(self):
        s = self.env.sim.solver.current_s
        x = self.env.design_space.get_x(s)
        sp = x.mean(0).float()
        sv = torch.tensor(self.config['start_velocity'])
        sa = torch.zeros((self.env.sim.solver.dim,))
        dep_rand = (torch.rand((self.env.sim.solver.dim,)) - 0.5) * 2. * torch.tensor(self.config['end_position_rand_range'])
        ep = sp + torch.tensor(self.config['end_position']) + dep_rand
        dev_rand = (torch.rand((self.env.sim.solver.dim,)) - 0.5) * 2. * torch.tensor(self.config['end_velocity_rand_range'])
        ev = torch.tensor(self.config['end_velocity']) + dev_rand
        ea = torch.zeros((self.env.sim.solver.dim,))
        et = torch.tensor([self.env.substep_dt * self.config['end_substep']])
        
        qps = []
        for i in range(self.env.sim.solver.dim):
            qp = QuinticPolynomials(sp[i][None,None], sv[i][None,None], sa[i][None,None], ep[i][None,None], ev[i][None,None], ea[i][None,None], et)
            qps.append(qp)
        self.qps = qps

        self.step_cnt = 0

        # for rendering
        traj = []
        for s in range(self.config['end_substep']):
            traj.append(self.get_x_tgt(s).data.cpu().numpy())
        traj = np.array(traj)
        traj = np.concatenate([traj, np.ones((traj.shape[0],1))], axis=1)
        self.traj_for_render = traj

    def get_obs(self, s):
        obs = torch.cat([self.get_x_tgt(s), self.get_v_tgt(s)])
        return obs

    def get_reward(self, s):
        self.step_cnt += 1
        if 'velocity' in self.config['reward_mode']:
            v = self.env.design_space.get_v(s)
            v_avg = v.mean(0)

            sm1 = max(s - 1, 0)
            v_tgt = self.get_v_tgt(sm1)
        else:
            x = self.env.design_space.get_x(s)
            x_avg = x.mean(0)

            sm1 = max(s - 1, 0)
            x_tgt = self.get_x_tgt(sm1)

        if self.config['reward_mode'] == 'velocity_separate_exp':
            v_avg_norm = v_avg.norm()
            v_avg_direction = v_avg / v_avg_norm

            v_tgt_norm = v_tgt.norm()
            v_tgt_direction = v_tgt / v_tgt_norm

            v_norm_diff = torch.abs(v_tgt_norm - v_avg_norm)
            rew_norm = torch.exp(-v_norm_diff / self.config['sigma_norm'])

            v_direction_diff = ((v_tgt_direction - v_avg_direction) ** 2).sum()
            rew_direction = torch.exp(-v_direction_diff / self.config['sigma_direction'])
            
            rew = self.config['weight_norm'] * rew_norm + self.config['weight_direction'] * rew_direction
        elif self.config['reward_mode'] == 'velocity_rmse':
            rew = (v_avg - v_tgt).norm()
        elif self.config['reward_mode'] == 'velocity_separate_linear':
            v_avg_norm = v_avg.norm()
            v_avg_direction = v_avg / v_avg_norm
            v_avg_direction = v_avg_direction.float()

            v_tgt_norm = v_tgt.norm()
            v_tgt_direction = v_tgt / v_tgt_norm

            rew_norm = -((v_tgt_norm - v_avg_norm) ** 2)
            rew_direction = v_avg_direction.dot(v_tgt_direction)

            rew = self.config['weight_norm'] * rew_norm + self.config['weight_direction'] * rew_direction
        elif self.config['reward_mode'] == 'waypoint_sqsum':
            rew = -((x_avg - x_tgt) ** 2).sum()
        else:
            raise ValueError('Unrecognized reward mode'.format(self.config['reward_mode']))

        rew = rew.item()

        return rew

    def get_x_tgt(self, s):
        # get target waypoint
        t = self.env.substep_dt * s
        x = []
        for qp in self.qps:
            xi = qp.ps(t)[0,0]
            x.append(xi)
        x = torch.cat(x)
        return x

    def get_v_tgt(self, s):
        # get target velocity
        t = self.env.substep_dt * s
        v = []
        for qp in self.qps:
            vi = qp.ps_dot(t)[0,0]
            v.append(vi)
        v = torch.cat(v)
        return v

    def get_done(self):
        return not (self.step_cnt < self.max_episode_steps)

    @property
    def obs_shape(self):
        return (self.env.sim.solver.dim * 2,)

    def objective_render_init(self, data, n_bodies):
        bodies_n_particles, bodies_particle_radius, bodies_particle_offset, \
            bodies_color, bodies_needs_smoothing, bodies_draw_density, bodies_draw_diffuse, \
                bodies_draw_ellipsoids, bodies_draw_points, bodies_anisotropy_scale = data

        if 'waypoint' in self.config['reward_mode']:
            # trajectory
            bodies_n_particles.append(self.config['end_substep'])
            bodies_particle_radius.append(0.003)
            bodies_particle_offset.append(bodies_particle_offset[-1] + bodies_n_particles[-2])
            bodies_color.append([1., 0., 0., 1.])
            bodies_needs_smoothing.append(False)
            bodies_draw_density.append(False)
            bodies_draw_diffuse.append(False)
            bodies_draw_ellipsoids.append(False)
            bodies_draw_points.append(True)
            bodies_anisotropy_scale.append(1.)
            n_bodies += 1

            # target waypoint
            bodies_n_particles.append(1)
            bodies_particle_radius.append(0.01)
            bodies_particle_offset.append(bodies_particle_offset[-1] + bodies_n_particles[-2])
            bodies_color.append([0., 1., 0., 1.])
            bodies_needs_smoothing.append(False)
            bodies_draw_density.append(False)
            bodies_draw_diffuse.append(False)
            bodies_draw_ellipsoids.append(False)
            bodies_draw_points.append(True)
            bodies_anisotropy_scale.append(1.)
            n_bodies += 1
        elif 'velocity' in self.config['reward_mode']:
            n_particles = 100
            radius = 0.003

            self.render_velocity_n_particles = n_particles

            # ego velocity
            bodies_n_particles.append(n_particles)
            bodies_particle_radius.append(radius)
            bodies_particle_offset.append(bodies_particle_offset[-1] + bodies_n_particles[-2])
            bodies_color.append([0., 1., 0., 1.])
            bodies_needs_smoothing.append(False)
            bodies_draw_density.append(False)
            bodies_draw_diffuse.append(False)
            bodies_draw_ellipsoids.append(False)
            bodies_draw_points.append(True)
            bodies_anisotropy_scale.append(1.)
            n_bodies += 1

            # target velocity
            bodies_n_particles.append(n_particles)
            bodies_particle_radius.append(radius)
            bodies_particle_offset.append(bodies_particle_offset[-1] + bodies_n_particles[-2])
            bodies_color.append([1., 0., 0., 1.])
            bodies_needs_smoothing.append(False)
            bodies_draw_density.append(False)
            bodies_draw_diffuse.append(False)
            bodies_draw_ellipsoids.append(False)
            bodies_draw_points.append(True)
            bodies_anisotropy_scale.append(1.)
            n_bodies += 1

        return n_bodies

    def objective_render(self, data):
        if 'waypoint' in self.config['reward_mode']:
            traj = self.traj_for_render

            s = self.env.sim.solver.current_s
            x_tgt = self.get_x_tgt(s).data.cpu().numpy()
            x_tgt = np.concatenate([x_tgt, [0]])[None,:]

            data = np.concatenate([data, traj, x_tgt], 0)
        elif 'velocity' in self.config['reward_mode']:
            s = self.env.sim.solver.current_s
            x = self.env.design_space.get_x(s)
            x_avg = x.mean(0)

            v_tgt = self.get_v_tgt(s)
            v_avg = self.env.design_space.get_v_avg(s)
            
            ego_line_0 = x_avg
            ego_line_1 = x_avg + v_avg / v_avg.norm() * 0.1

            tgt_line_0 = x_avg
            tgt_line_1 = x_avg + v_tgt / v_tgt.norm() * 0.1

            ego_line = np.linspace(ego_line_0.numpy(), ego_line_1.numpy(), self.render_velocity_n_particles)
            tgt_line = np.linspace(tgt_line_0.numpy(), tgt_line_1.numpy(), self.render_velocity_n_particles)

            ego_line = np.concatenate([ego_line, np.zeros((ego_line.shape[0],1))], axis=1)
            tgt_line = np.concatenate([tgt_line, np.zeros((tgt_line.shape[0],1))], axis=1)

            data = np.concatenate([data, ego_line, tgt_line], axis=0)
        
        return data

    def render(self):
        if not isinstance(self.env.renderer, GGUIRenderer):
            return

        s = self.env.sim.solver.current_s
        v_tgt = self.get_v_tgt(s)

        x = self.env.design_space.get_x(s)
        x_avg = x.mean(0)

        if not hasattr(self, 'line_points_v'):
            self.line_points_v = ti.Vector.field(self.env.sim.solver.dim, dtype=ti.f32, shape=2)
        self.line_points_v[0] = x_avg
        self.line_points_v[1] = x_avg + v_tgt / v_tgt.norm() * 0.1

        self.env.renderer.scene.lines(self.line_points_v, color=(0.28, 0.68, 0.99), width=5.0)

        if self.draw_x:
            if not hasattr(self, 'line_points_x'):
                self.line_points_x = ti.Vector.field(self.env.sim.solver.dim, dtype=ti.f32, shape=self.config['end_substep'])
                xs = []
                for ss in range(self.config['end_substep']):
                    tt = self.env.substep_dt * ss
                    x = []
                    for qp in self.qps:
                        xi = qp.ps(tt)[0,0]
                        x.append(xi)
                    xs.append(x)
                xs = torch.tensor(xs)
                self.line_points_x.from_torch(xs)

            self.env.renderer.scene.lines(self.line_points_x, color=(0.48, 0.99, 0.68), width=5.0)

            if not hasattr(self, 'particle_point_x_avg'):
                self.particle_point_x_avg = ti.Vector.field(self.env.sim.solver.dim, dtype=ti.f32, shape=(1,))
            self.particle_point_x_avg[0] = x_avg
            self.env.renderer.scene.particles(self.particle_point_x_avg, color=(0., 1., 0.5), radius=0.005)

            if not hasattr(self, 'particle_point_x_tgt'):
                self.particle_point_x_tgt = ti.Vector.field(self.env.sim.solver.dim, dtype=ti.f32, shape=(1,))
            x_tgt = self.get_x_tgt(s)
            self.particle_point_x_tgt[0] = x_tgt
            self.env.renderer.scene.particles(self.particle_point_x_tgt, color=(1., 0.5, 0.), radius=0.005)

        if self.draw_v_avg:
            if not hasattr(self, 'line_points_v_avg'):
                self.line_points_v_avg = ti.Vector.field(self.env.sim.solver.dim, dtype=ti.f32, shape=2)

            v_avg = self.env.design_space.get_v_avg(s)
            self.line_points_v_avg[0] = x_avg
            self.line_points_v_avg[1] = x_avg + v_avg / v_avg.norm() * 0.1

            self.env.renderer.scene.lines(self.line_points_v_avg, color=(0.99, 0.28, 0.68), width=5.0)

            if self.draw_v_avg_raw: # non-normalized velocity
                if not hasattr(self, 'line_points_v_avg_raw'):
                    self.line_points_v_avg_raw = ti.Vector.field(self.env.sim.solver.dim, dtype=ti.f32, shape=2)
                self.line_points_v_avg_raw[0] = x_avg
                self.line_points_v_avg_raw[1] = x_avg + v_avg

                self.env.renderer.scene.lines(self.line_points_v_avg_raw, color=(0.99, 0.0, 0.68), width=5.0)

            if self.draw_v_avg_proj: # projection on heading direction
                if not hasattr(self, 'line_points_v_avg_proj'):
                    self.line_points_v_avg_proj = ti.Vector.field(self.env.sim.solver.dim, dtype=ti.f32, shape=2)
                v_avg_proj = self.env.design_space.get_v_avg(s, mode=1)
                self.line_points_v_avg_proj[0] = x_avg
                self.line_points_v_avg_proj[1] = x_avg + v_avg_proj / v_avg.norm() * 0.1

                self.env.renderer.scene.lines(self.line_points_v_avg_proj, color=(0.99, 0.99, 0.0), width=5.0)


class QuinticPolynomials:
    def __init__(self,
        sp, # start position; shape = (Batch, 1)
        sv, # start velocity; shape = (Batch, 1)
        sa, # start acceleration; shape = (Batch, 1)
        ep, # end position; shape = (Batch, 1)
        ev, # end velocity; shape = (Batch, 1)
        ea, # end accleration; shape = (Batch, 1)
        et, # end time; shape = (Batch,)
    ):
        self._a0 = sp
        self._a1 = sv
        self._a2 = sa / 2.

        A_r0 = torch.stack([et ** 3, et ** 4, et ** 5], dim=1) # row 0
        A_r1 = torch.stack([3 * et ** 2, 4 * et ** 3, 5 * et ** 4], dim=1) # row 1
        A_r2 = torch.stack([6 * et, 12 * et ** 2, 20 * et ** 3], dim=1) # row 2
        A = torch.stack([A_r0, A_r1, A_r2], dim=1)

        b = torch.stack([(ep - self._a0) - self._a1 * et - self._a2 * et ** 2,
                         (ev - self._a1) - 2 * self._a2 * et,
                         (ea - 2 * self._a2)], dim=1)

        x = torch.linalg.solve(A, b)
        self._a3 = x[:, 0]
        self._a4 = x[:, 1]
        self._a5 = x[:, 2]

        self._a0 = self._a0[:, None]
        self._a1 = self._a1[:, None]
        self._a2 = self._a2[:, None]
        self._a3 = self._a3[:, None]
        self._a4 = self._a4[:, None]
        self._a5 = self._a5[:, None]

        self._A = A
        self._b = b
        self._x = x
        self._sp = sp
        self._sv = sv
        self._sa = sa
        self._ep = ep
        self._ev = ev
        self._ea = ea
        self._et = et

    def ps(self, ts):
        ps = self._a0 + self._a1 * ts + self._a2 * ts ** 2 + \
             self._a3 * ts ** 3 + self._a4 * ts ** 4 + self._a5 * ts ** 5
        return ps

    def ps_dot(self, ts):
        ps_dot = self._a1 + 2 * self._a2 * ts + 3 * self._a3 * ts ** 2 + \
                 4 * self._a4 * ts ** 3 + 5 * self._a5 * ts ** 4
        return ps_dot

    def ps_ddot(self, ts):
        ps_ddot = 2 * self._a2 + 6 * self._a3 * ts + 12 * self._a4 * ts ** 2 + 20 * self._a5 * ts ** 3
        return ps_ddot
    
    def ps_dddot(self, ts):
        ps_dddot = 6 * self._a3 + 24 * self._a4 * ts + 60 * self._a5 * ts ** 2
        return ps_dddot
