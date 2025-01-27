"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

try:
    from diffusion.nn import mean_flat
    from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
except:
    from softzoo.diffusion.diffusion.nn import mean_flat
    from softzoo.diffusion.diffusion.losses import normal_kl, discretized_gaussian_log_likelihood

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
        
    def q_sample_xs(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None: # the noise 
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start, _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        # return (
        #     _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        #     # + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        #     # * noise
        # )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        pred_outputs = {}
        
        # for key in x:
        #     cur_x = x[key]
        #     B, C = cur_x 


        B, C = x['X'].shape[:2]
        assert t.shape == (B,)
        
        scaled_t =  self._scale_timesteps(t)
        batched_scaled_t = scaled_t.unsqueeze(-1) 
        
        if 'obj_task_setting' in x:
            cond = x['obj_task_setting']
            model_output = model(x['X'], x['E'], batched_scaled_t, cond=cond)
        elif 'X_cond' in x:
            node_masks = th.ones((x['X'].shape[0], x['X'].shape[1]), device=x['X'].device)
            model_output = model(x['X'], x['E'], batched_scaled_t, X_cond=x['X_cond'], feat_cond=x['E_cond'], node_mask=node_masks)
        else:
            node_masks = th.ones((x['X'].shape[0], x['X'].shape[1]), device=x['X'].device)
            ### get the output from the model ###
            ## need to be a point cloud model ###
            # model_output = model(x, self._scale_timesteps(t), **model_kwargs)
            model_output = model(x['X'], x['E'], batched_scaled_t, node_mask=node_masks)
        
        
        model_output = {
            'X': model_output.X,
            'E': model_output.E,
            # 'y': model_output.y
        }

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            
            model_variance = {}
            model_log_variance = {}
            
            for key in model_output:
                cur_model_output = model_output[key]
                assert cur_model_output == (B, C * 2, *x[key].shape[2:])
                cur_model_output, cur_model_var_values = th.split(cur_model_output, C, dim=1)
                if self.model_var_type == ModelVarType.LEARNED:
                    cur_model_log_variance = cur_model_var_values
                    cur_model_variance = th.exp(cur_model_log_variance)
                else:
                    min_log = _extract_into_tensor(
                        self.posterior_log_variance_clipped, t, x[key].shape
                    )
                    max_log = _extract_into_tensor(np.log(self.betas), t, x[key].shape)
                    frac = (cur_model_var_values + 1) / 2
                    cur_model_log_variance = frac * max_log + (1 - frac) * min_log
                    cur_model_variance = th.exp(cur_model_log_variance)
                model_variance[key] = cur_model_variance
                model_log_variance[key] = cur_model_log_variance
                
        else:
            model_variance = {}
            model_log_variance = {}
            
            
            tmp_model_variance, tmp_model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            
            for key in x:
                cur_model_variance = _extract_into_tensor(tmp_model_variance, t, x[key].shape)
                cur_model_log_variance = _extract_into_tensor(tmp_model_log_variance, t, x[key].shape)
                
                model_variance[key] = cur_model_variance
                model_log_variance[key] = cur_model_log_variance
                
        
        ### TODO: implement the following code ###
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
        
            # for ke
            pred_xstart = {}
            
            for key in model_output:
                pred_xstart[key] = process_xstart(
                    self._predict_xstart_from_xprev(x_t=x[key], t=t, xprev=model_output[key])
                )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            pred_xstart = {}
            model_mean = {}
            for key in model_output:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart[key] = process_xstart(model_output[key])
                else:
                    pred_xstart[key] = process_xstart(
                        self._predict_xstart_from_eps(x_t=x[key], t=t, eps=model_output[key])
                    )
            
                model_mean[key], _, _ = self.q_posterior_mean_variance(
                    x_start=pred_xstart[key], x_t=x[key], t=t
                )
        else:
            raise NotImplementedError(self.model_mean_type)

        for key in model_mean:
            assert (
                model_mean[key].shape == model_variance[key].shape == model_log_variance[key].shape == x[key].shape
            )
        # assert (
        #     model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        # )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }


    def p_mean_variance_AE_Diff(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # pred_outputs = {}
        
        # for key in x:
        #     cur_x = x[key]
        #     B, C = cur_x 


        B, C = x['pts_feat'].shape[:2]
        assert t.shape == (B,)
        
        scaled_t =  self._scale_timesteps(t)
        batched_scaled_t = scaled_t.unsqueeze(-1) 
        
        if 'obj_task_setting' in x:
            cond = x['obj_task_setting']
            model_output = model(x['X'], x['E'], batched_scaled_t, cond=cond)
        elif 'X_cond' in x:
            # print(f"in the conditional sampling setting, x: {x.keys()}")
            # node_masks = th.ones((x['X'].shape[0], x['X'].shape[1]), device=x['X'].device)
            # model_output = model(x['X'], x['E'], batched_scaled_t, X_cond=x['X_cond'], feat_cond=x['E_cond'], node_mask=node_masks)
            # print(f"in the conditional sampling setting")
            cond = {
                'X': x['X_cond'],
                'E': x['E_cond']
            }
            if 'history_E_cond' in x:
                cond['history_E'] = x['history_E_cond'] # ge the history E cond #
            if 'history_E_window_idx' in x:
                cond['history_E_window_idx'] = x['history_E_window_idx']
            model_output = model(x['pts_feat'], x['feat_feat'], batched_scaled_t, cond=cond)
        
        else:
            # get the cond feats and the uncond feats #
            # node_masks = th.ones((x['X'].shape[0], x['X'].shape[1]), device=x['X'].device)
            ### get the output from the model ###
            ## need to be a point cloud model ###
            # model_output = model(x, self._scale_timesteps(t), **model_kwargs)
            # model_output = model(x['X'], x['E'], batched_scaled_t, node_mask=node_masks)
            
            # if 'X_cond'
            model_output = model(x['pts_feat'], x['feat_feat'], batched_scaled_t)
        
        
        model_output = {
            'pts_feat': model_output.X,
            'feat_feat': model_output.E,
            # 'y': model_output.y
        }

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            
            model_variance = {}
            model_log_variance = {}
            
            for key in model_output:
                cur_model_output = model_output[key]
                assert cur_model_output == (B, C * 2, *x[key].shape[2:])
                cur_model_output, cur_model_var_values = th.split(cur_model_output, C, dim=1)
                if self.model_var_type == ModelVarType.LEARNED:
                    cur_model_log_variance = cur_model_var_values
                    cur_model_variance = th.exp(cur_model_log_variance)
                else:
                    min_log = _extract_into_tensor(
                        self.posterior_log_variance_clipped, t, x[key].shape
                    )
                    max_log = _extract_into_tensor(np.log(self.betas), t, x[key].shape)
                    frac = (cur_model_var_values + 1) / 2
                    cur_model_log_variance = frac * max_log + (1 - frac) * min_log
                    cur_model_variance = th.exp(cur_model_log_variance)
                model_variance[key] = cur_model_variance
                model_log_variance[key] = cur_model_log_variance
                
        else:
            model_variance = {}
            model_log_variance = {}
            
            
            tmp_model_variance, tmp_model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            
            for key in model_output:
                if t.shape[0] == x[key].shape[1]:
                    cur_model_variance = _extract_into_tensor(tmp_model_variance, t, x[key].transpose(0, 1).shape).transpose(0, 1)
                    cur_model_log_variance = _extract_into_tensor(tmp_model_log_variance, t, x[key].transpose(0, 1).shape).transpose(0, 1)
                else:
                    cur_model_variance = _extract_into_tensor(tmp_model_variance, t, x[key].shape)
                    cur_model_log_variance = _extract_into_tensor(tmp_model_log_variance, t, x[key].shape)
                    
                model_variance[key] = cur_model_variance
                model_log_variance[key] = cur_model_log_variance
                
        
        ### TODO: implement the following code ###
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
        
            # for ke
            pred_xstart = {}
            
            for key in model_output:
                pred_xstart[key] = process_xstart(
                    self._predict_xstart_from_xprev(x_t=x[key], t=t, xprev=model_output[key])
                )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            pred_xstart = {}
            model_mean = {}
            for key in model_output:
                if self.model_mean_type == ModelMeanType.START_X:
                    pred_xstart[key] = process_xstart(model_output[key])
                else:
                    if t.shape[0] == x[key].shape[1]:
                        pred_xstart[key] = process_xstart(
                            self._predict_xstart_from_eps(x_t=x[key].transpose(0, 1), t=t, eps=model_output[key].transpose(0, 1))
                        ).transpose(0, 1)
                    else:
                        pred_xstart[key] = process_xstart(
                            self._predict_xstart_from_eps(x_t=x[key], t=t, eps=model_output[key])
                        )

                if t.shape[0] == x[key].shape[1]:
                    model_mean[key], _, _ = self.q_posterior_mean_variance(
                        x_start=pred_xstart[key].transpose(0, 1), x_t=x[key].transpose(0, 1), t=t
                    )
                    model_mean[key] = model_mean[key].transpose(0, 1)
                else:
                    model_mean[key], _, _ = self.q_posterior_mean_variance(
                        x_start=pred_xstart[key], x_t=x[key], t=t
                    )
        else:
            raise NotImplementedError(self.model_mean_type)

        for key in model_mean:
            assert (
                model_mean[key].shape == model_variance[key].shape == model_log_variance[key].shape == x[key].shape
            )
        # assert (
        #     model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        # )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }


    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t # .float()

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        # print(f"p_sample: x: {x.keys()}")
        # pts_feat in x --- 
        if 'pts_feat' in x:
            out = self.p_mean_variance_AE_Diff(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
        else:
            out = self.p_mean_variance(
                model,
                x,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
        sample =  {}
        for key in out['mean']:
        
            noise = th.randn_like(x[key])
            # nonzero_mask = (
            #     (t != 0).float().view(-1, *([1] * (len(x[key].shape) - 1)))
            # )  # no noise when t == 0
            if t.shape[0] == x[key].shape[1]:
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x[key].transpose(0,1).shape) - 1)))
                )  # no noise when t == 0
                nonzero_mask = nonzero_mask.transpose(0, 1)
            else:
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x[key].shape) - 1)))
                )  # no noise when t == 0
                        
            if cond_fn is not None:
                
                out["mean"] = self.condition_mean(
                    cond_fn, out, x, t, model_kwargs=model_kwargs
                )
            sample[key] = out["mean"][key] + nonzero_mask * th.exp(0.5 * out["log_variance"][key]) * noise
        for key in x:
            if key not in sample:
                sample[key] = x[key]
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_AE_Diff(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        use_t=None,
        data=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        assert data is not None
        processing_keys = ['X', 'E']
        pts = data['X'] # pts # X
        feat = data['E']
        
        with th.no_grad(): # encoded features #
            encoded_feats = model.encode(pts, feat)
        # 
        shape = {}
        for key in encoded_feats:
            shape[key] = encoded_feats[key].shape
        for key in data:
            if key not in shape and key not in processing_keys:
                try:
                    shape[key] = data[key].shape # get the data keys # # get the data keys ##
                except:
                    pass
            
        
        # conditional #
        for sample in self.p_sample_loop_progressive_AE_Diff(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            use_t=use_t,
            data=data
        ):
            final = sample
        
        # 
        sample = final['sample']
        with th.no_grad():
            denoised_pts_feat = sample['pts_feat']
            denoised_feat_feat = sample['feat_feat']
            # 
            denoised_feat = {
                'pts_feat': denoised_pts_feat,
                'feat_feat': denoised_feat_feat
            }
            tot_decoded_feats = model.decode(denoised_feat) 
            sample = {
                'X': tot_decoded_feats['X'],
                'E': tot_decoded_feats['feat']
            }
            # sample = tot_decoded_feats # get he decoded feats  #
            for key in data:
                if key not in sample:
                    sample[key] = data[key]
            
        return sample


    def p_sample_loop_AE(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        use_t=None,
        data=None,
        ret_encoded_feat=False
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        assert data is not None
        pts = data['X']
        feat = data['E']
        
        # with th.no_grad():
        
        # if ret_encoded_feat:
        encoded_feats = model.encode(pts, feat)
            
        decoded_feats = model.decode(encoded_feats)
        
        
        decoded_pts = decoded_feats['X']
        decoded_feat = decoded_feats['feat']
        
        sample = {
            'X': decoded_pts,
            'E': decoded_feat
        }
        if ret_encoded_feat:
            sample.update(encoded_feats)
        # # AE # # 
        # shape = {}
        # for key in encoded_feats:
        #     shape[key] = encoded_feats[key].shape
        
        
        # for sample in self.p_sample_loop_progressive_AE_Diff(
        #     model,
        #     shape,
        #     noise=noise,
        #     clip_denoised=clip_denoised,
        #     denoised_fn=denoised_fn,
        #     cond_fn=cond_fn,
        #     model_kwargs=model_kwargs,
        #     device=device,
        #     progress=progress,
        #     use_t=use_t,
        #     data=data
        # ):
        #     final = sample
        
        # # 
        # sample = final['sample']
        # with th.no_grad():
        #     denoised_pts_feat = sample.X
        #     denoised_feat_feat = sample.E
        #     # 
        #     denoised_feat = {
        #         'pts_feat': denoised_pts_feat,
        #         'feat_feat': denoised_feat_feat
        #     }
        #     tot_decoded_feats = model.decode(denoised_feat) 
        #     sample = {
        #         'X': tot_decoded_feats.X,
        #         'E': tot_decoded_feats.E
        #     }
            
        return sample



    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        use_t=None,
        data=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            use_t=use_t,
            data=data
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None, # 
        progress=False,
        use_t=None,
        data=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list, dict))
        
        to_noise_keys = ['X', 'E']
        
        if noise is not None:
            img = noise
        else:
            img = {}
            for key in shape:
                if key in to_noise_keys:
                    img[key] = th.randn(*shape[key], device=device)
                else:
                    if data is not None and key in data:
                        img[key] = data[key]
                    
        
        if use_t is not None:
            indices = list(range(use_t))[::-1]
            
            for key in shape:
                if key in to_noise_keys:
                    print(f"Start sampling with t: {use_t}")
                    
                    img[key] = self.q_sample(data[key], t=th.tensor([use_t] * shape[key][0], device=device), noise=img[key])
                else:
                    img[key]  = data[key]
                
        else:
            indices = list(range(self.num_timesteps))[::-1] ## indicies ##

        if progress: # progress sampling # # progress # #
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[list(shape.keys())[0]][0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]


    def p_sample_loop_progressive_AE_Diff(
        self,
        model,
        shape,
        noise=None, # 
        clip_denoised=True, # 
        denoised_fn=None, # 
        cond_fn=None, # 
        model_kwargs=None, # 
        device=None, # 
        progress=False, # progress # #
        use_t=None,
        data=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list, dict))
        
        # to_noise_keys = ['X', 'E']
        # the
        to_noise_keys = ['pts_feat', 'feat_feat']
        
        img = {}
        
        for key in shape:
            if key in to_noise_keys:
                img[key] = th.randn(*shape[key], device=device)
            else:
                if data is not None and key in data:
                    img[key] = data[key]
        
        
        
        indices = list(range(self.num_timesteps))[::-1] ## indicies ##

        if progress: # progress sampling # # progress # #
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[list(shape.keys())[0]][0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]



    def p_sample_loop_pcdguided(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        use_t=None,
        data=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive_pcdguided(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            use_t=use_t,
            data=data
        ):
            final = sample
        return final["sample"]
    
    def p_sample_loop_progressive_pcdguided(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        use_t=None,
        data=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list, dict))
        
        
        if noise is not None:
            img = noise
        else:
            img = {}
            for key in shape:
                img[key] = th.randn(*shape[key], device=device)
        
        # if use_t is not None:
        #     indices = list(range(use_t))[::-1]
            
        #     for key in shape:
        #         print(f"Start sampling with t: {use_t}")
        #         img[key] = self.q_sample(data[key], t=th.tensor([use_t] * shape[key][0], device=device), noise=img[key])
            
        # else:
        indices = list(range(self.num_timesteps))[::-1] ## indicies ##
        
        guided_pcd = data['X']
        guided_noised_pcd = []
        noise_pcd = th.randn_like(guided_pcd)
        for t in range(self.num_timesteps):
            guided_noised_pcd.append(self.q_sample(guided_pcd, t=th.tensor([t] * shape['X'][0], device=device), noise=noise_pcd))
            

        if progress: # progress sampling # # progress # #
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[list(shape.keys())[0]][0], device=device)
            cur_t_noised_pcd = guided_noised_pcd[i]
            img['X'] = cur_t_noised_pcd
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]


    def p_sample_loop_wpcd(
        self,
        model,
        shape,
        x_start=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        use_t=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive_wpcd(
            model,
            shape,
            x_start=x_start,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            use_t=use_t,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]



    def p_sample_loop_progressive_wpcd(
        self,
        model,
        shape,
        x_start=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        use_t=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list, dict))
        
        assert x_start is not None ## 
        
        clean_pcd = x_start['X']
        ## get corrupted samples 
        # corruped_pcds = {}
        print(f"clean_pcd: {clean_pcd.device}, device: {device}")
        ts_to_corrupted_pcds = {}
        noise_sampled = th.randn_like(clean_pcd)
        for t in range(self.num_timesteps):
            # key_t = self.q_sample(clean_pcd, t, noise=noise_key)
            t_indices = th.tensor([t] * shape['X'][0], device=clean_pcd.device)
            corrupted_pcd = self.q_sample(clean_pcd, t=t_indices, noise=noise_sampled)
            ts_to_corrupted_pcds[t] = corrupted_pcd ## get the corrupted pcd ##
            
            # pass
        ## ts to corrupted pcds ## 
        
            
        
        if noise is not None:
            img = noise
        else:
            img = {}
            for key in shape:
                img[key] = th.randn(*shape[key], device=device)
        # indices = list(range(self.num_timesteps))[::-1]
        
        # use_t
        indices = list(range(use_t))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            
            # from the img with noise to a sample #
            # t i the inidices # 
            # cur_ts_corrup ##
            
            t = th.tensor([i] * shape[list(shape.keys())[0]][0], device=clean_pcd.device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                
                #
                cur_ts_corruped_pcd = ts_to_corrupted_pcds[i] #
                # replace pcd in samples with this corrupted pcd #
                cur_sample = out['sample'] ## get 
                cur_sample['X'] = cur_ts_corruped_pcd # ['X']
                out['sample'] = cur_sample ## add the cufrrent sample ## 
                
                yield out
                img = out["sample"]



    def p_sample_loop_wpcd_wacts(
        self,
        model,
        shape,
        use_t=None,
        x_start=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        ## final #
        for sample in self.p_sample_loop_progressive_wpcd_wacts(
            model,
            shape,
            use_t=use_t,
            x_start=x_start,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]



    def p_sample_loop_progressive_wpcd_wacts(
        self,
        model,
        shape,
        use_t=None,
        x_start=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list, dict))
        
        assert x_start is not None ## 
        
        clean_pcd = x_start['X'] ##
        ## get corrupted samples #
        # corruped_pcds = {} # 
        print(f"clean_pcd: {clean_pcd.device}, device: {device}")
        ts_to_corrupted_pcds = {}
        noise_sampled = th.randn_like(clean_pcd)
        
        ## ## diffuse and get the corrupted pcds ## ##
        if use_t is None:
            use_t = self.num_timesteps ## get num-timesteps ##
        
        ## t in range(use_t) ##
        for t in range(use_t):
            # key_t = self.q_sample(clean_pcd, t, noise=noise_key)
            t_indices = th.tensor([t] * shape['X'][0], device=clean_pcd.device)
            corrupted_pcd = self.q_sample(clean_pcd, t=t_indices, noise=noise_sampled)
            ts_to_corrupted_pcds[t] = corrupted_pcd ## get the corrupted pcd ## ##ts to corrupted ptcds ##
            
            ## ts to ##
        ## ts to corrupted pcds ## ## ts to corrupted pcds ## to be corrupted pcds ##
        input_acts = x_start['E'] ## input acts ##
        ## get the corrupted pcds ##
        noise_sampled_acts = th.randn_like(input_acts)
        t_acts_indices = th.tensor([use_t - 1] * shape['X'][0], device=clean_pcd.device)
        corrupted_acts = self.q_sample(input_acts, t=t_acts_indices, noise=noise_sampled_acts) ## 
        
        
        if noise is not None:
            img = noise
        else:
            img = {}
            for key in shape:
                img[key] = th.randn(*shape[key], device=device)
                
        img['E'] = corrupted_acts
        indices = list(range(use_t))[::-1]

        ## progress ##
        ## 

        if progress:
            # lazy import so that we don't depend on tqdm # Lazy # ## lazy --- lazy ## we don't on tqdm #
            # Lazy import so that we don't depend on tqdm # Lazy import so that we don't on tqdm #
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        # for i
        for i in indices:
            
            # from the img with noise to a sample #
            # t i the inidices # for t in indices #
            # cur_ts_corrup # for t in indices #
            
            t = th.tensor([i] * shape[list(shape.keys())[0]][0], device=clean_pcd.device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                
                # 
                cur_ts_corruped_pcd = ts_to_corrupted_pcds[i] #
                # replace pcd in samples with this corrupted pcd #
                cur_sample = out['sample'] ## get #
                cur_sample['X'] = cur_ts_corruped_pcd # ['X'] #
                out['sample'] = cur_sample ## add the cufrrent sample ## 
                
                ## from scaled samples to the out samples ##
                
                yield out
                img = out["sample"]


    def p_sample_loop_transfer_pcds(
        self,
        model,
        shape,
        pcd_target=None,
        use_t=None,
        x_start=None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive_transfer_pcds(
            model,
            shape,
            pcd_target=pcd_target,
            use_t=use_t,
            x_start=x_start,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive_transfer_pcds( ## smaple loop with pcd 
        self,
        model,
        shape,
        pcd_target=None,
        use_t=None,
        x_start=None, # batched trajectories ##
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list, dict))
        
        assert x_start is not None
        assert pcd_target is not None
        
        clean_pcd = x_start['X']
        clean_acts = x_start['E']
        ## get corrupted samples 
        # corruped_pcds = {}
        ts_to_corrupted_pcds = {}
        ts_to_corrupted_acts = {}
        
        noise_sampled = th.randn_like(clean_pcd)
        noise_acts_sampled = th.randn_like(clean_acts)
        for t in range(self.num_timesteps):
            # key_t = self.q_sample(clean_pcd, t, noise=noise_key)
            t_indices = th.tensor([t] * shape[0], device=clean_pcd.device)
            corrupted_pcd = self.q_sample(clean_pcd, t=t_indices, noise=noise_sampled)
            ts_to_corrupted_pcds[t] = corrupted_pcd ## get the corrupted pcd ##
            
            corrupted_acts = self.q_sample(clean_acts, t=t_indices, noise=noise_acts_sampled)
            ts_to_corrupted_acts[t] = corrupted_acts
        ## ts to corrupted pcds ##
        ## ts to corrupted pcds ##
        ## ts to corrupted pcds ## ##
        ## path following --- start from the noise; sample 
        ## 1) start from the noise; 2) start from the corrupted actions; 3) sample for actions at each tiem #
        # bathced_corrupted_target = []
        ts_to_batched_corrupted_target = {}
        
        # batched corrupted target # 
        for t in range(self.num_timesteps):
            # cur_t_corrupted_target = []
            # for i_bsz in range(noise_sampled.shape[0]):
            #     cur_noise = noise_sampled[i_bsz] ## 
            #     cur_ts_cur_noise_corrupted_pcd = self.q_sample(pcd_target, t=t, noise=cur_noise)
            #     cur_t_corrupted_target.append(cur_ts_cur_noise_corrupted_pcd)
            # cur_t_corrupted_target = th.stack(cur_t_corrupted_target, dim=0)
            # ts_to_batched_corrupted_target[t] = cur_t_corrupted_target ## get the corrupted target
            
            t_indices = th.tensor([t] * shape[0], device=clean_pcd.device)
            # print(f"")
            # print(f"pcd_target: {pcd_target.size()}, noise_sampled: {noise_sampled.size()}, clean_pcd: {clean_pcd.size()}")
            corrupted_target_pcd = self.q_sample(pcd_target, t=t_indices, noise=noise_sampled)
            ts_to_batched_corrupted_target[t] = corrupted_target_pcd 
        
        
        # if noise is not None:
        #     img = noise
        # else:
        #     img = {}
        #     for key in shape:
        #         img[key] = th.randn(*shape[key], device=device)
        
        ###### ====== Construct `img` for the following sampling ====== ######
        img = {} 
        ## replace img['X'] with the original noise ##
        # img['X'] = noise_sampled ##         
        
        use_t = self.num_timesteps if use_t is None else use_t
        
        img['X'] = ts_to_batched_corrupted_target[use_t]
        # ts_to_corrupted_pcds
        # img['X'] = ts_to_corrupted_pcds[use_t]
        img['E'] = ts_to_corrupted_acts[use_t]
        
        indices = list(range(use_t))[::-1]
        
        # get indices #
        # indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            
            # from the img with noise to a sample #
            # t i the inidices # 
            # cur_ts_corrup
            
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                
                #
                # cur_ts_corruped_pcd = ts_to_corrupted_pcds[i] #
                # # replace pcd in samples with this corrupted pcd #
                # cur_sample = out['sample'] ## get 
                # cur_sample['X'] = cur_ts_corruped_pcd # ['X']
                # out['sample'] = cur_sample ## add the cufrrent sample ## 
                
                cur_ts_corrupted_pcd = ts_to_batched_corrupted_target[i] ## getthe corrupted datasets ##
                # cur_ts_corrupted_pcd = ts_to_corrupted_pcds[i] #
                cur_sample = out['sample']
                cur_sample['X'] = cur_ts_corrupted_pcd # ts corrupted pcd #
                out['sample'] = cur_sample ## add the cufrrent sample ##  # get sample 

                
                yield out
                img = out["sample"]


    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs. ## kls ####
                 - 'pred_xstart': the x_0 predictions. ## kls ##
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL, # at the first timestep return the decoder NLL #
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t)) # p(x_{t-1} | x_t) # # # # #
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}


    ## # train the AE only ###  ctl traj ae # 
    def training_losses_CtlTraj_AE(self, model, x_start, t, model_kwargs=None, noise=None, calculate_loss_keys=None):
        """ # traiing the losses # 
        Compute training losses for a single timestep. # 

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
            
        pts = x_start['X']
        feat = x_start['E']
        
        encoded_feats = model.encode(pts, feat)
        decoded_feats = model.decode(encoded_feats)
        
        # 
        decoded_pts = decoded_feats['X']
        decoded_feat = decoded_feats['feat']
        
        mse_pts = th.mean(
            (decoded_pts - pts) ** 2
        )
        mse_feats = th.mean(
            (decoded_feat - feat) ** 2
        )
        mse_loss = mse_pts + mse_feats
        
        # mse x and mse feat #
        # mse_x = th.sum(
        #     (pts - decoded_pts) ** 2
        # )
        
        # # # bsz x nn_pts x 3 #  # training losses AE Diff #
        # dist_pts_decoded_pts = th.sum(
        #     (pts.unsqueeze(2) - decoded_pts.unsqueeze(1)) ** 2, dim=-1
        # ) # bsz x nn_pts x nn_pts 
        # minn_dist, minn_dist_idx = th.min(dist_pts_decoded_pts, dim=-1)
        # minn_dist_2, minn_dist_idx_2 = th.min(dist_pts_decoded_pts, dim=1)
        # cd_pts = th.mean(minn_dist) + th.mean(minn_dist_2)
        
        # mse_feats = th.mean(
        #     (decoded_feat - feat) ** 2
        # )
        
        # mse and the loss #
        terms = {
            'mse': mse_loss,  # cd_pts + mse_feats,
            'loss': mse_loss #  cd_pts + mse_feats
        }
        
        return terms


    ## train AE with diffusion ##
    def training_losses_CtlTraj_AE_Diff(self, model, x_start, t, model_kwargs=None, noise=None, calculate_loss_keys=None):
        pts = x_start['X']
        feat = x_start['E']
        n_bsz = feat.size(0)
        # print('x_start', x_start.keys())
        
        with th.no_grad():
            encoded_feats = model.encode(pts, feat)
        # encoded feats #
        # decoded_feats = model.decode(encoded_feats)
        # 
        # encoded_feats = 
        noise_t = {}
        noise = {}
        for key in encoded_feats: # encoded feats #
            key_noise = th.randn_like(encoded_feats[key])
            noise[key] = key_noise
            if t.shape[0] == encoded_feats[key].shape[1]:
                noise_t[key] = self.q_sample(encoded_feats[key].transpose(0, 1), t, key_noise.transpose(0, 1)).transpose(0, 1)
            else:
                noise_t[key] = self.q_sample(encoded_feats[key], t, key_noise)
        
        # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        scaled_t =  self._scale_timesteps(t)
        # print(f"scaled_t: {scaled_t.size()}")
        # batched_scaled_t = th.full((x_t['X'].shape[0], 1), scaled_t, device=x_t['X'].device)
        batched_scaled_t = scaled_t.unsqueeze(-1) # bathced scaled t --- for the sampling ##
        
        noised_pts_feat = noise_t['pts_feat']
        noised_feat_feat = noise_t['feat_feat']
        if 'X_cond' in x_start:
            cond = {
                'X': x_start['X_cond'],
                'E': x_start['E_cond']
            }
            if 'history_E_cond' in x_start:
                cond['history_E'] =x_start['history_E_cond']
            if 'history_E_window_idx' in x_start:
                cond['history_E_window_idx'] = x_start['history_E_window_idx']
        else:
            cond = x_start
        denoised_feat = model(noised_pts_feat, noised_feat_feat, batched_scaled_t, cond=cond)
        
        denoised_feat = {
            'pts_feat': denoised_feat.X, # denoised feat
            'feat_feat': denoised_feat.E
        }
        
        terms = {}
        terms['mse'] = 0.0
        
        for key in denoised_feat:
            cur_x_start = encoded_feats[key]
            cur_x_t = noise_t[key]
            cur_noise = noise[key]
            # previous_x, start_x, epsilon #
            target = { ## add the noise prediction losses ##
                # ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                #     x_start=cur_x_start, x_t=cur_x_t, t=t
                # )[0]  , # 
                ModelMeanType.START_X: cur_x_start, # cur_x_start
                ModelMeanType.EPSILON: cur_noise, # cur_noise
            }[self.model_mean_type] 
            # print(f"{key} {denoised_feat[key].shape} {target.shape} {encoded_feats[key].shape}")
            # print(f"{model_output[key].shape} {target.shape} {x_start[key].shape}")
            assert denoised_feat[key].shape == target.shape == encoded_feats[key].shape
            
            if key in ['feat_feat'] and target.size(0) != n_bsz:
                target = target.contiguous().transpose(1, 0).contiguous()
                denoised_feat[key] = denoised_feat[key].contiguous().transpose(1, 0).contiguous()
            # print(f"target: {target.size()}")
            # print(f"denoised_feat[key]: {denoised_feat[key].size()}")
            terms["mse"] += mean_flat((target - denoised_feat[key]) ** 2)
        terms['loss'] = terms['mse']
        return terms

    
    # trainng losses in the AE mode ? #
    ## training losses ### training losses ##
    def training_losses_AE(self, model, x_start, t, model_kwargs=None, noise=None, calculate_loss_keys=None):
        """ # traiing the losses # 
        Compute training losses for a single timestep. # 

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
            
        pts = x_start['X']
        feat = x_start['E']
        
        encoded_feats = model.encode(pts, feat)
        decoded_feats = model.decode(encoded_feats)
        
        # 
        decoded_pts = decoded_feats['X']
        decoded_feat = decoded_feats['feat']
        
        # mse x and mse feat #
        # mse_x = th.sum(
        #     (pts - decoded_pts) ** 2
        # )
        
        # # bsz x nn_pts x 3 #  # training losses AE Diff #
        dist_pts_decoded_pts = th.sum(
            (pts.unsqueeze(2) - decoded_pts.unsqueeze(1)) ** 2, dim=-1
        ) # bsz x nn_pts x nn_pts 
        minn_dist, minn_dist_idx = th.min(dist_pts_decoded_pts, dim=-1)
        minn_dist_2, minn_dist_idx_2 = th.min(dist_pts_decoded_pts, dim=1)
        cd_pts = th.mean(minn_dist) + th.mean(minn_dist_2)
        
        
        mse_feats = th.mean(
            (decoded_feat - feat) ** 2
        )
        
        mse_feats = th.mean(
            th.sum((decoded_feat - feat) ** 2, dim=-1)
        )
        
        if calculate_loss_keys is not None and 'X' not in calculate_loss_keys:
            cd_pts = 0.0
        
        terms = {
            'mse': cd_pts + mse_feats,
            'loss': cd_pts + mse_feats
        }
        
        return terms



    ## train AE with diffusion ##
    def training_losses_AE_Diff(self, model, x_start, t, model_kwargs=None, noise=None, calculate_loss_keys=None):
        pts = x_start['X']
        feat = x_start['E']
        
        # print('x_start', x_start.keys())
        
        with th.no_grad():
            encoded_feats = model.encode(pts, feat)
        # 
        # decoded_feats = model.decode(encoded_feats)
        # 
        # encoded_feats = 
        noise_t = {}
        noise = {}
        for key in encoded_feats:
            key_noise = th.randn_like(encoded_feats[key])
            noise[key] = key_noise
            if t.shape[0] == encoded_feats[key].shape[1]:
                noise_t[key] = self.q_sample(encoded_feats[key].transpose(0, 1), t, key_noise.transpose(0, 1)).transpose(0, 1)
            else:
                noise_t[key] = self.q_sample(encoded_feats[key], t, key_noise)
        
        # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
        scaled_t =  self._scale_timesteps(t)
        # print(f"scaled_t: {scaled_t.size()}")
        # batched_scaled_t = th.full((x_t['X'].shape[0], 1), scaled_t, device=x_t['X'].device)
        batched_scaled_t = scaled_t.unsqueeze(-1)
        
        
        noised_pts_feat = noise_t['pts_feat']
        noised_feat_feat = noise_t['feat_feat']
        if 'X_cond' in x_start:
            cond = {
                'X': x_start['X_cond'], 
                'E': x_start['E_cond']
            }
            if 'history_E_cond' in x_start:
                cond['history_E'] = x_start['history_E_cond']
            if 'history_E_window_idx' in x_start:
                cond['history_E_window_idx'] = x_start['history_E_window_idx']
        else:
            cond = x_start
        denoised_feat = model(noised_pts_feat, noised_feat_feat, batched_scaled_t, cond=cond)
        
        denoised_feat = {
            'pts_feat': denoised_feat.X,
            'feat_feat': denoised_feat.E
        }
        
        terms = {}
        terms['mse'] = 0.0
        
        for key in denoised_feat:
            cur_x_start = encoded_feats[key]
            cur_x_t = noise_t[key]
            cur_noise = noise[key]
            # previous_x, start_x, epsilon #
            target = { ## add the noise prediction losses ##
                # ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                #     x_start=cur_x_start, x_t=cur_x_t, t=t
                # )[0]  , # 
                ModelMeanType.START_X: cur_x_start, # cur_x_start
                ModelMeanType.EPSILON: cur_noise, # cur_noise
            }[self.model_mean_type] 
            # print(f"{key} {denoised_feat[key].shape} {target.shape} {encoded_feats[key].shape}")
            # print(f"{model_output[key].shape} {target.shape} {x_start[key].shape}")
            assert denoised_feat[key].shape == target.shape == encoded_feats[key].shape
            if key in ['feat_feat']:
                target = target.contiguous().transpose(1, 0).contiguous()
                denoised_feat[key] = denoised_feat[key].contiguous().transpose(1, 0).contiguous()
            # print(f"target: {target.size()}")
            # print(f"denoised_feat[key]: {denoised_feat[key].size()}")
            terms["mse"] += mean_flat((target - denoised_feat[key]) ** 2)
        terms['loss'] = terms['mse']
        return terms

    
    ## training losses ### training losses ##
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, calculate_loss_keys=None):
        """ # traiing the losses # 
        Compute training losses for a single timestep. # 

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        ## x_start: {'X': bsz x nn_nodes x nn_node_features, 'E': bsz x nn_nodes x nn_nodes x nn_edge_features }
        x_t = {}
        noise = {}
        to_noise_keys = ['X', 'E']
        
        for key in x_start: # X and E in the to_noise_keys #
            if key in to_noise_keys: # X and E in the to_noise_keys #
                key_start = x_start[key]
                noise_key = th.randn_like(key_start)
                key_t = self.q_sample(key_start, t, noise=noise_key)
            else:
                key_t = x_start[key] ## key_t ##
                noise_key = None
            x_t[key] = key_t
            noise[key] = noise_key
        
        # if noise is None: # x_start #
        #     noise = th.randn_like(x_start) # x_start #
        # x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            scaled_t =  self._scale_timesteps(t)
            # print(f"scaled_t: {scaled_t.size()}")
            # batched_scaled_t = th.full((x_t['X'].shape[0], 1), scaled_t, device=x_t['X'].device)
            batched_scaled_t = scaled_t.unsqueeze(-1) # .float()
            
            if 'obj_task_setting' in x_start:
                # print(f"model: {model}")
                model_output = model(x_t['X'], x_t['E'], batched_scaled_t, cond=x_t['obj_task_setting'])
            else:
                node_masks = th.ones((x_t['X'].shape[0], x_t['X'].shape[1]), device=x_t['X'].device)
                
                # print(f"model: {model}")
                model_output = model(x_t['X'], x_t['E'], batched_scaled_t, node_mask=node_masks)
            
            model_output = {
                'X': model_output.X,
                'E': model_output.E,
                # 'y': model_output.y
            }

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                terms_vb = {}
                for key in x_t:
                    cur_x_t = x_t[key]
                    B, C = x_t[key].shape[:2]
                    assert model_output[key].shape == (B, C * 2, *cur_x_t.shape[2:])
                    model_output[key], model_var_values = th.split(model_output[key], C, dim=1)
                    # Learn the variance using the variational bound, but don't let
                    # it affect our mean prediction.
                    frozen_out = th.cat([model_output[key].detach(), model_var_values], dim=1)
                    # terms["vb"] = 
                    terms_vb[key] = self._vb_terms_bpd(
                        model=lambda *args, r=frozen_out: r,
                        x_start=x_start[key],
                        x_t=x_t[key],
                        t=t,
                        clip_denoised=False,
                    )["output"]
                    if self.loss_type == LossType.RESCALED_MSE:
                        terms_vb[key] *= self.num_timesteps / 1000.0
                terms["vb"] = sum(terms_vb.values())
                    
            terms["mse"] = 0.0
            
            tot_calculate_loss_keys = x_start.keys()
            
            if calculate_loss_keys is not None:
                tot_calculate_loss_keys = calculate_loss_keys
            
            for key in tot_calculate_loss_keys:
                if key not in to_noise_keys or noise[key] is None:
                    continue
            # for key in ['X']:

                cur_x_start = x_start[key]
                cur_x_t = x_t[key]
                cur_noise = noise[key]
                # previous_x, start_x, epsilon #
                target = { ## add the noise prediction losses ##
                    ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                        x_start=cur_x_start, x_t=cur_x_t, t=t
                    )[0], # 
                    ModelMeanType.START_X: cur_x_start, # cur_x_start
                    ModelMeanType.EPSILON: cur_noise, # cur_noise
                }[self.model_mean_type] 
                
                # print(f"{model_output[key].shape} {target.shape} {x_start[key].shape}")
                assert model_output[key].shape == target.shape == x_start[key].shape
                terms["mse"] += mean_flat((target - model_output[key]) ** 2)
            
                
                # terms[key] = self._vb_terms_bpd(
                #     model=model,
                #     x_start=cur_x_start,
                #     x_t=cur_x_t,
                #     t=t,
                #     clip_denoised=False,
                #     model_kwargs=model_kwargs,
                # )
            
            # target = {
            #     ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
            #         x_start=x_start, x_t=x_t, t=t
            #     )[0],
            #     ModelMeanType.START_X: x_start,
            #     ModelMeanType.EPSILON: noise,
            # }[self.model_mean_type]
            
            # training losses #
            
            # assert model_output.shape == target.shape == x_start.shape
            # terms["mse"] = mean_flat((target - model_output) ** 2)
            
            
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    ## training losses ## training losses ## traning losses ## training losses ## 
    ## record each step of successful trails ## --- for each optimization ##
    ## specified tasks #
    ## training losses ### training losses ##
    def training_losses_traj_translations(self, model, x_start, t, model_kwargs=None, noise=None):
        """ # traiing the losses # 
        Compute training losses for a single timestep. # 

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        ## x_start: {'X': bsz x nn_nodes x nn_node_features, 'E': bsz x nn_nodes x nn_nodes x nn_edge_features }
        x_t = {}
        noise = {}
        to_noise_keys = ['X', 'E']
        
        ###  
        
        for key in x_start: # X and E in the to_noise_keys #
            if key in to_noise_keys: # X and E in the to_noise_keys #
                key_start = x_start[key]
                noise_key = th.randn_like(key_start)
                key_t = self.q_sample(key_start, t, noise=noise_key)
            
                # x_t is the regular x_t # 
                # but noise should be changed # 
                
                ###### ===== target transformation formulation ===== ######
                target_key = key + "_target"
                x_start_target = x_start[target_key] 
                # x_start_target #
                
                ## jnoised jtarget keys ##
                noised_x_part_x, noised_x_part_noise_coef = self.q_sample_xs(x_start_target, t, noise_key)
                # noised_x_part_x + coef * target_noise = key_t
                noise_key = (key_t - noised_x_part_x) / noised_x_part_noise_coef
                ###### ===== target transformation formulation ===== ######

            else:
                key_t = x_start[key] ## key_t ##
                noise_key = None
            x_t[key] = key_t

            noise[key] = noise_key
        
        # if noise is None: # x_start #
        #     noise = th.randn_like(x_start) # x_start #
        # x_t = self.q_sample(x_start, t, noise=noise)
        ### 
        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            scaled_t =  self._scale_timesteps(t)
            # print(f"scaled_t: {scaled_t.size()}")
            # batched_scaled_t = th.full((x_t['X'].shape[0], 1), scaled_t, device=x_t['X'].device)
            batched_scaled_t = scaled_t.unsqueeze(-1) # .float()
            
            if 'obj_task_setting' in x_start:
                # print(f"model: {model}")
                model_output = model(x_t['X'], x_t['E'], batched_scaled_t, cond=x_t['obj_task_setting'])
            else:
                node_masks = th.ones((x_t['X'].shape[0], x_t['X'].shape[1]), device=x_t['X'].device)
                
                # print(f"model: {model}")
                model_output = model(x_t['X'], x_t['E'], batched_scaled_t, node_mask=node_masks)
            
            model_output = {
                'X': model_output.X,
                'E': model_output.E, ## get the model predicted x and es ## 
                ### the model is fully tuned 3##
                # 'y': model_output.y
            }

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                terms_vb = {}
                for key in x_t:
                    cur_x_t = x_t[key]
                    B, C = x_t[key].shape[:2]
                    assert model_output[key].shape == (B, C * 2, *cur_x_t.shape[2:])
                    model_output[key], model_var_values = th.split(model_output[key], C, dim=1)
                    # Learn the variance using the variational bound, but don't let
                    # it affect our mean prediction.
                    frozen_out = th.cat([model_output[key].detach(), model_var_values], dim=1)
                    # terms["vb"] = 
                    terms_vb[key] = self._vb_terms_bpd(
                        model=lambda *args, r=frozen_out: r,
                        x_start=x_start[key],
                        x_t=x_t[key],
                        t=t,
                        clip_denoised=False,
                    )["output"]
                    if self.loss_type == LossType.RESCALED_MSE:
                        terms_vb[key] *= self.num_timesteps / 1000.0
                terms["vb"] = sum(terms_vb.values())
                    
            terms["mse"] = 0.0
            for key in x_start:
                if key not in to_noise_keys or noise[key] is None:
                    continue
            # for key in ['X']:

                cur_x_start = x_start[key]
                cur_x_t = x_t[key]
                cur_noise = noise[key]
                # previous_x, start_x, epsilon #
                target = { ## add the noise prediction losses ##
                    ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                        x_start=cur_x_start, x_t=cur_x_t, t=t
                    )[0], # 
                    ModelMeanType.START_X: cur_x_start, # cur_x_start
                    ModelMeanType.EPSILON: cur_noise, # cur_noise
                }[self.model_mean_type] 
                
                assert model_output[key].shape == target.shape == x_start[key].shape
                terms["mse"] += mean_flat((target - model_output[key]) ** 2)
            
                
                # terms[key] = self._vb_terms_bpd(
                #     model=model,
                #     x_start=cur_x_start,
                #     x_t=cur_x_t,
                #     t=t,
                #     clip_denoised=False,
                #     model_kwargs=model_kwargs,
                # )
            
            # target = {
            #     ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
            #         x_start=x_start, x_t=x_t, t=t
            #     )[0],
            #     ModelMeanType.START_X: x_start,
            #     ModelMeanType.EPSILON: noise,
            # }[self.model_mean_type]
            
            # training losses #
            
            # assert model_output.shape == target.shape == x_start.shape
            # terms["mse"] = mean_flat((target - model_output) ** 2)
            
            
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms


    def training_losses_traj_translations_cond(self, model, x_start, t, model_kwargs=None, noise=None):
        """ # traiing the losses # 
        Compute training losses for a single timestep. # 

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        ## x_start: {'X': bsz x nn_nodes x nn_node_features, 'E': bsz x nn_nodes x nn_nodes x nn_edge_features }
        x_t = {}
        noise = {}
        to_noise_keys = ['X', 'E']
        
        ###  
        
        for key in x_start: # X and E in the to_noise_keys #
            if key in to_noise_keys: # X and E in the to_noise_keys #
                key_start = x_start[key]
                noise_key = th.randn_like(key_start)
                key_t = self.q_sample(key_start, t, noise=noise_key)
            
                # x_t is the regular x_t # 
                # but noise should be changed # 
                
                
                # ###### ===== target transformation formulation ===== ######
                # target_key = key + "_target"
                # x_start_target = x_start[target_key] 
                # # x_start_target #
                
                # ## jnoised jtarget keys ##
                # noised_x_part_x, noised_x_part_noise_coef = self.q_sample_xs(x_start_target, t, noise_key)
                # # noised_x_part_x + coef * target_noise = key_t
                # noise_key = (key_t - noised_x_part_x) / noised_x_part_noise_coef
                # ###### ===== target transformation formulation ===== ######

            else:
                key_t = x_start[key] ## key_t ##
                noise_key = None
            x_t[key] = key_t

            noise[key] = noise_key
        
        # if noise is None: # x_start #
        #     noise = th.randn_like(x_start) # x_start #
        # x_t = self.q_sample(x_start, t, noise=noise)
        ### 
        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            scaled_t =  self._scale_timesteps(t)
            # print(f"scaled_t: {scaled_t.size()}")
            # batched_scaled_t = th.full((x_t['X'].shape[0], 1), scaled_t, device=x_t['X'].device)
            batched_scaled_t = scaled_t.unsqueeze(-1) # .float()
            
            if 'obj_task_setting' in x_start:
                # print(f"model: {model}")
                model_output = model(x_t['X'], x_t['E'], batched_scaled_t, X_cond=x_t['X_cond'], feat_cond=x_t['E_cond'],  cond=x_t['obj_task_setting'])
            else:
                node_masks = th.ones((x_t['X'].shape[0], x_t['X'].shape[1]), device=x_t['X'].device)
                
                # print(f"model: {model}")
                model_output = model(x_t['X'], x_t['E'], batched_scaled_t, X_cond=x_t['X_cond'], feat_cond=x_t['E_cond'],  node_mask=node_masks)
            
            model_output = {
                'X': model_output.X,
                'E': model_output.E, ## get the model predicted x and es ## 
                ### the model is fully tuned 3##
                # 'y': model_output.y
            }

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                terms_vb = {}
                for key in x_t:
                    cur_x_t = x_t[key]
                    B, C = x_t[key].shape[:2]
                    assert model_output[key].shape == (B, C * 2, *cur_x_t.shape[2:])
                    model_output[key], model_var_values = th.split(model_output[key], C, dim=1)
                    # Learn the variance using the variational bound, but don't let
                    # it affect our mean prediction.
                    frozen_out = th.cat([model_output[key].detach(), model_var_values], dim=1)
                    # terms["vb"] = 
                    terms_vb[key] = self._vb_terms_bpd(
                        model=lambda *args, r=frozen_out: r,
                        x_start=x_start[key],
                        x_t=x_t[key],
                        t=t,
                        clip_denoised=False,
                    )["output"]
                    if self.loss_type == LossType.RESCALED_MSE:
                        terms_vb[key] *= self.num_timesteps / 1000.0
                terms["vb"] = sum(terms_vb.values())
                    
            terms["mse"] = 0.0
            for key in x_start:
                if key not in to_noise_keys or noise[key] is None:
                    continue
            # for key in ['X']:

                cur_x_start = x_start[key]
                cur_x_t = x_t[key]
                cur_noise = noise[key]
                # previous_x, start_x, epsilon #
                target = { ## add the noise prediction losses ##
                    ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                        x_start=cur_x_start, x_t=cur_x_t, t=t
                    )[0], # 
                    ModelMeanType.START_X: cur_x_start, # cur_x_start
                    ModelMeanType.EPSILON: cur_noise, # cur_noise
                }[self.model_mean_type] 
                
                assert model_output[key].shape == target.shape == x_start[key].shape
                terms["mse"] += mean_flat((target - model_output[key]) ** 2)
            
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    # 

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
