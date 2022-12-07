import inspect
import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from model.core.base_network import BaseNetwork


class Palette(BaseNetwork):
    def __init__(self, beta_schedule,
                 network_struct='sr3',
                 **kwargs):
        super(Palette, self).__init__(init_type=kwargs['init_type'], gain=0.02)
        self.loss_fn = None
        self.kwargs = kwargs

        if network_struct == 'sr3':
            from .sr3_modules.unet import UNet
        elif network_struct == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet

        class_args = inspect.getfullargspec(UNet.__init__).args[1:]
        args1 = {}
        for arg in class_args:
            if arg in kwargs.keys():
                args1[arg] = kwargs[arg]

        self.model_unet = UNet(**args1)

        self.beta_schedule = beta_schedule

        # 模型装载是原有代码，为改成lighting
        if 'load_dict' in kwargs:
            weights = torch.load(kwargs['load_dict'])
            self.model_unet.load_state_dict(weights, strict=False)

            print('模型load结束')
        else:
            self.init_weights()

        print('初始化结束')

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, ):
        to_torch = partial(torch.tensor, dtype=torch.float32)

        betas = make_beta_schedule(self.beta_schedule, self.kwargs['n_timestep'], linear_start=self.kwargs['linear_start'], linear_end=self.kwargs['linear_end'], cosine_s=8e-3)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
                extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
                extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
                extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1))
        y_0_hat = self.predict_start_from_noise(
            y_t, t=t, noise=self.model_unet(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    # 添加噪声
    def q_sample(self, y_ground_truth, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_ground_truth))
        return (
                sample_gammas.sqrt() * y_ground_truth +
                (1 - sample_gammas).sqrt() * noise
        )

    # 去掉噪声，在p_mean_variance方法中。
    # 输入y_t：上一次输出结果结果，没有添加noise，第一次y_t和y_cond相同。 y_cond原始输入，就是x_input
    # ？？？return添加噪声
    #   在p_mean_variance中调用网络降噪。noise=denoise_fn(cat(x_input, y_t))，输入的作用对网络进行引导，估计出噪声。
    #   再细节一点，就要结合论文，理解noise系数。
    def p_sample(self, y_t, t, clip_denoised=True, x_input=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=x_input)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        # out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return model_mean + noise * (0.5 * model_log_variance).exp()

    def restoration(self, x_input, y_t=None, y_ground_truth=None, mask=None, sample_num=8):
        b, *_ = x_input.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)

        y_t = default(y_t, lambda: torch.randn_like(x_input))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=x_input.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, x_input=x_input)
            if mask is not None:
                y_t = y_ground_truth * (1. - mask) + mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)

        return y_t, ret_arr

    def forward(self, y_ground_truth, x_input=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_ground_truth.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=x_input.device).long()
        gamma_t1 = extract(self.gammas, t - 1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2 - gamma_t1) * torch.rand((b, 1), device=x_input.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_ground_truth))
        y_noisy = self.q_sample(
            y_ground_truth=y_ground_truth, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.model_unet(torch.cat([x_input, y_noisy * mask + (1. - mask) * y_ground_truth], dim=1), sample_gammas)
            loss = self.loss_fn(mask * noise, mask * noise_hat)
        else:
            noise_hat = self.model_unet(torch.cat([x_input, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape=(1, 1, 1, 1)):
    b, *_ = t.shape
    # dim=-1 表示最后一个维度
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas
