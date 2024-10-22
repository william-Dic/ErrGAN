import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import torch.optim as optim
import random

from PIL import Image

from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.Net.my_model.diffusion_model import VisualDiffusion
from BearRobot.Net.my_model.t5 import T5Encoder
from BearRobot.Net.encoder.DecisionNCE import DecisionNCE_encoder, DecisionNCE_lang, DecisionNCE_visual_diff
from data.libero.data_process import demo2frames

# from V_net.V_network import V_model
# from TransitionNCE import Transition_NCE,Transition_NCE_loss
from errGAN import Discriminator,Generator
from Q_GAN import Generator as Generator_q

frame_length_dic = demo2frames.frame_counts_dict()
def extract(a, x_shape):
       '''
       align the dimention of alphas_cumprod_t to x_shape
       
       a: alphas_cumprod_t, B
       x_shape: B x F x F x F
       output: alphas_cumprod_t B x 1 x 1 x 1]
       '''
       b, *_ = a.shape
       return a.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
       """
       linear schedule, proposed in original ddpm paper
       """
       scale = 1000 / timesteps
       beta_start = scale * 0.0001
       beta_end = scale * 0.02
       return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
       """
       cosine schedule
       as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
       """
       steps = timesteps + 1
       t = torch.linspace(0, timesteps, steps) / timesteps
       alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
       alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
       betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
       return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
       """
       sigmoid schedule
       proposed in https://arxiv.org/abs/2212.11972 - Figure 8
       better for images > 64x64, when used during training
       """
       steps = timesteps + 1
       t = torch.linspace(0, timesteps, steps) / timesteps
       v_start = torch.tensor(start / tau).sigmoid()
       v_end = torch.tensor(end / tau).sigmoid()
       alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
       alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
       betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
       return torch.clip(betas, 0, 0.999)


def vp_beta_schedule(timesteps):
       """Discret VP noise schedule
       """
       t = torch.arange(1, timesteps + 1)
       T = timesteps
       b_max = 10.
       b_min = 0.1
       alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
       betas = 1 - alpha
       return betas       


SCHEDULE = {'linear': linear_beta_schedule,
            'cosine': cosine_beta_schedule,
            'sigmoid': sigmoid_beta_schedule,
            'vp': vp_beta_schedule}
                     
class DDPM_BC(BaseAgent):
       def __init__(
              self, 
              policy: torch.nn.Module, 
              beta: str='cosine',
              T: int=5,
              ac_num: int=1,
       ):
              super().__init__(
                     policy, None, None, ac_num=ac_num
              )
              
              self.device = policy.device
              if beta not in SCHEDULE.keys():
                     raise ValueError(f"Invalid schedule '{beta}'. Expected one of: {list(SCHEDULE.keys())}")
              self.schedule = SCHEDULE[beta]
              
              self.num_timesteps = T
              self.betas = self.schedule(self.num_timesteps).to(self.device)
              self.alphas = (1 - self.betas).to(self.device)
              self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)

       def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None):
              """
              predict the noise
              Input:
              xt: noisy samples, here xt is typically the ground truth action
              t: timestep
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              
              Return: predicted noise
              """
              noise_pred = self.policy(xt, t, cond)
              return noise_pred
       
       def predict_noise(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor=None):
              """
              predict the noise
              
              Input:
              xt: noisy samples, here xt is typically the noisy ground truth action
              t: timestep
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              
              Return: predicted noise
              """
              noise_pred = self.policy(xt, t, cond)
              return noise_pred
       
       def policy_loss(self, x0: torch.Tensor, cond: torch.Tensor=None):
              '''
              calculate ddpm loss
              
              Input:
              x0: ground truth value, here x0 is typically the ground truth action
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              
              Return: ddpm loss
              '''
              batch_size = x0.shape[0]
              
              noise = torch.randn_like(x0, device=self.device)
              t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
              
              xt = self.q_sample(x0, t, noise)
              
              noise_pred = self.predict_noise(xt, t, cond)
              loss = (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
              
              return loss
              
       def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
              """
              sample noisy xt from x0, q(xt|x0), forward process
              
              Input:
              x0: ground truth value, here x0 is typically the ground truth action
              t: timestep
              noise: noise
              
              Return: noisy samples
              """
              alphas_cumprod_t = self.alphas_cumprod[t]
              xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
              return xt
       
       @torch.no_grad()
       def  p_sample(self, xt: torch.Tensor, t: torch.Tensor, noise_pred: torch.Tensor, guidance_strength=0, clip_sample=False, ddpm_temperature=1.):
              """
              sample xt-1 from xt, p(xt-1|xt)
              
              Input:
              xt: noisy samples, here xt is typically the noisy ground truth action
              t: timestep
              noise_pred: predicted noise
              guidance_strength: the strength of the guidance
              clip_sample: whether to clip the sample to [-1, 1]
              ddpm_temperature: the temperature of the noise
              
              Return: sample xt-1
              """
              alpha1 = 1 / torch.sqrt(self.alphas[t])
              alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
              
              xtm1 = alpha1 * (xt - alpha2 * noise_pred)
              
              noise = torch.randn_like(xtm1, device=self.device) * ddpm_temperature
              xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)
              
              if clip_sample:
                     xtm1 = torch.clip(xtm1, -1., 1.)
              return xtm1
       
       @torch.no_grad()
       def ddpm_sampler(self, shape, cond=None, guidance_strength=0, clip_sample=False):
              """
              sample x0 from xT, reverse process
              
              Input:
              shape: the desired shape of the sample
              cond: condition information, here cond is typically the state rather than other condition like language or goal image
              guidance_strength: the strength of the guidance
              clip_sample: whether to clip the sample to [-1, 1]
              
              Return: sampled x0
              """
              x = torch.randn(shape, device=self.device)
              cond = cond.repeat(x.shape[0], 1)
              
              for t in reversed(range(self.num_timesteps)):
                     noise_pred = self.predict_noise(x, t, cond)
                     x = self.p_sample(x, torch.full((shape[0], 1), t, device=self.device, clip_sample=clip_sample), noise_pred)
              return x

       @torch.no_grad()
       def get_action(self, state, num=1, clip_sample=False):
              return self.ddpm_sampler((num, self.policy.output_dim), cond=state, clip_sample=clip_sample)
              

class VLDDPM_BC(DDPM_BC):
       """
       this is a vision-language diffusion agent, based on DDPM
       """
       def __init__(
              self, 
              policy: VisualDiffusion, 
              beta: str='cosine',  
              T: int=5,
              ac_num: int=1,
              text_encoder: str="T5",#TODO change this back 
              device = 'cuda',
              *args, **kwargs
       ):
              super().__init__(policy, beta, T, ac_num)
              
              # self.img_size = self.policy.img_size
              assert text_encoder in ['T5', 'DecisionNCE-T', 'DecisionNCE-P', "DecisionNCE-V"]
              if text_encoder == 'T5':
                     self.lang_encoder = T5Encoder(device=device)
              elif text_encoder in ['DecisionNCE-T', 'DecisionNCE-P']:
                     mm_encoder = DecisionNCE_encoder(text_encoder, device=device)
                     self.lang_encoder = DecisionNCE_lang(mm_encoder)
              elif text_encoder == "DecisionNCE-V":
                     text_encoder = 'DecisionNCE-T'
                     mm_encoder = DecisionNCE_encoder(text_encoder, device=device)
                     self.frame_diff_encoder = DecisionNCE_visual_diff(mm_encoder)
              else:
                     raise ValueError(f"Invalid text_encoder '{text_encoder}'. Expected one of: ['t5', 'DecisionNCE-T', 'DecisionNCE-P']")
              print("lang encoder load success")
              self.device = device
              
       def forward(self, images: torch.Tensor, cond: dict, action_gt: torch.Tensor, state=None, img_goal=False):
              '''
              calculate ddpm loss
              # images: batch of frames of different views, [B, Frame, View, C, H, W]
              # texts: list of instructions
              # state shape [B, D_s], batch of robot arm x,y,z, gripper state, et al
              # action_gt shape [B, D_a], batch of robot control value, e.g., delta_x, delta_y, delta_z,..., et al.
              # img_begin: one begin frame of a video
              # img_end: one end frame of a video
              # img_goal: bool, whether to use img_begin and img_end to calculate visual difference as goal
              '''
              img_begin = cond['img_begin']
              img_end = cond['img_end']
              texts = cond['lang']
       
              if img_goal:
                     if img_begin != None and img_end != None:      
                            text_emb_visiual_diff = self.frame_diff_encoder.embed_frame(img_begin, img_end).to(images.device).detach()
                            loss = self.policy_loss(action_gt, images, text_emb_visiual_diff, state)
                            loss_dict = {"policy_loss": loss}
                            return loss_dict
                     else:
                            raise ValueError("img_begin or img_end is None")
              else:
                     text_emb = self.lang_encoder.embed_text(texts).to(images.device).detach()
                     loss = self.policy_loss(action_gt, images, text_emb, state)
                     loss_dict = {"policy_loss": loss}
                     return loss_dict
       
       
       def policy_loss(self, x0: torch.Tensor, imgs: torch.Tensor, condition: torch.Tensor, state: torch.Tensor=None):
              '''
              calculate ddpm loss
              Input:
              x0: [B, D_a] ground truth action
              imgs: [B, F, V, C, H, W] batch of frames of different views
              condition: [B, D] batch of instruction embbeding
              state: [B, D_s] batch of robot arm x,y,z, gripper state
              
              return: loss
              '''
              batch_size = x0.shape[0]
              
              noise = torch.randn_like(x0, device=self.device)
              t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
              xt = self.q_sample(x0, t, noise)
              
              noise_pred = self.predict_noise(xt, t, imgs, condition, state)
              loss = (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
              
              return loss
       
       def predict_noise(self, xt: torch.Tensor, t: torch.Tensor, imgs: torch.Tensor, condition: torch.Tensor, state: torch.Tensor=None):
              '''
              calculate ddpm loss
              
              Input:
              xt: [B, D_a] noise action, generated by ground truth action
              t: [B, 1] time step
              imgs: [B, F, V, C, H, W] batch of frames of different views
              condition: [B, D] batch of instruction embbeding
              state: [B, D_s] batch of robot arm x,y,z, gripper state
              
              Return: [B, D_a] predicted noise
              '''
              noise_pred = self.policy(xt, t, imgs, condition, state)
              return noise_pred
       
       def get_best_actions(B, actions, s_avoid):
              s_avoid_tensor = torch.stack(s_avoid)

              actions_expanded = actions.unsqueeze(2)  #[B, num, 1, D_a]
              s_avoid_expanded = s_avoid_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, len(s_avoid), D_s]

              distances = torch.norm(actions_expanded - s_avoid_expanded, dim=-1)  # [B, num, len(s_avoid)]

              min_distances, _ = torch.min(distances, dim=-1)  # [B, num]

              best_action_indices = torch.argmax(min_distances, dim=-1)  # [B]

              best_actions = actions[torch.arange(B), best_action_indices]  # [B, D_a]

              return best_actions.view(B, -1, actions.shape[-1])
       @torch.no_grad()
       def ddpm_sampler(self, shape, imgs: torch.Tensor, lang: list, state: torch.Tensor=None, guidance_strength=0, clip_sample=False, img_begin=None, img_end=None, img_goal=False, s_avoid=None):
              """
              sample x0 from xT, reverse process. Note this ddpm_sampler is different from the original ddpm_sampler as the condition type including both img & lang
              
              Input:
              shape: [num_samples, D_a], desired shapes of samples
              imgs: [B, F, V, C, H, W] batch of frames of different views
              lang: list of instructions
              state: [B, D_s] batch of robot arm x,y,z, gripper state. Do not use this when is None
              guidance_strength: strength of guidance
              clip_sample: whether to clip samples to [-1, 1]
              
              Return: [num_samples, D_a] samples
              """
              x = torch.randn(shape, device=self.device)
              if img_goal:
                     if img_begin is not None and img_end is not None:
                            cond = self.frame_diff_encoder.embed_frame(img_begin, img_end)
                     else:
                            raise ValueError("img_begin or img_end is None")
              else:
                     cond = self.lang_encoder.embed_text(lang) if self.lang_encoder is not None else lang        
              
              for t in reversed(range(self.num_timesteps)):
                     t_tensor = torch.tensor([t]).unsqueeze(0).repeat(x.shape[0], 1).to(self.device)
                     noise_pred = self.predict_noise(x, t_tensor, imgs, cond, state)
                     x = self.p_sample(x, torch.full((shape[0], 1), t, device=self.device), noise_pred, clip_sample=clip_sample)
              return x

       @torch.no_grad()
       def get_action(self, imgs, lang, state=None, num=1, t=1, k=0.25, clip_sample=True, img_begin=None, img_end=None, img_goal=False):
              if not isinstance(imgs, torch.Tensor):
                     # transform lists to torch.Tensor
                     imgs = torch.stack([self.transform(Image.fromarray(frame).convert("RGB")) for frame in imgs]).unsqueeze(0).unsqueeze(0).to('cuda')
              else:
                     imgs = imgs.to('cuda')                     
                     
              B, F, V, C, H, W = imgs.shape

              try:
                     output_dim = self.policy.output_dim
              except:
                     output_dim = self.policy.module.output_dim
              
              # use img goal or language goal
              if img_goal:
                     if img_begin is not None and img_end is not None:
                            if len(img_begin.shape) == 3:
                                   img_begin_pools = img_begin.unsqueeze(0).repeat(B, 1, 1, 1)
                            elif len(img_begin.shape) == 4 and img_begin.shape[0] == B:
                                   img_begin_pools = img_begin
                            else:
                                   raise ValueError(f"Please check the shape of img_begin: {img_begin.shape}")
                            img_end_pools = img_end.unsqueeze(0).repeat(B, 1, 1, 1) 
                            if num != 1:
                                   state = state.unsqueeze(1).repeat(1,num,1).reshape(B*num,-1)
                            else:
                                   state = state
                            action = self.ddpm_sampler((B, output_dim), imgs, lang, state, clip_sample=clip_sample, img_begin=img_begin_pools, img_end=img_end_pools, img_goal=True).detach().cpu().reshape()
                            action = action.view(B, -1, 7)
                     else:
                            raise ValueError("img_begin or img_end is None")
              else:
                     if num != 1:
                            state = state.unsqueeze(1).repeat(1,num,1).reshape(B*num,-1)
                     action = self.ddpm_sampler((B, output_dim), imgs, [lang] * B, state, clip_sample=clip_sample).detach().cpu()
                     action = action.view(B, -1, 7)
              
              B, N, D_a = action.shape
              a_max = self.a_max.repeat(B, N, 1)
              a_min = self.a_min.repeat(B, N, 1)
              a_mean = self.a_mean.repeat(B, N, 1)
              a_std = self.a_std.repeat(B, N, 1)
              action = (action + 1) * (a_max - a_min) / 2 + a_min
              
              # resize action
              action = self.get_ac_action(action.numpy(), t, k)
              return action

       
       
class IDQL_Agent(BaseAgent):
       def __init__(
              self, 
              policy_model: torch.nn.Module,
              schedule: str='vp', 
              num_timesteps: int=5,
              num_sample: int=64,
              gamma: float=0.99,
              expectile: float=0.9, 
              k: float=0.2,
              s_dim: int=9,
              device = 'cuda',
              num_episodes = 10,
              *args, **kwargs
       ):       
              super().__init__(
                     policy_model,
                     v_model=None,
                     q_models=None,
                     gamma=gamma
              )
              self.expectile = expectile
              self.num_sample = num_sample
              self.device = self.policy.device  
              self.device = device
              self.s_dim = s_dim
              
              # self.v_model = V_model(state_dim=s_dim).to(self.device)
              # self.v_model = Transition_NCE(state_dim=s_dim).to(self.device)
              self.criterion = nn.BCELoss()
              self.generator = Generator().to(self.device)
              self.generator_q = Generator_q().to(self.device)
              self.discriminator = Discriminator().to(self.device)
              self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002)
              self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002)
              vlddpm_policy = VLDDPM_BC(policy_model, schedule=schedule, num_timesteps=num_timesteps, **kwargs)
              self.policy = vlddpm_policy
              # self.criterion = Transition_NCE_loss()
              
              


       def get_target_distribution(self,b):
              target_distribution = torch.zeros(50)
              if b == 0:
                     target_distribution[b] = 2/3
                     target_distribution[b + 1] = 1/3
              elif b >= 49:
                     target_distribution[-1] = 2/3
                     target_distribution[-2] = 1/3
              else:
                     target_distribution[b - 1] = 1/3
                     target_distribution[b] = 1/3
                     target_distribution[b + 1] = 1/3
              return target_distribution # [2% 4% ... 98% 100%] shape (50,)
       
       def get_one_hot_target_distribution(self, b):
              target_distribution = torch.zeros(100)
              if b < 99:
                     target_distribution[b] = 1
              else:
                     target_distribution[-1] = 1
              return target_distribution
       
       def get_progressive_target_distribution(self,b):

              if b > 99:
                     b = 99

              target_distribution = torch.zeros(100)
              
              exponent_range = torch.arange(b + 1)
              exponential_distribution = torch.exp(exponent_range)
              
              exponential_distribution /= exponential_distribution.sum()

              target_distribution[0:b + 1] = exponential_distribution
              
              return target_distribution
       
              
       def v_loss(self, imgs: torch.Tensor, state: torch.Tensor, lang:torch.Tensor, t: torch.Tensor, T: torch.Tensor):
              # Input:
              # imgs: [B, F, V, C, H, W] batch of frames of different views
              # state: [B, D_s] batch of robot arm x,y,z, gripper state
              # t: [B, 1] timestep
              # T: [B, 1] total timesteps
              #
              # Output:
              # cross_entropy loss
              
              # resize images
              seq_len = 5
              
              imgs = imgs.squeeze(1) # [B, V, C, H, W]
              imgs_tuple = torch.chunk(imgs, 2, dim=1)
              image_1 = imgs_tuple[0].squeeze(1) # [B, C, H, W] 
              image_2 = imgs_tuple[1].squeeze(1) # [B, C, H, W]
              
              # calculate b
              t = torch.Tensor(t).to(self.device)
              T = torch.Tensor(T).to(self.device)
              b = torch.round(t/T/0.01)

              target_distributions = torch.stack([self.get_one_hot_target_distribution(int(i)) for i in b]).to(self.device)
              
              t_buffer = torch.zeros(t.size(0), seq_len).to(self.device)
              for i in range(seq_len):
                     t_buffer[:, i] = torch.clamp(t.squeeze() - (seq_len - i), min=0)

              T_expanded = T.squeeze().unsqueeze(1).expand(-1, seq_len)  
              label_buffer = torch.round(t_buffer / T_expanded / 0.01)

              # calculate v
              v = self.v_model(image_1, image_2, state,lang,t,t_buffer,label_buffer).to(self.device) # [B, 50]
              
              loss = F.cross_entropy(v, target_distributions)
              
              return loss
       
       def v_loss_expectile(self, imgs: torch.Tensor, state: torch.Tensor,next_imgs: torch.Tensor, next_state: torch.Tensor, lang:torch.Tensor, t: torch.Tensor, T: torch.Tensor, reward:torch.Tensor):
              # Input:
              # imgs: [B, F, V, C, H, W] batch of frames of different views
              # state: [B, D_s] batch of robot arm x,y,z, gripper state
              # t: [B, 1] timestep
              # T: [B, 1] total timesteps
              #
              # Output:
              # cross_entropy loss
              
              # resize images
              imgs = imgs.squeeze(1) # [B, V, C, H, W]
              imgs_tuple = torch.chunk(imgs, 2, dim=1)
              image_1 = imgs_tuple[0].squeeze(1) # [B, C, H, W] 
              image_2 = imgs_tuple[1].squeeze(1) # [B, C, H, W]
              
              next_imgs = next_imgs.squeeze(1) # [B, V, C, H, W]
              next_imgs_tuple = torch.chunk(next_imgs, 2, dim=1)
              next_image_1 = next_imgs_tuple[0].squeeze(1) # [B, C, H, W] 
              next_image_2 = next_imgs_tuple[1].squeeze(1) # [B, C, H, W]
              
              # calculate b
              t = torch.Tensor(t).to(self.device)
              t_next = (t + 1).to(self.device)
              T = torch.Tensor(T).to(self.device) #TODO T的逻辑有问题
              # print(t,T)
              b = torch.round(t/T/0.02) #TODO 查bug
              b_next = torch.round(t_next/T/0.02)
              reward = torch.Tensor(reward).to(self.device) #[B,1]
              target_distributions = torch.stack([self.get_one_hot_target_distribution(int(i)) for i in b]).to(self.device)
              target_distributions_next = torch.stack([self.get_one_hot_target_distribution(int(i)) for i in b_next]).to(self.device)
              
              # calculate v
              v = self.v_model(image_1, image_2, state,lang,t).to(self.device) # [B, 50]
              v_next = self.v_model(next_image_1, next_image_2, next_state,lang,t_next).to(self.device) # [B, 50]
              
              v_index = torch.argmax(v, dim=1).to(self.device)  # [B]
              v_next_index = torch.argmax(v_next, dim=1).to(self.device)  # [B]

              # print(b,v_index,v_next_index)
              v_loss = F.mse_loss(v, target_distributions_next)
              # v_loss = F.cross_entropy(v, target_distributions)
              v_loss_next = F.mse_loss(v_next, target_distributions_next)
              # v_loss_next = F.cross_entropy(v_next, target_distributions_next)
              
              diff = reward + self.gamma * v_loss_next - v_loss 
              # print(reward,diff)
              loss = self.expectile_loss(diff)
              
              return loss
       
       def v_loss_transitionNCE(self,proprio_seq_e,lang_e):
              
              loss = self.criterion(proprio_seq_e, lang_e)
              
              return loss
       
       def GAN_loss(self, lang, state_seq, image_seq, action):
              batch_size = len(lang)
              
              real_labels = torch.ones(batch_size, 1).to(self.device)  # [B, 1]
              fake_labels = torch.zeros(batch_size, 1).to(self.device)  # [B, 1]

              self.optimizer_D.zero_grad()
              
              real_logits, real_classes = self.discriminator(image_seq, state_seq, action, lang)
              real_loss = self.criterion(real_classes, real_labels)
              
              fake_actions = self.generator(state_seq, image_seq, lang)
              
              fake_logits, fake_classes = self.discriminator(image_seq, state_seq, fake_actions, lang)
              fake_loss = self.criterion(fake_classes, fake_labels)
              
              d_loss = real_loss + fake_loss
              d_loss.backward(retain_graph=True)
              self.optimizer_D.step()

              self.optimizer_G.zero_grad()
              
              fake_logits, fake_classes = self.discriminator(image_seq, state_seq, fake_actions, lang)
              g_loss = self.criterion(fake_classes, real_labels)
              
              g_loss.backward()
              self.optimizer_G.step()

              return g_loss.item(), d_loss.item()
       def q_loss(self, s, a, r, next_s, d):
              gamma = 1
              with torch.no_grad():
                     next_v = self.v_model(next_s).squeeze()
                     target_q_value = r + gamma * (1 - d) * next_v

              q_values = torch.cat([model(s) for model in self.q_models], dim=1)
              q_value = torch.gather(q_values, 1, a.long().unsqueeze(1)).squeeze(1)

              loss = (q_value - target_q_value).pow(2).mean()
              return loss,q_value
       
       def policy_loss(self, x0, cond=None):
              loss = self.policy.policy_loss(x0, cond)
              return loss
       
       
       def expectile_loss(self, diff):
              weight = torch.where(diff > 0, self.expectile, (1 - self.expectile))
              return torch.mean(weight * (diff**2))
       

       @torch.no_grad()
       def get_action(self, imgs, lang, state=None, num=1, t=1, k=0.25, clip_sample=True, img_begin=None, img_end=None, img_goal=False):
              
              
              # ddpm agent
              if img_goal:
                     action = self.policy.get_action(imgs, None, state=state, t=t, k=0.25, img_begin=img_begin, img_end = img_end, img_goal=img_goal)
              else:
                     action = self.policy.get_action(imgs, lang, state=state, t=t, k=0.25)

              return action
              
              

if __name__ == '__main__':
       timesteps = 100
       schedule = SCHEDULE['linear']
       print(schedule(timesteps)[0])

       
