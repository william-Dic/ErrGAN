import os

import torch
import torchvision.transforms as transforms
import numpy as np
import random
import DecisionNCE

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from BearRobot.utils.evaluation.base_eval import BaseEval
from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.utils.logger.base_log import BaseLogger
from BearRobot.utils.dataset.dataloader import openimage
from data.libero.data_process import demo2frames
from data.libero.data_process import get_libero_frame
from V_net.V_network import V_model #这个环境里pip install -e .是原版bearobot库，所以V_network直接写相对路径，不然找不到
from tqdm import tqdm

import cv2
import matplotlib
import matplotlib.pyplot as plt

import json
import multiprocessing
from functools import partial
import imageio
from collections import deque

torch.cuda.empty_cache()

EPS = 1e-5
benchmark_dict = benchmark.get_benchmark_dict()
frame_length_dict = demo2frames.frame_counts_dict()
cosine_similarity = torch.nn.functional.cosine_similarity

def has_normalize(transform):
       if isinstance(transform, transforms.Compose):
              for t in transform.transforms:
                     if isinstance(t, transforms.Normalize):
                            return True
       return False


class LIBEROEval_err(BaseEval):
       def __init__(
              self,
              task_suite_name: str, # can choose libero_spatial, libero_spatial, libero_object, libero_100, all, etc.
              obs_key: list=['agentview_image', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 'robot0_eef_pos', 'robot0_eef_quat'],
              data_statistics: dict=None,
              logger: BaseLogger=None,
              eval_horizon: int=600,
              num_episodes: int=10,
              eval_freq: int=10,
              seed: int=42,
              rank: int=0,
              checkpoint_path: str = None,
              k: float = 0.2,
       ):
              super(BaseEval, self).__init__()   
              self.task_suite_name = task_suite_name
              self.task_suite = benchmark_dict[task_suite_name]()
              self.obs_key = obs_key
              self.data_statistics = data_statistics
              self.state_dim = 9
              self.eval_horizon = eval_horizon
              self.num_episodes = num_episodes
              self.eval_freq = eval_freq
              self.logger = logger
              self.seed = seed
              self.s_avoid =  [ [] for _ in range(num_episodes)]
              self.rank = rank
              self.v_func = V_model(state_dim=9)
              self.k = k
              self.device = 'cuda' 
              self.last_state = None

       def _make_dir(self, save_path):
              if self.rank == 0:
                     task_suite_name = self.task_suite_name
                     path = os.path.join(save_path, task_suite_name)
                     if not os.path.exists(path):
                            os.makedirs(path)
                     self.base_dir = path
       
       def _init_env(self, task_id: int=0):
              # get task information and env args
              task = self.task_suite.get_task(task_id)
              task_name = task.name
              task_description = task.language
              task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
              print(f"[info] retrieving task {task_id} from suite {self.task_suite_name}, the " + \
                     f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

              # step over the environment
              env_args = {
                     "bddl_file_name": task_bddl_file,
                     "camera_heights": 128,
                     "camera_widths": 128
              }
              
              # init thesubprocess vector environment
              env_num = self.num_episodes
              env = SubprocVectorEnv(
                     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
              )
              
              # environment reset 
              self.seed += self.num_episodes
              env.seed(self.seed)
              env.reset()
              init_states = self.task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
              init_state_id = np.arange(self.num_episodes) % init_states.shape[0]
              obs = env.set_init_state(init_states[init_state_id])
              
              ### sample one begin and end image frame to construct image goal
              # Generating paths for img_begin and img_end
              task_name = task_description.replace(" ", "_") + "_demo"
              demo_paths = demo2frames.get_demos_for_task(task_name, frame_length_dict)
              demo_path = random.choice(demo_paths)

              # eval with the beginning frame and the endding frame
              env_dict = {}
              transform = transforms.ToTensor()
              base_dir='/home/dodo/ljx/BearRobot/data/libero/dataset/'
              env_dict['img_begin'] = transform(openimage(os.path.join(base_dir, "libero/data_jpg/", demo_path, "image0/0.jpg")))
              end_idx = frame_length_dict[demo_path] - 1 
              env_dict['img_end'] = transform(openimage(os.path.join(base_dir, "libero/data_jpg/", demo_path, f"image0/{end_idx}.jpg")))

              # return the environment
              env_dict['env'] = env
              env_dict['language_instruction'] = task_description
              env_dict['obs'] = obs
              
              return env_dict
       
       def _log_results(self, metrics: dict, steps: int):
              if self.logger is None:
                     # just print out and save the results and pass
                     print(metrics)
                     save_name = os.path.join(self.base_dir, 'results.json')
                     with open(save_name, 'a+') as f:
                            line = json.dumps(metrics)
                            f.write(line+'\n')
              else:
                     # log the results to the logger
                     self.logger.log_metrics(metrics, steps)
                     self.logger.save_metrics(metrics, steps, self.base_dir)
       
       def raw_obs_to_stacked_obs(self, obs, lang):
              env_num = len(obs)
              
              data = {
                     "obs": {},
                     "lang": lang,
              }
              
              for key in self.obs_key:
                     data["obs"][key] = []
                     
              for i in range(env_num):
                     for key in self.obs_key:
                            data['obs'][key].append(obs[i][key])
              
              for key in data['obs']:
                     data['obs'][key] = np.stack(data['obs'][key])
              
              return data     
       
       def np_image_to_tensor(self, image: np.array, normalize_img: bool):
              B, H, W, C = image.shape
              image = image / 255. if image.max() > 10 else image
              image = torch.from_numpy(image).permute(0, 3, 1, 2).to(torch.float32)  # B, C, H, W
              
              if normalize_img:
                     norm_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).repeat(B, 1, 1, 1)  # B, C, 1, 1
                     norm_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).repeat(B, 1, 1, 1)
                     
                     image = (image - norm_mean) / norm_std
              return image  # B, C, H, W
                     
       def b_visual(self, b_list, db_list_T, lang):
              b_list = np.array(b_list)
              db_list_T = np.array(db_list_T)

              fig, axs = plt.subplots(2, 1, figsize=(8, 10))

              for num in range(len(b_list[0])):
                     result = b_list[:, num]
                     axs[0].plot(result, label=f'b{num}')
              
              axs[0].set_title('Trends of b values across Batches')
              axs[0].set_xlabel('Index')
              axs[0].set_ylabel('Value')
              axs[0].legend()
              axs[0].grid(True)

              for num in range(len(db_list_T[0])):
                     result = db_list_T[:, num]
                     axs[1].plot(result, label=f'db{num}')
              
              axs[1].set_title('Trends of db values across Batches')
              axs[1].set_xlabel('Index')
              axs[1].set_ylabel('Value')
              axs[1].legend()
              axs[1].grid(True)

              plt.tight_layout()
              plt.savefig(f'b_trends/{lang}.png')
              plt.close()

       

       def sliding_window_average(self,data, window_size):
              if len(data) < window_size:
                     return sum(data) / len(data)
              return sum(data[-window_size:]) / window_size

       def _rollout(self, policy: BaseAgent, task_id: int=0, img_goal=False):
              """
              rollout one episode and return the episode return
              """
              env = self._init_env(task_id)
              self._env = env
              lang = env['language_instruction']
              obs = env['obs']
              img_begin = env['img_begin']
              img_end = env['img_end']
              k = self.k
              policy.policy._init_action_chunking(self.eval_horizon, self.num_episodes)
              
              images = []
              b_list = []
              seq_len = 10 #TODO change this to 5 when using diffenrent ckpt 
              backtrack_len = 30
              t_buffer = torch.zeros(self.num_episodes, seq_len).to(self.device) 
              
              current_times = torch.zeros(self.num_episodes).to(self.device)
              traj_list = [ [] for _ in range(self.num_episodes)]
              action_traj_list = [ [] for _ in range(self.num_episodes)]
              action_index = [ 0 for _ in range(self.num_episodes)]
              
              t_diff = [ [] for _ in range(self.num_episodes)]
              self.recover_flag = [False for _ in range(self.num_episodes)]
              self.backtrack_flag = [False for _ in range(self.num_episodes)]
              
              b_buffer = [[0 for _ in range(seq_len)] for _ in range(self.num_episodes)]
              
              for t in tqdm(range(self.eval_horizon), desc=f'{lang}'):
                     # time ++
                     current_times += 1
                     
                     ## image
                     data = self.raw_obs_to_stacked_obs(obs, lang)
                     obs, lang = data['obs'], data['lang']
                     
                     normalize_img = has_normalize(policy.transform)
                     agent_view = self.np_image_to_tensor(obs['agentview_image'], normalize_img).unsqueeze(1)
                     wrist_view = self.np_image_to_tensor(obs['robot0_eye_in_hand_image'], normalize_img).unsqueeze(1)
                     image_input = torch.cat([agent_view, wrist_view], dim=1).unsqueeze(1).to(self.device)  
                     
                     ### record the video
                     B, H, W, C = obs["agentview_image"].shape
                     images.append(np.flip(np.flip(obs["agentview_image"], 1), 2).reshape(B * H, W, C))
                     img_recovery = np.flip(np.flip(obs["agentview_image"], 1), 2).reshape(B * H, W, C)
                     
                     ## proprio
                     gripper_qpos = obs['robot0_gripper_qpos']
                     eef_pos = obs['robot0_eef_pos']
                     eef_quat = obs['robot0_eef_quat']
                     
                     # add images and proprio(state) to buffer
                     try:
                            s_dim = policy.s_dim
                     except:
                            s_dim = policy.policy.module.s_dim
                     state = np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=-1) if s_dim > 0 else None
                     
                     # process state
                     state = torch.from_numpy(state.astype(np.float32)).view(-1, policy.s_dim) if state is not None else None
                     state = ((state - policy.s_mean) / policy.s_std).to('cuda') if state is not None else None
                     state_clone = state.detach().clone()
              
                     # get action using img_begin and img_end embedding difference
                     if img_goal:
                            # using the current frame as the img_begin   # TODO delete this if not useful
                            img_begin = torch.from_numpy(obs['agentview_image']).permute(0, 3, 1, 2) / 255
                            
                            action = policy.get_action(image_input, None, state=state, current_time=current_times, t=t, k=0.25, img_begin=img_begin, img_end = img_end, img_goal=img_goal,s_avoid = self.s_avoid)
                     else:
                            action = policy.get_action(image_input, lang, state=state, current_time=current_times, t=t, k=0.25, s_avoid = self.s_avoid) 
                     # reshape
                     action = action.reshape(self.num_episodes, -1)
                     
                     # step
                     # if any(self.backtrack_flag):
                     #        for i in range(self.num_episodes):
                     #               print(f"action_index:{action_index[i]}",f"action_traj_len:{len(action_traj_list[i])}\n")
                     #               if self.backtrack_flag[i]:
                     #                      if action_index[i] < len(action_traj_list[i]) - backtrack_len:
                     #                             action[i] = action_traj_list[i][action_index[i]]
                     #                             action_index[i] += 1
                     #                             # print(f"Episode_{i} is Backtracking\n")
                     #                      else:
                     #                             action_index[i] = 0
                     #                             action_traj_list[i].clear()
                     #                             self.backtrack_flag[i] = False
                                                 
                     #                             img_recovery[i * H:(i + 1) * H, :W] = [255,255, 0]
                                          
                     #                             for _ in range(30): 
                     #                                    images.append(img_recovery) 
                                                        
                     #                             # print(f"Episode_{i} Backtracking Finished\n")
              
                     obs, reward, done, info = env['env'].step(action)
                            
                     print("done:", done, " reward:", reward - 1)    
                     
                     
                     if done.all():
                            break
                     
                     
                     # ----------------------Trial && Error part--------------------------              
                     # resize images
                     imgs = image_input.squeeze(1) # [B, V, C, H, W]
                     imgs_tuple = torch.chunk(imgs, 2, dim=1)
                     image_1 = imgs_tuple[0].squeeze(1) # [B, C, H, W] 
                     image_2 = imgs_tuple[1].squeeze(1) # [B, C, H, W]
                     imgs = image_input.squeeze(1) # [B, V, C, H, W]
                     
                     for i in range(seq_len):
                            t_buffer[:, i] = torch.clamp(current_times.squeeze() - (seq_len - i), min=0)
                            
                     # add traj
                     b = policy.v_model(image_1, image_2, state,lang,current_times,t_buffer,torch.tensor(b_buffer)).argmax(dim=-1)
                     
                     b_list.append(b.cpu().detach().numpy())
                            
                     for i in range(self.num_episodes):
                            traj_list[i].append(b[i])
                            b_buffer[i].pop(0) 
                            b_buffer[i].append(b[i].item())
                            
                            if not self.backtrack_flag[i]:
                                   action_traj_list[i].append(action[i])
                            
                            current_t = current_times[i].item() - 1 
                            k_value = 20
                            t_k = int(current_t - k_value)
                                   
                            if done[i] or t_k <= 0 or self.backtrack_flag[i]:
                                   continue
                            
                            window_size = 10
                            t_b = self.sliding_window_average(traj_list[i], window_size)
                            t_k_b = self.sliding_window_average(traj_list[i][:t_k+1], window_size)
                            # t_b = traj_list[i][-1]
                            # t_k_b = traj_list[i][t_k]

                            print(t_b - t_k_b)
                            t_diff[i].append(t_b - t_k_b)
                            if ((t_b - t_k_b) < 0) or (t_diff[i][-30:].count(0) >= 15):
                            # if ((t_b - t_k_b) < 0):
                            
                                   print(f"Episode {i} Recovered\n")
                                   self.recover_flag[i] = True
                                   self.backtrack_flag[i] = True
                                   
                                   current_times[i] = 0 # curren_times reset
                                   
                                   traj_list[i].clear() # traj reset
                                   t_diff[i].clear()
                                   b_buffer[i] = [0 for _ in range(seq_len)]
                                   
                     
                     # visualize recovery
                     if any(self.recover_flag):
                            for i in range(B):
                                   # 如果对应的 recover_flag 为 True，将最上方 H*W 的图片变为纯红色
                                   if self.recover_flag[i]:
                                          img_recovery[i * H:(i + 1) * H, :W] = [255, 0, 0]
                                          
                            for i in range(30): 
                                   images.append(img_recovery) 
              
                            self.recover(action=action, task_id=task_id)
                                          
                            # new seed
                            self.seed += self.num_episodes
                            self._env['env'].seed(self.seed)  
              
                     self.last_state = state_clone
              
              db_list_T = np.gradient(np.array(b_list), axis=0)
              
              self.b_visual(b_list,db_list_T,lang)
              save_path = f'{self.base_dir}/video/{lang}.mp4' 
              self._save_video(save_path, images, done, fps=30)
              num_success = 0
              for k in range(self.num_episodes):
                     num_success += int(done[k])
              avg_succ_rate = num_success / self.num_episodes
             
              metrics = {f'sim/{self.task_suite_name}/{lang}': avg_succ_rate}
              self._log_results(metrics, self.step)
              
              env['env'].close()
              return avg_succ_rate

       def recover(self, action, task_id: int=0):
              for i in range(self.num_episodes):
                     if self.recover_flag[i]:
                            if self.last_state[i] is None:
                                   print(f"!!!Episode index {i} last_state is None, do not wish this state, check!!!!!!!")
                                   raise ValueError
                            
                            # if len(self.s_avoid[i]) == 0:
                            #        self.s_avoid[i].append(self.last_state[i].detach())
                            #        self.s_avoid[i].append(torch.tensor(action[i]))
                            # else:
                            #        self.s_avoid[i][0].append(self.last_state[i].detach())
                            #        self.s_avoid[i][1].append(torch.tensor(action[i]))
                            
                            
                            # print(self.s_avoid[i][0],self.s_avoid[i][1])
                            
                            self.recover_flag[i] = False
                            self._env['env'].reset(i) 
       
                            
       
       def _save_video(self, save_path: str, images: list, done: list, fps=30): 
              imageio.mimsave(save_path, images, fps=fps)
              
       def _save_err_video(self, save_path: str, err_images: list, done: list, fps=30): 
              imageio.mimsave(save_path, err_images, fps=fps)
              
       def file_count(self, path):
              if not os.path.exists(path):
                     os.makedirs(path)
              return len(os.listdir(path))

       def eval_episodes(self, policy: BaseAgent, steps: int, save_path: str, img_goal=False):
              """
              rollout several episodes and log the mean episode return
              """
              self._make_dir(save_path)
              self.step = steps
              
              rews = []
              policy.eval()
              # for _ in tqdm(range(self.num_episodes), desc="Evaluating..."):
              for task_id in tqdm(range(len(self.task_suite.tasks)), desc="Evaluating..."):
                     rews.append(self._rollout(policy, task_id, img_goal))
              eval_rewards = sum(rews) / len(rews)
              metrics = {f'sim/{self.task_suite_name}/all': eval_rewards}
              self._log_results(metrics, self.step)
              return eval_rewards
              
       
       def close_env(self):
              for env in self.env:
                     env['env'].close()
                     

from mmengine import fileio
import io
import os
import json
import yaml

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from BearRobot.Agent import *
from BearRobot.Agent.ACT import ACTAgent 
from BearRobot.Net.my_model.diffusion_model import VisualDiffusion, VisualDiffusion_pretrain
from BearRobot.Net.my_model.ACT_model import ACTModel
from Err_ddpm_bc import IDQL_Agent

def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict


def convert_str(item):
        try:
                return int(item)
        except:
                try:
                        return float(item)
                except:
                        return item


def wandb_args2dict(ckpt_path, wandb_name: str=None):
        if wandb_name is None:
                wandb_name = 'latest-run'
        try:
                wandb_path = os.path.join('/'.join(ckpt_path.split('.pth')[0].split('/')[:-1]), f'wandb/{wandb_name}/files/wandb-metadata.json')
                meta_data = openjson(wandb_path)
                args = [convert_str(arg.split('--')[-1]) for arg in meta_data['args']]
                config_dict = dict(zip(args[::2], args[1::2]))
                print("-------load meta data from wandb succ!------------")
                print_dict = json.dumps(config_dict, indent=4, sort_keys=True)
                print(print_dict)
                return config_dict
        except:
                print("Automatically load wandb meta data fail, please provide your meta data mannually")
                return {}


def wandb_yaml2dict(ckpt_path, wandb_name: str=None, wandb_path: str=None):
        if wandb_name is None:
                wandb_name = 'latest-run'
        try:
                if wandb_path is None:
                        wandb_path = os.path.join('/'.join(ckpt_path.split('.pth')[0].split('/')[:-1]), f'wandb/{wandb_name}/files/config.yaml')
                with open(wandb_path, 'r') as stream:
                        config = yaml.safe_load(stream)
                del config['wandb_version']
                del config['_wandb']
                config_dict = {key: value['value'] for key, value in config.items()}
                print("-------load meta data from wandb succ!------------")
                print_dict = json.dumps(config_dict, indent=4, sort_keys=True)
                print(print_dict)
                return config_dict
        except:
                print("Automatically load wandb meta data fail, please provide your meta data mannually")
                return {}


def load_ckpt(agent, v_model_ckpt_path,policy_ckpt_path):
       from collections import OrderedDict

       ckpt_v = fileio.get(v_model_ckpt_path)
       ckpt_policy = fileio.get(policy_ckpt_path)
       
       # get policy weights
       with io.BytesIO(ckpt_policy) as f:
              ckpt_policy = torch.load(f, map_location='cuda')
       new_ckpt = OrderedDict()
       for key in ckpt_policy['model'].keys():
              new_key = key.replace(".module", '')
              new_ckpt[new_key] = ckpt_policy['model'][key]
       ckpt_policy['model'] = new_ckpt
       agent.policy.load_state_dict(ckpt_policy['model'])
       
       # get v_model weights
       with io.BytesIO(ckpt_v) as f:
              ckpt_v = torch.load(f, map_location='cuda')
       new_ckpt = OrderedDict()
       for key in ckpt_v['v_model'].keys():
              new_key = key.replace(".module", '')
              new_ckpt[new_key] = ckpt_v['v_model'][key]
       ckpt_v['v_model'] = new_ckpt
       agent.v_model.load_state_dict(ckpt_v['v_model'])

       agent.eval()
       agent.policy.eval()
       return agent.to(0)


def build_visual_diffsuion_err(policy_ckpt_path: str,v_model_ckpt_path: str,statistics_path: str, k: float=0.2, num_episodes: int=10 ,wandb_name: str=None, wandb_path: str=None):
       kwargs = wandb_yaml2dict(v_model_ckpt_path, wandb_name, wandb_path=wandb_path)
       model = VisualDiffusion(view_num=2,
                            output_dim=7 * kwargs['ac_num'],
                            **kwargs).to(0)
       agent = IDQL_Agent(policy_model=model, num_episodes=num_episodes, k=k, **kwargs) #TODO check gamma and expectile
       agent.policy.get_statistics(statistics_path)
       agent.policy.get_transform(kwargs['img_size'])
       agent.get_statistics(statistics_path)
       agent.get_transform(kwargs['img_size'])
       return load_ckpt(agent, v_model_ckpt_path,policy_ckpt_path)



