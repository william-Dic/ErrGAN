import os

import torch
torch.cuda.empty_cache()

import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from BearRobot.utils.evaluation.base_eval import BaseEval
from BearRobot.Agent.base_agent import BaseAgent
from BearRobot.utils.logger.base_log import BaseLogger
from tqdm import tqdm

import matplotlib.pyplot as plt

import json
import imageio


EPS = 1e-5
benchmark_dict = benchmark.get_benchmark_dict()

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
              ac_num:int=6
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
              self.rank = rank
              self.k = k
              self.device = 'cuda' 
              self.error_length = 50 # TODO maybe change this to 20?
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

              # eval with the beginning frame and the endding frame
              env_dict = {}
              
              # return the environment
              env_dict['env'] = env
              env_dict['language_instruction'] = task_description
              env_dict['obs'] = obs
              
              return env_dict
       
       def trial_init_env(self, task_id: int=0):
              # get task information and env args
              task = self.task_suite.get_task(task_id)
              task_name = task.name
              task_description = task.language
              task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

              # step over the environment
              env_args = {
                     "bddl_file_name": task_bddl_file,
                     "camera_heights": 128,
                     "camera_widths": 128
              }
              
              # init thesubprocess vector environment
              env_num = 1
              env = SubprocVectorEnv(
                     [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
              )
              
              # environment reset 
              self.seed += 100
              env.seed(self.seed)
              env.reset()
              init_states = self.task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
              init_state_id = [0]
              obs = env.set_init_state(init_states[init_state_id])
              
              ### sample one begin and end image frame to construct image goal

              # eval with the beginning frame and the endding frame
              env_dict = {}
              
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
                     
       def b_visual(self, b_list, lang):
              b_list = np.array(b_list)

              fig, ax = plt.subplots(figsize=(8, 5))  

              for num in range(len(b_list)):
                     result = b_list[num]
                     ax.plot(result, label=f'b{num}')
              
              ax.legend()
              ax.grid(True)
              
              plt.tight_layout()
              plt.savefig(f'b_trends/{lang}.png')
              plt.close(fig)  

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
              
              k = self.k
              policy.policy._init_action_chunking(self.eval_horizon, self.num_episodes)
              
              images = []
               
              action_index = [ 0 for _ in range(self.num_episodes)]
              traj_list = [ [] for _ in range(self.num_episodes)]
              b_list = [ [] for _ in range(self.num_episodes)]
              action_traj_list = [ [] for _ in range(self.num_episodes) ]
              
              self.backtrack_flag = [False for _ in range(self.num_episodes)]
              self.trial_flag = [False for _ in range(self.num_episodes)]
              state_seq = [ [] for _ in range(self.num_episodes) ]
              image_seq = [ [] for _ in range(self.num_episodes) ]
              k_value = 20
       
              with torch.no_grad():
                     
                     for t in tqdm(range(self.eval_horizon), desc=f'{lang}'):
                            
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

                            # print(f"state:{state},{state.shape}")
                            # print(state[0,:])
                            # backtracking
                            
                            action = policy.get_action(image_input, lang, state=state, t=t, k=0.25) 
                            action = action.reshape(self.num_episodes, -1) #二维的
                            obs,reward,done,_ = self._env['env'].step(action)
                            
                            if any(self.backtrack_flag):
                                   for i in range(self.num_episodes):
                                          end_index = len(action_traj_list[i]) - self.error_length
                                          if self.backtrack_flag[i]:                                                 
                                                 if action_index[i] < end_index:
                                                        action[i] = action_traj_list[i][action_index[i]]
                                                        action_index[i] += 1
                                                        
                                                        print(f"action_index:{action_index[i]}",f"action_traj_len:{len(action_traj_list[i])}\n")
                                                        print(f"Episode_{i} is Backtracking\n")
                                                 else:
                                                        action_index[i] = 0
                                                        self.backtrack_flag[i] = False
                                                        self.trial_flag[i] = True
                                                        img_recovery[i * H:(i + 1) * H, :W] = [255,255, 0]
                                                 
                                                        for _ in range(30): 
                                                               images.append(img_recovery) 
                                                               
                                                        print(f"Episode_{i} Backtracking Finished\n")
                            #TODO 回溯到错误的t-n时刻然后再选action 用generator？
                            #TODO 1.把trial结束后的action合到action traj里面 2.如果trial的尝试长度超过了error_length直接recover
                            
                            # if any(self.trial_flag):
                            #        for i in range(self.num_episodes):
                            #               if self.trial_flag[i]:
                                                 
                            #                      trial_action_list = []
                            #                      trial_logit_list = []
                                                 
                            #                      for _ in range(5):
                            #                             env_trial = self.trial_init_env(task_id)
                            #                             for action_trial_index in range(end_index):
                            #                                    action_trial = action_traj_list[i][action_trial_index]
                            #                                    env_trial["env"].step(action_trial)
        
                            #                             image_seq_trail = image_seq[i]
                            #                             state_seq_trail = state_seq[i]
                                                        
                            #                             action_trial = policy.get_action(image_input, lang, state=state, t=t, k=0.25, s_avoid = self.s_avoid) 
                            #                             action_trial = action_trial.reshape(self.num_episodes, -1)
                                                        
                            #                             obs_next, _, _, _ = env_trial["env"].step(action_trial)

                            #                             ## image
                            #                             data_next = self.raw_obs_to_stacked_obs(obs_next, lang)
                            #                             obs_next, _ = data_next['obs'], data_next['lang']

                            #                             normalize_img = has_normalize(policy.transform)
                            #                             agent_view_next = self.np_image_to_tensor(obs_next['agentview_image'], normalize_img).unsqueeze(1)
                            #                             wrist_view_next = self.np_image_to_tensor(obs_next['robot0_eye_in_hand_image'], normalize_img).unsqueeze(1)
                            #                             image_input_next = torch.cat([agent_view_next, wrist_view_next], dim=1).unsqueeze(1).to(self.device)  
                            #                             imgs_next = image_input_next.squeeze(1) # [B, V, C, H, W]
                            #                             imgs_tuple_next = torch.chunk(imgs_next, 2, dim=1)
                            #                             image_1_next = imgs_tuple_next[0].squeeze(1) # [B, C, H, W] 
                                                        
                            #                             ## proprio
                            #                             gripper_qpos_next = obs_next['robot0_gripper_qpos']
                            #                             eef_pos_next = obs_next['robot0_eef_pos']
                            #                             eef_quat_next = obs_next['robot0_eef_quat']
                                                        
                            #                             # add images and proprio(state) to buffer
                            #                             try:
                            #                                    s_dim = policy.s_dim
                            #                             except:
                            #                                    s_dim = policy.policy.module.s_dim
                            #                             state_next = np.concatenate([gripper_qpos_next, eef_pos_next, eef_quat_next], axis=-1) if s_dim > 0 else None
                                                        
                            #                             # process state
                            #                             state_next = torch.from_numpy(state_next.astype(np.float32)).view(-1, policy.s_dim) if state is not None else None
                            #                             state_next = ((state_next - policy.s_mean) / policy.s_std).to('cuda') if state is not None else None
                                                        
                            #                             state_seq_trail.append(state_next[i,:].cpu()) #[B,9]
                            #                             image_seq_trail.append(image_1_next[i,:,:,:].cpu()) #[B,3,128,128]
                                                        
                            #                             # print(state_seq_trail[0],state_seq_trail[-1])
                            #                             image_seq_tensor_a = torch.stack(image_seq_trail, dim=0).unsqueeze(dim=0).to(self.device)
                            #                             state_seq_tensor_a = torch.stack(state_seq_trail, dim=0).unsqueeze(dim=0).to(self.device)
                            #                             logits, _ = policy.discriminator(image_seq_tensor_a, state_seq_tensor_a, [lang])

                            #                             trial_action_list.append(action_trial)
                            #                             trial_logit_list.append(logits.cpu().detach().numpy())

                            #               best_action_index = np.argmax(trial_logit_list)
                            #               best_action = trial_action_list[best_action_index]
                            #               action[i] = best_action
                            #               env_trial['env'].close()
                            #        print(f"Trial action selected for episode {i}: {best_action}, with logit: {trial_logit_list[best_action_index]}")

                            
                            if any(self.trial_flag):
                                   num = 5
                                   action_list = []
                                   for i in range(self.num_episodes):
                                          if self.trial_flag[i]:

                                                 state_seq_trial_tensor = torch.stack(state_seq[i],dim=0).unsqueeze(dim=0).to(self.device) #[B,seq_len,1,9]
                                                 image_seq_trial_tensor = torch.stack(image_seq[i],dim=0).unsqueeze(dim=0).to(self.device) #[B,seq_len,3,128,128]

                                                 best_action = policy.generator_q(state_seq_trial_tensor, image_seq_trial_tensor, [lang])
                                                 
                                                 for _ in range(num):
                                                        action_list.append(torch.tensor(policy.get_action(image_input, lang, state=state, t=t, k=0.25)[0]))
                                                 action_list = torch.stack(action_list,dim=0)
                                                 
                                                 best_action_expanded = best_action.expand(action_list.size(0), -1)
                                                 # Compute cosine similarities
                                                 cosine_similarities = F.cosine_similarity(
                                                        action_list.to(self.device),  # Select the batch element i [10,7]
                                                        best_action_expanded.to(self.device),  # Expand to match the shape [10, 7]
                                                        dim=1
                                                 )  # Output shape: [10]

                                                 best_action_index_per_trial = torch.argmax(cosine_similarities, dim=0)  # [1]
                                                 best_action_per_trial = action_list[best_action_index_per_trial.item()]  # Select the best action from the 10

                                                 action[i] = best_action_per_trial
                                                 # print(f"Trial action selected for episode {i}: {best_action_per_trial}")

                                   
                            print("done:", done, " reward:", reward - 1)    

                            if done.all():
                                   break
                            
                            # ----------------------Trial && Error part--------------------------              
                            # resize images
                            imgs = image_input.squeeze(1) # [B, V, C, H, W]
                            imgs_tuple = torch.chunk(imgs, 2, dim=1)
                            image_1 = imgs_tuple[0].squeeze(1) # [B, C, H, W] 
                            # image_2 = imgs_tuple[1].squeeze(1) # [B, C, H, W]
                            # imgs = image_input.squeeze(1) # [B, V, C, H, W]
                            
                            # image_seq.append(image_1)
                            for i in range(self.num_episodes):
                                   
                                   state_seq[i].append(state[i,:].cpu()) #[B,9]
                                   image_seq[i].append(image_1[i,:,:,:].cpu()) #[B,3,128,128]
                                   
                                   if not self.backtrack_flag[i]:
                                          action_traj_list[i].append(action[i]) #TODO 改一下这里的logic recover两次之后action_traj_list没东西 看一下recover之后action_traj_list有没有新增加
                                   state_seq_tensor = torch.stack(state_seq[i],dim=0).unsqueeze(dim=0).to(self.device) #[B,seq_len,1,9]
                                   image_seq_tensor = torch.stack(image_seq[i],dim=0).unsqueeze(dim=0).to(self.device) #[B,seq_len,3,128,128]
                                   lang_list = [lang] 
                                   logits, classes = policy.discriminator(image_seq_tensor,state_seq_tensor,lang_list)
                                   print(logits)
                                   traj_list[i].append(logits.squeeze(dim=0).cpu().detach())
                                   b_list[i].append(logits.squeeze(dim=0).cpu().detach())
                                   
                                   if done[i]:
                                          traj_list[i] = []
                                          image_seq[i] = []
                                          state_seq[i] = []
                                          continue
                                   
                                   if len(traj_list[i]) <= k_value or self.backtrack_flag[i]:
                                          continue
                                   
                                   
                                   # t_b = traj_list[i][-1]
                                   # t_k_b = traj_list[i][-k_value]
                                   window_size = 10
                                   t_b = self.sliding_window_average(traj_list[i], window_size)
                                   t_k_b = self.sliding_window_average(traj_list[i][-30:-k_value], window_size)
                                   print(f'Value_diff: {t_b - t_k_b}')
                                   if ((t_b - t_k_b) < 0.1): #TODO change this back to 0 or 0.1
                                   
                                          print(f"Episode {i} Recovered\n")
                                          self.backtrack_flag[i] = True
                                          self.trial_flag[i] = False
                                          
                                          traj_list[i] = []
                                          image_seq[i] = []
                                          state_seq[i] = []

                                          for img_index in range(B):
                                                 img_recovery[img_index * H:(i + 1) * H, :W] = [255, 0, 0]
                                                        
                                          for _ in range(30): 
                                                 images.append(img_recovery) 
                                          
                                          self._env['env'].reset(i) 
                                          # new seed
                                          self.seed += self.num_episodes
                                          self._env['env'].seed(self.seed) 
                                   
              self.b_visual(b_list,lang)
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


def load_ckpt(agent, v_model_ckpt_path,policy_ckpt_path,q_model_ckpt_path=None):
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
       for key in ckpt_v['discriminator_state_dict'].keys():
              new_key = key.replace(".module", '')
              new_ckpt[new_key] = ckpt_v['discriminator_state_dict'][key]
       ckpt_v['discriminator_state_dict'] = new_ckpt
       agent.discriminator.load_state_dict(ckpt_v['discriminator_state_dict'])

       # get q_model weights
       if q_model_ckpt_path != None:
              ckpt_q = fileio.get(q_model_ckpt_path)
              with io.BytesIO(ckpt_q) as f:
                     ckpt_q = torch.load(f, map_location='cuda')
              new_ckpt = OrderedDict()
              for key in ckpt_q['generator_state_dict'].keys():
                     new_key = key.replace(".module", '')
                     new_ckpt[new_key] = ckpt_q['generator_state_dict'][key]
              ckpt_q['generator_state_dict'] = new_ckpt
              agent.generator_q.load_state_dict(ckpt_q['generator_state_dict'])
       
       agent.eval()
       agent.policy.eval()
       return agent.to(0)


def build_visual_diffsuion_err(policy_ckpt_path: str,v_model_ckpt_path: str,statistics_path: str, k: float=0.2, num_episodes: int=10 ,wandb_name: str=None, wandb_path: str=None):
       kwargs = wandb_yaml2dict(v_model_ckpt_path, wandb_name, wandb_path=wandb_path)
       kwargs['ac_num'] = 6
       model = VisualDiffusion(view_num=2,
                            output_dim=7 * kwargs['ac_num'],
                            # output_dim=7 * 6,
                            **kwargs).to(0)
       agent = IDQL_Agent(policy_model=model, num_episodes=num_episodes, k=k, **kwargs) #TODO check gamma and expectile
       agent.policy.get_statistics(statistics_path)
       agent.policy.get_transform(kwargs['img_size'])
       agent.get_statistics(statistics_path)
       agent.get_transform(kwargs['img_size'])
       return load_ckpt(agent,v_model_ckpt_path,policy_ckpt_path)



