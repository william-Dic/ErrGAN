import math
import os
import json
import random

import torch
import torch.nn as nn
import numpy as np

from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms

from BearRobot.utils import ddp
from torch.utils.data import DistributedSampler
from data.libero.data_process import demo2frames
from data.libero.data_process import get_libero_frame
from mmengine import fileio
import io

EPS = 1e-5

frame_length_dic = demo2frames.frame_counts_dict()

def openimage(path):
       value  = fileio.get(path)
       img_bytes = np.frombuffer(value, np.uint8)
       buff = io.BytesIO(img_bytes)
       img = Image.open(buff).convert('RGB')
       return img

def openimage_tensor(value):
       img_bytes = np.frombuffer(value, np.uint8)
       buff = io.BytesIO(img_bytes)
       img = Image.open(buff).convert('RGB')
       return img


def openjson(path):
       value  = fileio.get_text(path)
       dict = json.loads(value)
       return dict
   

class AIRKitchenDataset_err():
       def __init__(
              self,
              base_dir='',
              datalist=['/home/dodo/ljx/BearRobot/data/airkitchen/AIR-toykitchen-ac.json'],
              img_size: int=128,
              frames: int=3,
              view_list: list=['D435_image', 'wrist_image'],
              norm: str=None,
              discretize_actions: bool=False,
              ac_num: int=4,
              statistics: dict=None,
              mask_aug: bool=False,  # True for IVM training, False for normal training
              transform_list: list=None,  # e.g. [transforms.RandomResizedCrop(224, scale=(0.75, 1), interpolation=Image.BICUBIC)], you can feed your own transform
              img_goal: bool=False,
       ):
              self.base_dir = base_dir
              self.img_size = img_size
              self.frames = frames
              self.view_list = view_list
              self.discretize_actions = discretize_actions
              self.norm = norm
              self.ac_num = ac_num
              self.mask_aug = mask_aug
              self.img_goal = img_goal
              
              err_datalist = ["/home/dodo/wgm/CL/BearRobot/BearRobot/Agent/deployment/err_libero_goal-ac.json"]
              self.datalist = []
              self.err_datalist = []
              
              for one_list in datalist:
                     datalist = openjson(one_list)
                     self.datalist += datalist
                     
              for one_list in err_datalist:
                     err_datalist = openjson(one_list)
                     self.err_datalist += err_datalist
                     
              self._statistics(statistics)
              
              if transform_list == None:
                     transform_list  = [
                            transforms.ColorJitter(0.2, [0.8, 1.2], [0.8, 1.2], 0.1),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                     ]  
                     if self.img_size != 0:
                            transform_list.insert(0, transforms.RandomResizedCrop(self.img_size, scale=(0.75, 1), interpolation=Image.BICUBIC))     
              else:
                     transform_list = transform_list  
              self.transform = transforms.Compose(transform_list)

       def _statistics(self, statistics=None):
              if statistics is None:
                     all_action = [torch.tensor(data['action'][0]).view(1, -1) for data in self.datalist]
                     actions = torch.cat(all_action)
                     
                     self.a_max = actions.max(0)[0]
                     self.a_min = actions.min(0)[0]
                     self.a_mean = actions.mean(0)
                     self.a_std = actions.std(0)

                     all_state = [torch.tensor(data['state']).view(1, -1) for data in self.datalist]
                     states = torch.cat(all_state)
                     self.s_max = states.max(0)[0]
                     self.s_min = states.min(0)[0]
                     self.s_mean = states.mean(0)
                     self.s_std = states.std(0)                     
              else:
                     self.a_max = torch.tensor(statistics['a_max'])
                     self.a_min = torch.tensor(statistics['a_min'])
                     self.a_mean = torch.tensor(statistics['a_mean'])
                     self.a_std = torch.tensor(statistics['a_std'])

                     self.s_max = torch.tensor(statistics['s_max'])
                     self.s_min = torch.tensor(statistics['s_min'])
                     self.s_mean = torch.tensor(statistics['s_mean'])
                     self.s_std = torch.tensor(statistics['s_std'])                     
              
       
       def discretize(self, tensor, num_bins, min_val, max_val):
              """
              discretize the continuous actions from [min_val, max_val] to num_bins bins
              """
              normalized_tensor = (tensor - min_val) / (max_val - min_val)
              discretized_tensor = torch.floor(normalized_tensor * num_bins).clamp(0, num_bins - 1)
              discretized_tensor = discretized_tensor
              return discretized_tensor

       def aug_mask(self, step, image):
              threshold = np.random.uniform(0.4, 0.9)
              dilate_kernel_size = np.random.randint(3, 20) * 2 + 1

              mask_path = step['mask']
              mask = np.load(mask_path, allow_pickle=True)['arr_0']
              mask_output = np.where(mask > threshold, 1, 0).astype(np.uint8)
              kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_kernel_size,dilate_kernel_size)) # ksize=7x7,
              mask_output = cv2.dilate(mask_output,kernel,iterations=1).astype(np.float32)
              mask_output = cv2.GaussianBlur(mask_output, (dilate_kernel_size, dilate_kernel_size), 0)[:,:,np.newaxis]
              
              fill_color=(255, 255, 255)
              fill_tensor = torch.tensor(fill_color, dtype=torch.uint8).repeat(mask.shape[0], mask.shape[1], 1)
              masked_image = np.asarray(image).astype(np.float32) * mask  + fill_tensor.numpy() * (1 - mask)
              masked_image = Image.fromarray(masked_image.astype(np.uint8))  # highlight image with mask
              return masked_image
                      
       def __len__(self):
              return len(self.datalist)
       def __getitem__(self, idx):
              
              step = self.datalist[idx]
              imgs_path = [os.path.join(self.base_dir, step[f'{view}']) for view in self.view_list]
              end_index = frame_length_dic["/".join(imgs_path[0].split("/")[-5:-2])] - 1
              current_idx = int(imgs_path[0].split('/')[-1].split('.')[0])       
              
              if current_idx > end_index - 10:
                     idx -= 10 
                     
                     step = self.datalist[idx]
                     imgs_path = [os.path.join(self.base_dir, step[f'{view}']) for view in self.view_list]
                     current_idx = int(imgs_path[0].split('/')[-1].split('.')[0])
              
              # step = self.datalist[idx]
              # imgs_path = [os.path.join(self.base_dir, step[f'{view}']) for view in self.view_list]
              # end_index = frame_length_dic["/".join(imgs_path[0].split("/")[-5:-2])] - 1
              # current_idx = int(imgs_path[0].split('/')[-1].split('.')[0])
                     
              state = step['state']
              final_state = self.datalist[idx + (end_index - current_idx)]["state"]
              lang = step['instruction']

              # state
              state = torch.tensor(state)
              state_t = (state - self.s_mean) / self.s_std

              state_seq = []
              img_seq = []
              t_seq = []

              # for i in range(0, current_idx + 1): # seq_len = 0 - idx
              #        step = self.datalist[idx - i]
              #        state = step['state']
              #        state = torch.tensor(state)
              #        state = (state - self.s_mean) / self.s_std
              #        state_seq.append(state)

              #        img_seq_path = os.path.join(self.base_dir, step['D435_image'])
              #        img_seq.append(openimage(img_seq_path))

              for i in range(0, 11): 
                     step = self.datalist[idx + i]
                     state = step['state']
                     state = torch.tensor(state)
                     state = (state - self.s_mean) / self.s_std
                     state_seq.append(state)

                     img_seq_path = os.path.join(self.base_dir, step['D435_image'])
                     img_seq.append(openimage(img_seq_path))
                     t_seq.append(torch.tensor(current_idx+i))
              # for i in range(0, min(current_idx,9)+1): 
              #        step = self.datalist[idx - i]
              #        state = step['state']
              #        state = torch.tensor(state)
              #        state = (state - self.s_mean) / self.s_std
              #        state_seq.append(state)

              #        img_seq_path = os.path.join(self.base_dir, step['D435_image'])
              #        img_seq.append(openimage(img_seq_path))
                     
              # if len(state_seq) < 200:
              #        pad_length = 200 - len(state_seq)
              #        # state_seq = state_seq + [torch.zeros(9)] * pad_length #TODO change this back if not adding 0
              #        state_seq = [torch.zeros(9)] * pad_length + state_seq
              #        img_seq = [openimage_tensor(torch.zeros(3,128,128))] * pad_length + img_seq
              
              actions = step['action'][:self.ac_num]
              processed_action = []
              for action in actions:
                     # discretize the action     
                     action = torch.tensor(action)
                     if self.discretize_actions:
                            action[:6] = self.discretize(action[:6], 256, -1., 1.)
                            action[-1] = torch.tensor(255) if action[-1] == 1. else torch.tensor(0)
                     else:
                            if self.norm == 'mean':
                                   action = (action - self.a_mean) / self.a_std
                            elif self.norm == 'minmax':
                                   action = (action - self.a_min) / (self.a_max - self.a_min) * 2 - 1
                     processed_action.append(action)
                     
              action = torch.cat(processed_action, dim=-1)
              state_seq = torch.stack(state_seq, dim=0)
              t_seq =  torch.stack(t_seq, dim=0)
              img_seq_processed = []
              for img in img_seq:
                     transformed_img = self.transform(img).unsqueeze(0)  # [1, 3, H, W]
                     img_seq_processed.append(transformed_img)

              img_seq = torch.cat(img_seq_processed, dim=0)  # [1, seq_len, 3, H, W]

              # images
              images = [openimage(img_path) for img_path in imgs_path]
              if self.mask_aug:
                     images[0] = self.aug_mask(step, images[0])
              images = torch.cat([self.transform(image).unsqueeze(0).unsqueeze(0) for image in images], dim=1)
              
              # print(f"state_seq shape: {state_seq.shape}")
              # print(f"img_seq shape: {img_seq.shape}")
              # print(f"images shape: {images.shape}")

              return_dict = {"imgs": images,
                            "lang": lang,
                            "proprio": state_t,
                            "t": int(current_idx),
                            "T": int(end_index),
                            "proprio_seq":state_seq,
                            "final_state":final_state,
                            "img_seq":img_seq,
                            "t_seq":t_seq,
                            "final_action":action}
                     
              # err_getitem = False
              # # err_getitem = random.choice([True,False,False,False,False,False,False,False,False,False])
              # if not err_getitem:
              #        step = self.datalist[idx]
              #        imgs_path = [os.path.join(self.base_dir, step[f'{view}']) for view in self.view_list]
              #        end_index = frame_length_dic["/".join(imgs_path[0].split("/")[-5:-2])] - 1
              #        if idx == end_index:
              #               idx = idx - 1
                            
              #        step = self.datalist[idx]
              #        imgs_path = [os.path.join(self.base_dir, step[f'{view}']) for view in self.view_list]
              #        state = step['state']
              #        lang = step['instruction']
                     
              #        # state
              #        state = torch.tensor(state)
              #        state = (state - self.s_mean) / self.s_std
                                   
              #        # images
              #        images = [openimage(img_path) for img_path in imgs_path]
              #        if self.mask_aug:
              #               images[0] = self.aug_mask(step, images[0])           
              #        images = torch.cat([self.transform(image).unsqueeze(0).unsqueeze(0) for image in images], dim=1)
                     
              #        # index
              #        current_idx = int(imgs_path[0].split('/')[-1].split('.')[0])

              #        #reward
              #        if end_index - current_idx <= 5:
              #               reward = 0
              #        else:
              #               reward = -1
              #        ### get next state 
              #        step_next = self.datalist[idx+1]
              #        imgs_path = [os.path.join(self.base_dir, step_next[f'{view}']) for view in self.view_list]
              #        state_next = step_next['state']
                     
              #        # state
              #        state_next = torch.tensor(state_next)
              #        state_next = (state_next - self.s_mean) / self.s_std
                                   
              #        # images
              #        image_next = [openimage(img_path) for img_path in imgs_path]
              #        if self.mask_aug:
              #               image_next[0] = self.aug_mask(step_next, images[0])           
              #        images_next = torch.cat([self.transform(image).unsqueeze(0).unsqueeze(0) for image in image_next], dim=1)
                     
                     
              #        return_dict = {"imgs": images,
              #                      "lang": lang,
              #                      "proprio": state,
              #                      "t": int(current_idx),
              #                      "T": int(end_index),
              #                      "imgs_next": images_next,
              #                      "proprio_next": state_next,
              #                      "reward":reward}
                     
              # else:
              #        err_length = len(self.err_datalist) - 1
              #        idx = random.randint(0,err_length-1)   
                     
              #        step = self.err_datalist[idx]
              #        imgs_path = [os.path.join(self.base_dir, step[f'{view}']) for view in self.view_list]                     
              #        state = step['state']
              #        lang = step['instruction']
              #        reward = step["reward"]
                     
              #        # state
              #        state = torch.tensor(state)
              #        state = (state - self.s_mean) / self.s_std
                                   
              #        # images
              #        images = [openimage(img_path) for img_path in imgs_path]
              #        if self.mask_aug:
              #               images[0] = self.aug_mask(step, images[0])           
              #        images = torch.cat([self.transform(image).unsqueeze(0).unsqueeze(0) for image in images], dim=1)
                     
              #        # index
              #        current_idx = imgs_path[0].split('/')[-1].split('.')[0] 
              #        end_index = frame_length_dic["libero_goal/{}_demo/{}".format(*imgs_path[0].replace(" ", "_").split("/")[-4:-2])] - 1 

              #        ### get next state 
              #        step_next = self.err_datalist[idx+1]
              #        imgs_path = [os.path.join(self.base_dir, step_next[f'{view}']) for view in self.view_list]
              #        state = step_next['state']
              #        lang = step_next['instruction']
                     
              #        # state
              #        state = torch.tensor(state)
              #        state_next = (state - self.s_mean) / self.s_std
                                   
              #        # images
              #        images_next = [openimage(img_path) for img_path in imgs_path]
              #        if self.mask_aug:
              #               images_next[0] = self.aug_mask(step_next, images[0])           
              #        images_next = torch.cat([self.transform(image).unsqueeze(0).unsqueeze(0) for image in images_next], dim=1)
                     
                     
              #        return_dict = {"imgs": images,
              #                      "lang": lang,
              #                      "proprio": state,
              #                      "t": int(current_idx),
              #                      "T": int(end_index),
              #                      "imgs_next": images_next,
              #                      "proprio_next": state_next,
              #                      "reward":reward}
              
              # return image goal or not
              if self.img_goal:
                     img_begin_path, img_end_path = get_libero_frame.get_demofixed_idx_begin_frame(step,self.base_dir,frame_length_dic)
                     transform = transforms.ToTensor()
                     img_begin = transform(openimage(img_begin_path))
                     img_end = transform(openimage(img_end_path))
                     return_dict.update({"img_begin": img_begin, "img_end": img_end})
                           
              return return_dict
          
          
   
   
def AIRKitchenDataLoader_err(
       base_dir: str='',
       datalist: list=['/home/dodo/ljx/BearRobot/data/bridge/AIR-toykitchen.json'],
       img_size: int=128,
       frames: int=1,
       view_list: list=['D435_image', 'wrist_image'],
       norm: str="minmax",
       discretize_actions: bool=False,
       batch_size: int=64,
       num_workers: int=8,
       pin_mem: bool=True,
       ac_num: int=4,
       transform_list: list=None,
       img_goal: bool=False,
       **kwargs,
):
       dataset = AIRKitchenDataset_err(base_dir=base_dir,
                                   datalist=datalist,
                                   frames=frames, 
                                   img_size=img_size, 
                                   view_list=view_list, 
                                   discretize_actions=discretize_actions, 
                                   norm=norm,
                                   ac_num=ac_num,
                                   transform_list=transform_list,
                                   img_goal=img_goal)
       
       num_tasks = ddp.get_world_size()
       global_rank = ddp.get_rank()
       sampler = DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
       )
       
       dataloader = DataLoader(
              dataset, 
              sampler=sampler,
              batch_size=batch_size // num_tasks, 
              num_workers=num_workers,
              pin_memory=pin_mem,
              drop_last=True
       )
       
       statistics = {"a_min": dataset.a_min.cpu().numpy().tolist(),
                     "a_max": dataset.a_max.cpu().numpy().tolist(),
                     "a_mean": dataset.a_mean.cpu().numpy().tolist(),
                     "a_std": dataset.a_std.cpu().numpy().tolist(),
                     "s_min": dataset.s_min.cpu().numpy().tolist(),
                     "s_max": dataset.s_max.cpu().numpy().tolist(),
                     "s_mean": dataset.s_mean.cpu().numpy().tolist(),
                     "s_std": dataset.s_std.cpu().numpy().tolist(),}
       return dataloader, statistics
  