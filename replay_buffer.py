import numpy as np

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
        '''
        重放缓冲区，看起来是连续存放的
        obs_shape: 观察空间shape
        action_shape: 动作空间shape
        capacity: 缓冲区容量
        image_pad: 图像填充大小 todo 这个参数的作用？
        device: 设备
        '''
        self.capacity = capacity
        self.device = device

        # todo 看起来是先填充图片的大小后，然后随机裁剪
        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        
        # 循环存储，如果超过了最大的数量则从头开始
        self.idx = (self.idx + 1) % self.capacity
        # 标识是否已经达到了最大存储量
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        # 随机选择batch_size个样本
        # 看起来感觉采样的步数连续的样本？
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        obses_aug = obses.copy()
        next_obses_aug = next_obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
        next_obses_aug = torch.as_tensor(next_obses_aug,
                                         device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        obses = self.aug_trans(obses) # 观察增强
        next_obses = self.aug_trans(next_obses)

        obses_aug = self.aug_trans(obses_aug)
        next_obses_aug = self.aug_trans(next_obses_aug)

        return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug
