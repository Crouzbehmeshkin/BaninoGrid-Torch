import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class SupervisedDataset(Dataset):
    def __init__(self, data_dic):
        self.num_samples = len(data_dic['init_pos'])                # 1000000
        self.sequence_length = 100                                  # = ds_info.sequence_length

        # input
        self.init_pos = np.stack(data_dic['init_pos'])              # shape = (num_samples, 2)
        self.init_hd = np.stack(data_dic['init_hd'])                # shape = (num_samples, 1)
        self.ego_vel = np.stack(data_dic['ego_vel'])                # shape = (num_samples, seq_len, 3)

        # output
        self.target_pos = np.stack(data_dic['target_pos'])          # shape = (num_samples, seq_len, 2)
        self.target_hd = np.stack(data_dic['target_hd'])            # shape = (num_samples, seq_len, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample_init_pos = self.init_pos[idx:idx + 1, :]
        sample_init_hd = self.init_hd[idx:idx + 1, :]
        sample_ego_vel = self.ego_vel[idx, :, :]

        sample_target_pos = self.target_pos[idx, :, :]
        sample_target_hd = self.target_hd[idx, :, :]

        return ((sample_init_pos, sample_init_hd, sample_ego_vel),
                (sample_target_pos, sample_target_hd))