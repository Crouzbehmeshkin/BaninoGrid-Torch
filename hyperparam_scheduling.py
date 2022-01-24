import numpy as np


class DPScheduler():
    def __init__(self, upperbound, lowerbound, total_epochs, update_frequency, init_dp):
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.total_epochs = total_epochs
        self.update_freq = update_frequency
        self.init_dp = init_dp
        self.max_step_size = min(abs(upperbound - init_dp), abs(lowerbound - init_dp))/(total_epochs / update_frequency)
        self.scaling_factors = [10**(-i) for i in range(7)]
        self.current_dp = init_dp

    def get_dp(self, current_loss, prev_loss):
        loss_diff = current_loss - prev_loss
        max_step_scaled = self.max_step_size / loss_diff
        scaling_factor = min(self.scaling_factors, key=lambda x: abs(x - abs(max_step_scaled)))
        delta_dp = loss_diff * scaling_factor
        self.current_dp -= delta_dp
        return self.current_dp


class LRScheduler():
    def __init__(self, lr_max, lr_min, n_updates):
        self.lr_max = lr_max
        self.lr_min = lr_min
        n_updates = n_updates

    def get_all_lr(self):
        lr_all = np.logspace(np.log(self.lr_max), np.log(self.lr_min), 100, base=np.exp(1)).tolist()
        # Safe measure in case of training for more epochs
        for i in range(3):
            lr_all.append(lr_all[-1])
        return lr_all