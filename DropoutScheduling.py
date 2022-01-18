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
