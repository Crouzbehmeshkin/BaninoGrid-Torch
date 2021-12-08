import torch
import numpy as np
import torch.nn.functional as F


def one_hot_max(x, axis=-1):
    max_idx = torch.argmax(x, dim=axis)
    depth = x.shape[-1]
    return F.one_hot(max_idx, depth)


def softmax(x, axis=-1):
    return F.softmax(x, dim=axis)


def softmax_sample(x):
    dist = torch.distributions.one_hot_categorical.OneHotCategorical(logits=x)
    return dist.sample().type(torch.FloatTensor)


Loss_Function = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')


class CellEnsemble(object):
    def __init__(self, n_cells, soft_targets, soft_init, device):
        self.n_cells = n_cells
        if soft_targets not in ["softmax", "voronoi", "sample", "normalized"]:
            raise ValueError
        else:
            self.soft_targets = soft_targets
        # Provide initialization of LSTM in the same way as targets if not specified
        # i.e one-hot if targets are Voronoi
        if soft_init is None:
            self.soft_init = soft_targets
        else:
            if soft_init not in [
                "softmax", "voronoi", "sample", "normalized", "zeros"
            ]:
                raise ValueError
            else:
                self.soft_init = soft_init

    def get_targets(self, x):
        if self.soft_targets == "normalized":
            targets = torch.exp(self.unnor_logpdf(x))
        elif self.soft_targets == "softmax":
            lp = self.log_posterior(x)
            targets = softmax(lp)
        elif self.soft_targets == "sample":
            lp = self.log_posterior(x)
            targets = softmax_sample(lp)
        elif self.soft_targets == "voronoi":
            lp = self.log_posterior(x)
            targets = one_hot_max(lp)
        return targets

    def get_init(self, x):
        """Type of initialisation."""
        if self.soft_init == "normalized":
            init = torch.exp(self.unnor_logpdf(x))
        elif self.soft_init == "softmax":
            lp = self.log_posterior(x)
            init = softmax(lp)
        elif self.soft_init == "sample":
            lp = self.log_posterior(x)
            init = softmax_sample(lp)
        elif self.soft_init == "voronoi":
            lp = self.log_posterior(x)
            init = one_hot_max(lp)
        elif self.soft_init == "zeros":
            init = torch.zeros_like(self.unnor_logpdf(x))
        return init

    def loss(self, predictions, targets):
        """Loss."""

        if self.soft_targets == "normalized":
            smoothing = 1e-2
            labels = (1. - smoothing) * targets + smoothing * 0.5
            loss = Loss_Function(predictions, labels)
        else:
            loss = Loss_Function(predictions, labels)
        return loss

    def log_posterior(self, x):
        logp = self.unnor_logpdf(x)
        log_posteriors = logp - torch.logsumexp(logp, dim=2, keepdim=True)
        return log_posteriors


class PlaceCellEnsemble(CellEnsemble):
    """Calculates the dist over place cells given an absolute position."""

    def __init__(self, n_cells, device, stdev=0.35, pos_min=-5, pos_max=5, seed=None,
               soft_targets=None, soft_init=None):
        super(PlaceCellEnsemble, self).__init__(n_cells, soft_targets, soft_init, device)
        # Create a random MoG with fixed cov over the position (Nx2)
        
        rs = np.random.RandomState(seed)
        self.means = rs.uniform(pos_min, pos_max, size=(self.n_cells, 2))
        self.means = torch.Tensor(self.means).to(device)

        self.variances = torch.ones_like(self.means).to(device) * stdev**2

    def unnor_logpdf(self, trajs):
        # Output the probability of each component at each point (BxTxN)
        diff = trajs[:, :, None, :] - self.means[np.newaxis, np.newaxis, ...]
        unnor_logp = -0.5 * torch.sum((diff**2)/self.variances, dim=-1)
        return unnor_logp


class HeadDirectionCellEnsemble(CellEnsemble):
    """Calculates the dist over HD cells given an absolute angle."""

    def __init__(self, n_cells, device, concentration=20, seed=None,
           soft_targets=None, soft_init=None):
        super(HeadDirectionCellEnsemble, self).__init__(n_cells,
                                                        soft_targets,
                                                        soft_init,
                                                        device)
        # Create a random Von Mises with fixed cov over the position
        rs = np.random.RandomState(seed)
        self.means = rs.uniform(-np.pi, np.pi, (n_cells))
        self.means = torch.Tensor(self.means).to(device)

        self.kappa = torch.ones_like(self.means).to(device) * concentration

    def unnor_logpdf(self, x):
        return self.kappa * torch.cos(x - self.means[np.newaxis, np.newaxis, :])