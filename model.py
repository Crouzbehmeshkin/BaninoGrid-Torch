"""
Pytorch implementation of the supervised part of:

"Vector-based navigation using grid-like representations in artificial agents" (Banino et al., 2018)

Based on a version written by Lucas Pompe in 2019
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


def truncated_normal_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def init_trunc_normal(t, size):
    std = 1. / np.sqrt(size)
    return truncated_normal_(t, 0, std)


# https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.Tensor(input_size, hidden_size * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.init_weights()

    def init_weights(self):
        # stdv = 1.0 / np.sqrt(self.hidden_size)
        nn.init.kaiming_uniform_(self.W)
        nn.init.kaiming_uniform_(self.U)
        nn.init.zeros_(self.bias)

    def forward(self, x_t, prev_state):
        HS = self.hidden_size
        h_t, c_t = prev_state

        # batch the computations into a single matrix multiplication
        gates = x_t @ self.W + h_t @ self.U + self.bias
        i_t, f_t, g_t, o_t = (
            torch.sigmoid(gates[:, :HS]),  # input
            torch.sigmoid(gates[:, HS:HS * 2]),  # forget
            torch.tanh(gates[:, HS * 2:HS * 3]),
            torch.sigmoid(gates[:, HS * 3:]),  # output
        )
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class GridTorch(nn.Module):
    """LSTM core implementation for the grid cell network."""
    def __init__(self,
                 target_ensembles,
                 nh_lstm,
                 nh_bottleneck,
                 nh_embed=None,
                 dropoutrates_bottleneck=None,
                 bottleneck_weight_decay=0.0,
                 bottleneck_has_bias=False,
                 init_weight_disp=0.0):
        """Constructor of the RNN cell.
            Args:
              target_ensembles: Targets, place cells and head direction cells.
              nh_lstm: Size of LSTM cell.
              nh_bottleneck: Size of the linear layer between LSTM output and output.
              nh_embed: Number of hiddens between input and LSTM input.
              dropoutrates_bottleneck: Iterable of keep rates (0,1]. The linear layer is
                partitioned into as many groups as the len of this parameter.
              bottleneck_weight_decay: Weight decay used in the bottleneck layer.
              bottleneck_has_bias: If the bottleneck has a bias.
              init_weight_disp: Displacement in the weights initialisation.
              name: the name of the module.
            """
        super(GridTorch, self).__init__()
        self._target_ensembles = target_ensembles
        self._nh_lstm = nh_lstm
        self._nh_bottleneck = nh_bottleneck
        self._nh_embed = nh_embed
        self._dropoutrates_bottleneck = dropoutrates_bottleneck
        self._bottleneck_weight_decay = 0.0
        self._bottleneck_has_bias = False
        self._init_weight_disp = 0.0

        self._init_conds_size = 268

        #LSTM Layer
        self.rnn = CustomLSTM(3, self._nh_lstm,)
        self.init_h_embed = nn.Linear(self._init_conds_size, self._nh_lstm)
        self.init_c_embed = nn.Linear(self._init_conds_size, self._nh_lstm)

        #Linear "Bottleneck" Layer
        self.bottleneck = nn.Linear(self._nh_lstm, self._nh_bottleneck,
                                    bias=bottleneck_has_bias)

        #Dropout
        self.dropouts = []
        for rate in self._dropoutrates_bottleneck:
            self.dropouts.append(nn.Dropout(rate))
        self.section_size = self._nh_bottleneck // len(self.dropouts)

        #Linear layer projecting to place and head direction cells
        self.pc_logits = nn.Linear(self._nh_bottleneck, self._target_ensembles[0].n_cells)
        self.hd_logits = nn.Linear(self._nh_bottleneck, self._target_ensembles[1].n_cells)

        #initializing weights
        self.init_weights()

    @property
    def l2_loss(self, ):
        return (self.bottleneck.weight.norm(2) +
                self.pc_logits.weight.norm(2) +
                self.hd_logits.weight.norm(2))

    def init_weights(self):
        self.init_h_embed.weight = init_trunc_normal(self.init_h_embed.weight, 128)
        self.init_c_embed.weight = init_trunc_normal(self.init_c_embed.weight, 128)
        self.bottleneck.weight = init_trunc_normal(self.bottleneck.weight, 256)
        self.pc_logits.weight = init_trunc_normal(self.pc_logits.weight, 256)
        self.hd_logits.weight = init_trunc_normal(self.hd_logits.weight, 12)

        nn.init.zeros_(self.init_h_embed.bias)
        nn.init.zeros_(self.init_c_embed.bias)
        nn.init.zeros_(self.pc_logits.bias)
        nn.init.zeros_(self.hd_logits.bias)


    def forward(self, x, init_conds):
        init = torch.cat(init_conds, dim=1)

        #Getting initial hidden and cell states from their respective layers
        init_h = self.init_h_embed(init)
        init_c = self.init_c_embed(init)

        #Preparing to run through the LSTM
        h_t, c_t = init_h, init_c
        logits_hd = []
        logits_pc = []
        bottleneck_acts = []
        rnn_states = []
        rnn_cells = []

        #Going through the LSTM
        for x_t in x:
            h_t, c_t = self.rnn(x_t, (h_t, c_t))

            bottleneck_out = self.bottleneck(h_t)

            #splitting and doing dropout for each split
            splits = torch.split(bottleneck_out, self.section_size, dim=1)
            split_drops_out = []
            for i, split in enumerate(splits):
                split_drops_out.append(self.dropouts[i](split))
            dropout_out = torch.concat(split_drops_out, dim=1)

            #place cell and head direction cell predictions
            pc_preds = self.pc_logits(dropout_out)
            hd_preds = self.hd_logits(dropout_out)

            #accumulating results
            logits_hd.append(hd_preds)
            logits_pc.append(pc_preds)
            bottleneck_acts.append(dropout_out)
            rnn_states.append(h_t)
            rnn_cells.append(c_t)

        final_state = h_t
        outs = (torch.stack(logits_hd),
                torch.stack(logits_pc),
                torch.stack(bottleneck_acts),
                torch.stack(rnn_states),
                torch.stack(rnn_cells))

        return outs

