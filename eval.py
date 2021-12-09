import collections

import numpy as np

import scores
import utils
import model
import torch
import tensorflow as tf
from dataloader import SupervisedDataset
from torch.utils.data import DataLoader
import pandas as pd

## initial config
N_EPOCHS = 1000
RESULT_PER_EPOCH = 10 * 1
CHECKPOINT_PER_EPOCH = 50
CHECKPOINT_PATH = 'checkpoints/'

NH_LSTM = 128
NH_BOTTLENECK = 256

ENV_SIZE = 2.2
BATCH_SIZE = 10
GRAD_CLIPPING = 1e-5
SEED = 9101
N_PC = [256]
N_HDC = [12]
# Change Dropout to the required format
BOTTLENECK_DROPOUT = [0.5]
WEIGHT_DECAY = 1e-5
LR = 1e-5
MOMENTUM = 0.9
TIME = 50
PAUSE_TIME = None
SAVE_LOC = 'experiments/'

scores_filename = 'rates_'
scores_directory = 'results/scores/'

# path = 'data/tf-records/'
path = 'data/'
DatasetInfo = collections.namedtuple(
    'DatasetInfo', ['basepath', 'size', 'sequence_length', 'coord_range'])

_DATASETS = dict(
    square_room=DatasetInfo(
        basepath='square_room_100steps_2.2m_1000000',
        size=100,
        sequence_length=100,
        coord_range=((-1.1, 1.1), (-1.1, 1.1))), )

ds_info = _DATASETS['square_room']

feature_map = {
    'init_pos':
        tf.io.FixedLenFeature(shape=[2], dtype=tf.float32),
    'init_hd':
        tf.io.FixedLenFeature(shape=[1], dtype=tf.float32),
    'ego_vel':
        tf.io.FixedLenFeature(
            shape=[ds_info.sequence_length, 3],
            dtype=tf.float32),
    'target_pos':
        tf.io.FixedLenFeature(
            shape=[ds_info.sequence_length, 2],
            dtype=tf.float32),
    'target_hd':
        tf.io.FixedLenFeature(
            shape=[ds_info.sequence_length, 1],
            dtype=tf.float32),
}

data_params = {
    'batch_size': BATCH_SIZE,
    'shuffle': True,
    'num_workers': 2}
test_params = {
    'batch_size': 100,
    'shuffle': True,
    'num_workers': 6}

# Equivalent of tf.nn.softmax_crossentropy_with_logits
Loss_Function = torch.nn.MultiLabelSoftMarginLoss(reduction='none')


def softmax_crossentropy_with_logits(labels, logits):
    labels_2d = labels.reshape((-1, labels.size()[2]))
    logits_2d = logits.reshape((-1, logits.size()[2]))
    return Loss_Function(logits_2d, labels_2d)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # loading the data
    data_dic = utils.load_datadic_from_tfrecords(path, _DATASETS, 'square_room', feature_map)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('Using device:', device)

    # Dataset and Dataloader
    dataset = SupervisedDataset(data_dic)
    dataloader = DataLoader(dataset, **data_params)

    # Getting place and head direction cell ensembles
    place_cell_ensembles = utils.get_place_cell_ensembles(
        env_size=ENV_SIZE,
        neurons_seed=SEED,
        targets_type='softmax',
        lstm_init_type='softmax',
        n_pc=N_PC,
        pc_scale=[0.01],
        device=device)

    head_direction_ensembles = utils.get_head_direction_ensembles(
        neurons_seed=SEED,
        targets_type='softmax',
        lstm_init_type='softmax',
        n_hdc=N_HDC,
        hdc_concentration=[20.],
        device=device)

    target_ensembles = place_cell_ensembles + head_direction_ensembles

    # Defining the model and getting its parameters
    gridtorchmodel = model.GridTorch(target_ensembles, NH_LSTM, NH_BOTTLENECK,
                                     dropoutrates_bottleneck=BOTTLENECK_DROPOUT).to(device)
    params = gridtorchmodel.parameters()

    # Optimizer
    optimizer = torch.optim.RMSprop(params,
                                    lr=LR,
                                    momentum=MOMENTUM,
                                    alpha=0.9,
                                    eps=1e-10)

    saved_epochs = [0, 5, 10]
    for saved_epoch in saved_epochs:
        # Creating Scorer Objects
        starts = [0.2] * 10
        ends = (np.linspace(0.4, 1.0, num=10)).tolist()
        masks_parameters = zip(starts, ends)

        latest_epoch_scorer = scores.GridScorer(20, _DATASETS['square_room'].coord_range,
                                                masks_parameters)

        # Loading Selected Checkpoint
        checkpoint = torch.load(CHECKPOINT_PATH + f'model_{saved_epoch:04d}.pt')
        gridtorchmodel.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f'Epoch: {epoch:04d}  loss: {loss}')

        gridtorchmodel.eval()

        # Creating Scorer Objects
        starts = [0.2] * 10
        ends = (np.linspace(0.4, 1.0, num=10)).tolist()
        masks_parameters = zip(starts, ends)

        latest_epoch_scorer = scores.GridScorer(20, _DATASETS['square_room'].coord_range,
                                                masks_parameters)

        activations = []
        posxy = []

        print('Evaluating ...')


        activations = []
        posxy = []
        EVAL_STEPS = 400
        steps = 0

        for X, y in dataloader:
            steps += 1

            init_pos, init_hd, ego_vel = X
            target_pos, target_hd = y

            init_pos = init_pos.to(device)
            init_hd = init_hd.to(device)
            ego_vel = torch.swapaxes(ego_vel.to(device), 0, 1)

            target_pos = target_pos.to(device)
            target_hd = target_hd.to(device)

            # Getting initial conditions
            init_conds = utils.encode_initial_conditions(init_pos, init_hd, place_cell_ensembles,
                                                         head_direction_ensembles)

            # Getting ensemble targets
            ensemble_targets = utils.encode_targets(target_pos, target_hd, place_cell_ensembles,
                                                    head_direction_ensembles)

            # Running through the model
            outs = gridtorchmodel(ego_vel, init_conds)

            # Collecting different parts of the output
            logits_hd, logits_pc, bottleneck_acts, rnn_states, rnn_cells = outs
            # print(f'hd logits size: {logits_hd.size()}')
            # print(f'pc logits size: {logits_pc.size()}')
            # print()
            # print(f'hd targets size: {ensemble_targets[0].size()}')
            # print(f'pc targets size: {ensemble_targets[1].size()}')
            #
            # print(ensemble_targets[0].sum(axis=-1) == 1)

            # accumulating targets x and y and activations for scorer
            # only if it's the right epoch!!!!!

            activations.append(torch.swapaxes(bottleneck_acts.detach().cpu(), 0, 1))
            posxy.append(target_pos.detach().cpu())
            if steps == EVAL_STEPS:
                break


        # Evaluation
        activations = torch.cat(activations).numpy()
        posxy = torch.cat(posxy).numpy()
        results_filename = scores_filename + f'{epoch:04d}.pdf'
        utils.get_scores_and_plot(latest_epoch_scorer, posxy, activations, scores_directory, results_filename,
                                      sort_by_score_60=False)


