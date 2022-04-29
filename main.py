import collections

import numpy as np

import scores
import utils
import model
import torch
import tensorflow as tf
from dataloader import SupervisedDataset
from torch.utils.data import DataLoader
from hyperparam_scheduling import DPScheduler, LRScheduler
import torch.nn.functional as F
import os

# for running stuff locally (Tensorflow didn't support my cuda version)
tf.config.set_visible_devices([], 'GPU')

# for turning off annoying warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

##############################################################################
########################### ADDED CONFIGURATION ##############################
##############################################################################
# Neuromodulation config
LSTM_TYPE = 'Simple_NM'  # default, Simple_NM

RAND_INIT_NOISE = False

# Dropout Scheduling config
DP_SCHEDULING = True
DP_LOWERB = 0.2
DP_UPPERB = 0.8
DP_INIT = 0.5
DP_UPDATE_FREQ = 10  # in epochs

# Learning rate scheduling
LR_SCHEDULING = True
LR_MAX = 1e-4
LR_MIN = LR_MAX*1e-1
LR_UPDATE_FREQ = 10

############################################################################
############################### BASE CONFIG ################################
############################################################################
N_EPOCHS = 1000
RESULT_PER_EPOCH = 10 * 1
CHECKPOINT_PER_EPOCH = 5
EVAL_STEPS = 400 # Original 400
CHECKPOINT_PATH = 'checkpoints/run33/'

NH_LSTM = 128
NH_BOTTLENECK = 256

ENV_SIZE = 2.2
BATCH_SIZE = 10  # original 10
TRAINING_STEPS_PER_EPOCH = 1000    # original 1000
GRAD_CLIPPING = 1e-5  # original 1e-5
SEED = 9101
N_PC = [256]
N_HDC = [12]
BOTTLENECK_DROPOUT = [0.5]
WEIGHT_DECAY = 1e-5
LR = 1e-4  # Original 1e-5
MOMENTUM = 0.9  # Original 0.9
TIME = 50
PAUSE_TIME = None
SAVE_LOC = 'experiments/'

# TRAIN_DATA_RANGE = [0, 3]
# TEST_DATA_RANGE = [3, 6]
TRAIN_DATA_RANGE = [0, 90]
TEST_DATA_RANGE = [90, 100]

scores_filename = 'rates_'
scores_directory = 'results/scores/'
base_trace_filename = 'traces_'
trace_directory = 'results/traces/'

# path = 'data/tmp/'
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
    'batch_size': BATCH_SIZE,
    'shuffle': True,
    'num_workers': 2}

# For storing losses
epoch_losses = []
test_losses = []
epoch_hd_losses = []
epoch_pc_losses = []
scheduled_dropouts = []


# Equivalent of tf.nn.softmax_crossentropy_with_logits
def Loss_Function(logits, labels):
    logits = F.log_softmax(logits, dim=-1)
    return -(labels*logits).sum(dim=-1)


def softmax_crossentropy_with_logits(labels, logits):
    logits= torch.swapaxes(logits, 0, 1)
    # print('labels', labels.size(), labels[(labels!=0)&(labels!=1)].size())
    # print('logits', logits.size())
    # if labels.size()[2] == 12:
    #     print(labels[0,0,:])
    labels_2d = labels.reshape((-1, labels.size()[2]))
    logits_2d = logits.reshape((-1, logits.size()[2]))
    return Loss_Function(logits_2d, labels_2d)


def log_losses(losses, pc_losses, hd_losses):
    losses_np = np.array(losses)
    epoch_loss_mean = losses_np.mean()
    epoch_loss_std = losses_np.std()
    epoch_losses.append(epoch_loss_mean)
    print(f'Epoch {epoch:4d}:  Loss Mean:{epoch_loss_mean:4.4f}  Loss Std:{epoch_loss_std:4.4f}')

    pc_losses_np = np.array(pc_losses)
    hd_losses_np = np.array(hd_losses)
    pc_losses_mean = pc_losses_np.mean()
    pc_losses_std = pc_losses_np.std()
    epoch_pc_losses.append(pc_losses_mean)
    hd_losses_mean = hd_losses_np.mean()
    hd_losses_std = hd_losses_np.std()
    epoch_hd_losses.append(hd_losses_mean)
    print(f'Mean PC loss:{pc_losses_mean:4.4f}  PC loss std:{pc_losses_std:4.4f}')
    print(f'Mean HD loss:{hd_losses_mean:4.4f}  HD loss std:{hd_losses_std:4.4f}')


def save_log_files():
    epoch_losses_np = np.array(epoch_losses)
    np.save('epochlosses.npy', epoch_losses_np)

    test_losses_np = np.array(test_losses)
    np.save('testlosses.npy', test_losses_np)

    epoch_hd_losses_np = np.array(epoch_hd_losses)
    np.save('epochhdlosses.npy', epoch_hd_losses_np)

    epoch_pc_losses_np = np.array(epoch_pc_losses)
    np.save('epochpclosses.npy', epoch_pc_losses_np)

    if DP_SCHEDULING:
        dropouts_np = np.array(scheduled_dropouts)
        np.save('dropouts.npy', dropouts_np)


def log_evaluations(losses, activations, target_posxy, pred_posxy):
    losses_t = torch.tensor(losses)
    test_loss_mean = losses_t.mean()
    test_loss_std = losses_t.std()
    test_losses.append(test_loss_mean)
    print(f'Mean test loss: {test_loss_mean:4.4f}  Test loss std: {test_loss_std:4.4f}')

    activations_np = torch.cat(activations).cpu().numpy()
    target_posxy_np = torch.cat(target_posxy).cpu().numpy()
    pred_posxy_np = torch.cat(pred_posxy).cpu().numpy()

    results_filename = scores_filename + f'{epoch:04d}.pdf'
    trace_filename = base_trace_filename + f'{epoch:04d}.pdf'
    utils.get_scores_and_plot(latest_epoch_scorer, target_posxy_np, activations_np, scores_directory, results_filename)
    utils.get_traces_and_plot(target_posxy_np, pred_posxy_np, place_cell_ensembles[0].means.cpu().numpy(), trace_directory,
                              trace_filename)
    spatial_err_mean, spatial_err_std = utils.get_spatial_error(target_posxy_np, pred_posxy_np, place_cell_ensembles[0].means.cpu().numpy())
    print(f'Mean spatial error: {spatial_err_mean:4.4f}')
    print(f'Spatial error std: {spatial_err_std:4.4f}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Creating the checkpoint path
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    # loading the data
    train_data_dic = utils.load_datadic_from_tfrecords(path, _DATASETS, 'square_room', feature_map, TRAIN_DATA_RANGE)
    test_data_dic = utils.load_datadic_from_tfrecords(path, _DATASETS, 'square_room', feature_map, TEST_DATA_RANGE)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print('Using device:', device)

    # Dataset and Dataloader
    train_dataset = SupervisedDataset(train_data_dic)
    train_dataloader = DataLoader(train_dataset, **data_params)

    test_dataset = SupervisedDataset(test_data_dic)
    test_dataloader = DataLoader(test_dataset, **data_params)

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
                                     dropoutrates_bottleneck=BOTTLENECK_DROPOUT,
                                     LSTM_type=LSTM_TYPE).to(device)
    params = gridtorchmodel.parameters()

    # Optimizer
    optimizer = torch.optim.RMSprop(params,
                                    lr=LR,
                                    momentum=MOMENTUM,
                                    alpha=0.9,
                                    eps=1e-10)
    # optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM)

    # For Dropout Scheduling
    dp_scheduler = None
    if DP_SCHEDULING:
        dp_scheduler = DPScheduler(DP_UPPERB, DP_LOWERB, N_EPOCHS, DP_UPDATE_FREQ, DP_INIT)

    # For learning rate scheduling
    if LR_SCHEDULING:
        lr_scheduler = LRScheduler(LR_MAX, LR_MIN, N_EPOCHS//LR_UPDATE_FREQ + 1)
        lr_all = lr_scheduler.get_all_lr()

    start_epoch = 995
    # if trying to resume training from a checkpoint, comment otherwise
    checkpoint = torch.load(CHECKPOINT_PATH + f'model_{start_epoch:04d}.pt')
    for key in checkpoint:
        print(f'key {key}')
    start_epoch = checkpoint['epoch'] + 1
    gridtorchmodel.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch_losses = []
    # test_losses = []
    # epoch_pc_losses = []
    # epoch_hd_losses = []
    epoch_losses = checkpoint['epoch_losses']
    test_losses = checkpoint['test_losses']
    epoch_pc_losses = checkpoint['epoch_pc_losses']
    epoch_hd_losses = checkpoint['epoch_hd_losses']

    scheduled_dropouts = np.load('dropouts.npy').tolist()
    print(f'loaded current dp {scheduled_dropouts[-1]}')
    dp_scheduler.current_dp = scheduled_dropouts[-1]

    # Creating Scorer Objects
    starts = [0.2] * 10
    ends = (np.linspace(0.4, 1.0, num=10)).tolist()
    masks_parameters = zip(starts, ends)

    latest_epoch_scorer = scores.GridScorer(20, _DATASETS['square_room'].coord_range,
                                            masks_parameters)


    print('Started Training ...')
    # Training Loop

    for epoch in range(start_epoch, N_EPOCHS + 1):
        gridtorchmodel.train()
        step = 1
        losses = []
        hd_losses = []
        pc_losses = []

        activations = []
        posxy = []

        if LR_SCHEDULING:
            if epoch % LR_UPDATE_FREQ == 0:
                for param_gp in optimizer.param_groups:
                    param_gp['lr'] = lr_all[epoch//LR_UPDATE_FREQ]

        if RAND_INIT_NOISE:
            angle = np.random.uniform(0, 2) * np.pi
            radius = 0.2
            init_noise_mean = torch.Tensor([radius*np.cos(angle), radius*np.sin(angle)])
            init_noise_cov = torch.eye(2)*radius/2
            dist = torch.distributions.multivariate_normal.MultivariateNormal(init_noise_mean, init_noise_cov)

        # Training for the specified number of steps
        for X, y in train_dataloader:
            optimizer.zero_grad()

            init_pos, init_hd, ego_vel = X
            target_pos, target_hd = y

            if RAND_INIT_NOISE:
                init_pos = init_pos + dist.sample(init_pos.shape[:-1])
                init_pos = torch.clamp(init_pos, -1.1, 1.1)
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

            # Initial Hebbian Trace
            if LSTM_TYPE == 'Simple_NM':
                hebb = torch.zeros((init_pos.shape[0], NH_LSTM, NH_LSTM), device=device)
            else:
                hebb = None

            # Running through the model
            outs = gridtorchmodel(ego_vel, init_conds, hebb)

            # Collecting different parts of the output
            logits_hd, logits_pc, bottleneck_acts, rnn_states, rnn_cells = outs

            pc_loss = softmax_crossentropy_with_logits(labels=ensemble_targets[0], logits=logits_pc)
            hd_loss = softmax_crossentropy_with_logits(labels=ensemble_targets[1], logits=logits_hd)

            total_loss = pc_loss + hd_loss
            train_loss = total_loss.mean()

            # weight decay
            train_loss += gridtorchmodel.l2_loss * WEIGHT_DECAY

            train_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_value_(params, GRAD_CLIPPING)

            optimizer.step()

            losses.append(train_loss.clone().item())
            hd_losses.append(hd_loss.mean().clone().item())
            pc_losses.append(pc_loss.mean().clone().item())
            # print('Loss:', losses[-1])

            if step >= TRAINING_STEPS_PER_EPOCH:
                break
            step += 1

        # Logging
        log_losses(losses, pc_losses, hd_losses)

        # Dropout Scheduling
        # only works when we have a single dropout rate (no lists!)
        if DP_SCHEDULING and epoch % DP_UPDATE_FREQ == 0:
            if epoch == 0:
                print(f'Initial DP: {DP_INIT:4.4f}')
                scheduled_dropouts.append(DP_INIT)
            else:
                new_dp = dp_scheduler.get_dp(epoch_losses[-1], epoch_losses[-DP_UPDATE_FREQ - 1])
                gridtorchmodel.dropouts[0].p = 1 - new_dp
                scheduled_dropouts.append(new_dp)
                print(f'New DP: {new_dp:4.4f}')

        # Evaluation
        if epoch % CHECKPOINT_PER_EPOCH == 0:
            gridtorchmodel.eval()
            eval_steps = 1
            losses = []

            activations = []
            target_posxy = []
            pred_posxy = []

            with torch.no_grad():
                for X, y in test_dataloader:
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
                    # Initial Hebbian Trace
                    if LSTM_TYPE == 'Simple_NM':
                        hebb = torch.zeros((init_pos.shape[0], NH_LSTM, NH_LSTM), device=device)
                    else:
                        hebb = None
                    # Running through the model
                    outs = gridtorchmodel(ego_vel, init_conds, hebb)

                    # Collecting different parts of the output
                    logits_hd, logits_pc, bottleneck_acts, rnn_states, rnn_cells = outs

                    # Computing test loss
                    pc_loss = softmax_crossentropy_with_logits(labels=ensemble_targets[0], logits=logits_pc)
                    hd_loss = softmax_crossentropy_with_logits(labels=ensemble_targets[1], logits=logits_hd)

                    total_loss = pc_loss + hd_loss
                    test_loss = total_loss.mean()

                    # weight decay
                    test_loss += gridtorchmodel.l2_loss * WEIGHT_DECAY

                    losses.append(test_loss.clone().item())

                    # accumulating for plotting
                    activations.append(torch.swapaxes(bottleneck_acts.detach(), 0, 1))
                    target_posxy.append(target_pos.detach())
                    pred_posxy.append(torch.swapaxes(logits_pc.detach(), 0, 1))

                    if eval_steps >= EVAL_STEPS:
                        break
                    eval_steps += 1

            # Logging and plotting evaluation results
            log_evaluations(losses, activations, target_posxy, pred_posxy)

            # Checkpointing
            torch.save({
                'epoch': epoch,
                'model_state_dict': gridtorchmodel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses[-1],
                'epoch_losses': epoch_losses,
                'test_losses': test_losses,
                'epoch_pc_losses': epoch_pc_losses,
                'epoch_hd_losses': epoch_hd_losses
            }, CHECKPOINT_PATH+f'model_{epoch:04d}.pt')

            # Saving Log files
            save_log_files()
