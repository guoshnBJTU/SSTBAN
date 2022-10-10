import pandas as pd
import numpy as np
import os
import argparse
import configparser
import warnings
import torch
from copy import deepcopy
import time
import torch.utils.data
import torch.optim as optim
from model.sstban_model import SSTBAN, make_model
import time
import datetime
import math
import torch.nn as nn
import nni
from lib import sstban_utils


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMSD4_1dim_24.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
print('>>>>>>>  configuration   <<<<<<<')
with open(args.config, 'r') as f:
    print(f.read())
print('\n')
config.read(args.config)

data_config = config['Data']
training_config = config['Training']

# Data config
if config.has_option('Data', 'graph_signal_matrix_filename'):
    graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
else:
    graph_signal_matrix_filename = None

dataset_name = data_config['dataset_name']
print("dataset_name: ", dataset_name)
num_of_vertices = int(data_config['num_of_vertices'])
time_slice_size = int(data_config['time_slice_size'])

# nni
use_nni = int(training_config['use_nni'])
mode = training_config['mode']
ctx = training_config['ctx']
if use_nni:
    import nni
    params = nni.get_next_parameter()
    L = int(params['L'])
    training_config['L'] = str(L)
    K = int(params['K'])
    training_config['K'] = str(K)
    d = int(params['d'])
    training_config['d'] = str(d)
    miss_rate = float(params['node_miss_rate'])
    training_config['node_miss_rate']=str(miss_rate)
    T_miss_len=int(params['T_miss_len'])
    training_config['T_miss_len']=str(T_miss_len)
    self_weight_dis = float(params['self_weight_dis'])
    training_config['self_weight_dis'] = str(self_weight_dis)
else:
    L = int(training_config['L'])
    K = int(training_config['K'])
    d = int(training_config['d'])
    self_weight_dis = float(training_config['self_weight_dis'])
    reference = int(training_config['reference'])

# Training config
learning_rate = float(training_config['learning_rate'])
max_epoch = int(training_config['epochs'])
decay_epoch = int(training_config['decay_epoch'])
batch_size = int(training_config['batch_size'])
num_his = training_config['num_his']
num_pred = int(training_config['num_pred'])
patience = int(training_config['patience'])
in_channels = int(training_config['in_channels'])
# load dataset
if dataset_name == "PeMS_Bay":
    trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY, mean_, std_ = sstban_utils.load_data(config)
else :
    data = np.load(graph_signal_matrix_filename)
    trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY = data['train_x'], data['trainTE'], data['train_target'], data['val_x'], data['valTE'], data['val_target'], data['test_x'], data['testTE'], data['test_target']
    print("train: ", trainX.shape, trainY.shape)
    print("val: ", valX.shape, valY.shape)
    print("test: ", testX.shape, testY.shape)
    print("trainTE: ", trainTE.shape)
    print("valTE: ", valTE.shape)
    print("testTE: ", testTE.shape)
    trainX = torch.from_numpy(np.array(trainX, dtype='float64')).type(torch.FloatTensor)
    trainY = torch.from_numpy(np.array(trainY, dtype='float64')).type(torch.FloatTensor)
    valX = torch.from_numpy(np.array(valX, dtype='float64')).type(torch.FloatTensor)
    valY = torch.from_numpy(np.array(valY, dtype='float64')).type(torch.FloatTensor)
    testX = torch.from_numpy(np.array(testX, dtype='float64')).type(torch.FloatTensor)
    testY = torch.from_numpy(np.array(testY, dtype='float64')).type(torch.FloatTensor)
    trainTE = torch.from_numpy(np.array(trainTE, dtype='int32'))
    valTE = torch.from_numpy(np.array(valTE, dtype='int32'))
    testTE = torch.from_numpy(np.array(testTE, dtype='int32'))
    mean_, std_ = torch.mean(trainX.reshape(-1, in_channels), axis=0), torch.std(trainX.reshape(-1, in_channels), axis=0)  # reshape axis 速度、占有率、流量
    print("mean and std in every feature",mean_.shape, mean_, std_)
    trainX = (trainX - mean_) / std_
    valX = (valX - mean_) / std_
    testX = (testX - mean_) / std_

print("train: ", trainX.shape, trainY.shape)
print("val: ", valX.shape, valY.shape)
print("test: ", testX.shape, testY.shape)
print("trainTE: ", trainTE.shape)
print("valTE: ", valTE.shape)
print("testTE: ", testTE.shape)
#select device
gpu = int(training_config['gpu'])
if gpu:
    USE_CUDA=torch.cuda.is_available()
    if USE_CUDA:
        print("CUDA:", USE_CUDA, ctx)
        torch.cuda.set_device(int(ctx))
        device=torch.device("cuda")
    else:
        print("NO CUDA,Let's use cpu!")
        device = torch.device("cpu")
else:
    device=torch.device("cpu")
    print("Use CPU")
model = make_model(config, bn_decay=0.1)
model = model.to(device)
mean_ = mean_.to(device)
std_ = std_.to(device)
parameters = sstban_utils.count_parameters(model)
print('trainable parameters: {:,}'.format(parameters))

# train
print('Start training ...')

val_loss_min = float('inf')
wait = 0
best_model_wts = None
best_model = deepcopy(model.state_dict())
best_epoch = -1
best_loss = np.inf

num_train = trainX.shape[0]
num_val = valX.shape[0]
num_test = testX.shape[0]
train_num_batch = math.ceil(num_train / batch_size)
val_num_batch = math.ceil(num_val / batch_size)
test_num_batch = math.ceil(num_test / batch_size)
loss_criterion = nn.L1Loss()
loss_criterion_self = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=decay_epoch,
                                      gamma=0.9)

if use_nni:
    exp_id = nni.get_experiment_id()
    trail_id = nni.get_trial_id()
    dataset_name = dataset_name + '_' + str(exp_id) + '_' + str(trail_id)
exp_datadir="experiments/SSTBAN/"
if not os.path.exists(exp_datadir):
    os.makedirs(exp_datadir)
params_filename = os.path.join(exp_datadir, f"{dataset_name}_{num_of_vertices}_{num_his}_{num_pred}_{time_slice_size}_{K}_{L}_{d}_best_params")
train_time_epochs = []
val_time_epochs=[]
total_start_time = time.time()
for epoch_num in range(0, max_epoch):
    if wait >= patience:
        print(f'early stop at epoch: {epoch_num}, the val loss is {val_loss_min}')
        break
    # shuffle
    permutation = torch.randperm(num_train)
    trainX = trainX[permutation]
    trainTE = trainTE[permutation]
    trainY = trainY[permutation]
    start_train = time.time()
    model.train()
    train_loss = 0

    print(f"epoch {epoch_num} start!")
    for batch_idx in range(train_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_train, (batch_idx + 1) * batch_size)
        X = trainX[start_idx: end_idx].to(device)
        TE = trainTE[start_idx: end_idx].to(device)
        label = trainY[start_idx: end_idx].to(device)
        optimizer.zero_grad()
        pred,complete_X_enc,X_miss = model(X, TE,mode)
        # self_label=X[...,0]
        # self_label=self_label*std_[0]+mean_[0]
        # self_pred=self_pred*std_[0]+mean_[0]
        pred = pred * std_[0] + mean_[0]
        # print("pred:", pred.shape, "label: ", label.shape)
        loss_self=loss_criterion_self(complete_X_enc,X_miss)
        loss_batch = loss_criterion(pred, label)
        train_loss += float(loss_batch) * (end_idx - start_idx)
        loss_all=(1-self_weight_dis)*loss_batch+self_weight_dis*loss_self
        loss_all.backward()
        optimizer.step()

        if (batch_idx+1) % 10 == 0:
            print(f'Training batch: {batch_idx + 1} in epoch:{epoch_num}, training batch loss:{loss_batch:.4f}')
        del X, TE, label, pred, loss_batch
    train_loss /= num_train
    end_train = time.time()

    print("evaluating on valid set now!")
    val_loss = 0
    start_val = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * batch_size
            end_idx = min(num_val, (batch_idx + 1) * batch_size)
            X = valX[start_idx: end_idx].to(device)
            TE = valTE[start_idx: end_idx].to(device)
            label = valY[start_idx: end_idx].to(device)
            pred,self_pred= model(X, TE,'test')
            pred = pred * std_[0] + mean_[0]
            loss_batch = loss_criterion(pred, label)
            val_loss += loss_batch * (end_idx - start_idx)
            del X, TE, label, pred, loss_batch
    val_loss /= num_val
    end_val = time.time()

    if use_nni:
        nni.report_intermediate_result(val_loss.item())

    train_time_epochs.append(end_train - start_train)
    val_time_epochs.append(end_val - start_val)
    print('%s | epoch: %04d/%d, training time: %.1fs, validation time: %.1fs' %
        (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch_num + 1,
         max_epoch, end_train - start_train, end_val - start_val))
    print(f'train loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')
    if val_loss <= val_loss_min:
        wait = 0
        val_loss_min = val_loss
        best_model = deepcopy(model.state_dict())
        best_epoch = epoch_num
    else:
        wait += 1
    scheduler.step()
params_filename=params_filename+"_"+str(val_loss_min.cpu().numpy())
#torch.save(best_model, params_filename)
print(f"saving model to {params_filename}")


print("train one epoch for average: ", np.array(train_time_epochs).mean())
print("valid one epoch for average: ", np.array(val_time_epochs).mean())
print("train time: ", time.time() - total_start_time, "s")

# evaluation
print('evaluating on all set now!')
print('paramfile:',params_filename)
model.load_state_dict(torch.load(params_filename))
model.eval()

trainY = trainY.cpu()
valY = valY.cpu()
testY = testY.cpu()
mean_ = mean_.cpu().numpy()
std_ = std_.cpu().numpy()

with torch.no_grad():

    trainPred = []
    for batch_idx in range(train_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_train, (batch_idx + 1) * batch_size)
        X = trainX[start_idx: end_idx].to(device)
        TE = trainTE[start_idx: end_idx].to(device)
        pred_batch,self_pred = model(X, TE,'test')
        trainPred.append(pred_batch.detach().cpu().numpy())
        del X, TE, pred_batch
    trainPred = torch.from_numpy(np.concatenate(trainPred, axis=0))
    trainPred = trainPred * std_[0] + mean_[0]

    valPred = []
    for batch_idx in range(val_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_val, (batch_idx + 1) * batch_size)
        X = valX[start_idx: end_idx].to(device)
        TE = valTE[start_idx: end_idx].to(device)
        pred_batch,self_pred = model(X, TE,'test')
        valPred.append(pred_batch.detach().cpu().numpy())
        del X, TE, pred_batch
    valPred = torch.from_numpy(np.concatenate(valPred, axis=0))
    valPred = valPred * std_[0] + mean_[0]

    testPred = []
    start_test = time.time()
    for batch_idx in range(test_num_batch):
        start_idx = batch_idx * batch_size
        end_idx = min(num_test, (batch_idx + 1) * batch_size)
        X = testX[start_idx: end_idx].to(device)
        TE = testTE[start_idx: end_idx].to(device)
        pred_batch,self_pred = model(X, TE,'test')
        testPred.append(pred_batch.detach().cpu().numpy())
        del X, TE, pred_batch
    testPred = torch.from_numpy(np.concatenate(testPred, axis=0))
    testPred = testPred * std_[0] + mean_[0]
end_test = time.time()
train_mae, train_rmse, train_mape = sstban_utils.mae_rmse_mape(trainPred, trainY)
val_mae, val_rmse, val_mape = sstban_utils.mae_rmse_mape(valPred, valY)
test_mae, test_rmse, test_mape = sstban_utils.mae_rmse_mape(testPred, testY)


if use_nni:
    nni.report_final_result(test_mae)
print('testing time: %.1fs' % (end_test - start_test))


print('             LOSS\tMAE\tRMSE\tMAPE')
print('train      %.2f\t%.2f\t%.2f\t%.2f%%' %
        (train_mae, train_mae, train_rmse, train_mape))
print('val        %.2f\t%.2f\t%.2f\t%.2f%%' %
        (val_mae, val_mae, val_rmse, val_mape))
print('test       %.2f\t%.2f\t%.2f\t%.2f%%' %
        (test_mae , test_mae, test_rmse, test_mape))
print('performance in each prediction step')


columns = ['loss', 'mae', 'rmse', 'mape']
index = ['train', 'test', 'val']

values = [[train_mae, train_mae, train_rmse, train_mape],
        [val_mae, val_mae, val_rmse, val_mape],
        [test_mae, test_mae, test_rmse, test_mape]]
for i in range(len(values)):
    for j in range(len(values[0])):
        values[i][j] = round(values[i][j], 4)

MAE, RMSE, MAPE = [], [], []
values = []
for step in range(num_pred):
    mae, rmse, mape = sstban_utils.mae_rmse_mape(testPred[:, step], testY[:, step])
    MAE.append(mae)
    RMSE.append(rmse)
    MAPE.append(mape)
    values.append([mae, rmse, mape])
    print('step: %02d         %.2f\t%.2f\t%.2f%%' %
                   (step + 1, mae, rmse, mape))
average_mae = np.mean(MAE)
average_rmse = np.mean(RMSE)
average_mape = np.mean(MAPE)
print('average:         %.2f\t%.2f\t%.2f%%' %
             (average_mae, average_rmse, average_mape))