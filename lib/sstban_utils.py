import logging
import numpy as np
import pandas as pd
import os
import pickle
import scipy.sparse as sp
import sys
import torch
from scipy.sparse import linalg

from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from lib.metrics import masked_mape_np
import torch.nn.functional as F
import random


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seq2instance(data, num_his, num_pred):
    num_step, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = torch.zeros(num_sample, num_his, dims)
    y = torch.zeros(num_sample, num_pred, dims)
    for i in range(num_sample):
        x[i] = data[i: i + num_his]
        y[i] = data[i + num_his: i + num_his + num_pred]
    return x, y
def load_data(args):
	data_config = args['Data']
	training_config = args['Training']
	# Traffic
	df = pd.read_hdf(data_config['traffic_file'])
	traffic = torch.from_numpy(df.values)
	# train/val/test
	num_step = df.shape[0]
	train_steps = round(float(data_config['train_ratio']) * num_step)
	test_steps = round(float(data_config['test_ratio']) * num_step)
	val_steps = num_step - train_steps - test_steps
	print('traffic shape', traffic.shape)
	train = traffic[: train_steps]
	val = traffic[train_steps: train_steps + val_steps]
	test = traffic[-test_steps:]
	# X, Y
	num_his = int(training_config['num_his'])
	num_pred = int(training_config['num_pred'])
	trainX, trainY = seq2instance(train, num_his, num_pred)
	valX, valY = seq2instance(val, num_his, num_pred)
	testX, testY = seq2instance(test, num_his, num_pred)
	trainX=trainX.unsqueeze(-1)
	trainY = trainY.unsqueeze(-1)
	valX = valX.unsqueeze(-1)
	valY = valY.unsqueeze(-1)
	testX = testX.unsqueeze(-1)
	testY = testY.unsqueeze(-1)
	# normalization
	mean, std = torch.mean(trainX), torch.std(trainX)
	mean=mean.unsqueeze(0)
	std=std.unsqueeze(0)
	trainX = (trainX - mean) / std
	valX = (valX - mean) / std
	testX = (testX - mean) / std

	# temporal embedding
	time = pd.DatetimeIndex(df.index)
	dayofweek = torch.reshape(torch.tensor(time.weekday), (-1, 1))
	timeofday = (time.hour * 3600 + time.minute * 60 + time.second) \
				// (5 * 60)
	timeofday = torch.reshape(torch.tensor(timeofday), (-1, 1))
	time = torch.cat((dayofweek, timeofday), -1)
	# train/val/test
	train = time[: train_steps]
	val = time[train_steps: train_steps + val_steps]
	test = time[-test_steps:]
	# shape = (num_sample, num_his + num_pred, 2)
	trainTE = seq2instance(train, num_his, num_pred)
	trainTE = torch.cat(trainTE, 1).type(torch.int32)
	valTE = seq2instance(val, num_his, num_pred)
	valTE = torch.cat(valTE, 1).type(torch.int32)
	testTE = seq2instance(test, num_his, num_pred)
	testTE = torch.cat(testTE, 1).type(torch.int32)

	return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
			mean, std)

def mae_rmse_mape(y_pred, y_true):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    loss = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = masked_mape_np(y_true, y_pred, null_val=0)
    return loss, rmse, mape

