'''
@Author: Yitao Qiu
Inspired by: https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_classes = configs.num_classes

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.LSTM_Seasonal = nn.LSTM(input_size = self.seq_len,hidden_size = configs.pred_len, num_layers=1,batch_first =True)
        self.LSTM_Trend = nn.LSTM(input_size = self.seq_len,hidden_size = configs.pred_len, num_layers=1,batch_first =True)

        self.LSTM_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.LSTM_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

        self.fc1 = nn.Linear(configs.dec_in * configs.pred_len, self.num_classes)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        
        seasonal_output, _ = self.LSTM_Seasonal(seasonal_init)
        trend_output, _ = self.LSTM_Trend(trend_init)

        x = seasonal_output + trend_output

        batch_size,dec_in,_ = x.shape
        
        x = x.reshape(batch_size,self.pred_len*dec_in)

        x = self.fc1(x)
        output = torch.softmax(x, dim=1)

        return  output