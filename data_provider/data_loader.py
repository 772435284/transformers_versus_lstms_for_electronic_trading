'''
data_loader.py
Based on: https://github.com/thuml/Autoformer/blob/main/data_provider/data_loader.py
'''
import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from joblib import dump, load
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self,all_data, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',label='label_5',product='btc'):
        
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val','backtest']
        type_map = {'train': 0, 'val': 1, 'test': 2,'backtest':2}
        self.set_type = type_map[flag]
        self.product = product
        self.features = features
        self.target = target
        self.label = label
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.all_data = all_data
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = self.all_data

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        label  = df_raw[self.label]
        label = label.values
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('label_1')
        cols.remove('label_2')
        cols.remove('label_3')
        cols.remove('label_4')
        cols.remove('label_5')
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = 5110211
        num_test = 2580162
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            dump(self.scaler, 'std_scaler.bin', compress=True)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.label = label[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        label_begin = r_begin + self.label_len
        label_end = r_begin + self.label_len + 1

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        label = self.label[label_begin:label_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,label

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    