'''
exp_main.py
Based on: https://github.com/thuml/Autoformer/blob/main/exp/exp_main.py
'''
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, Reformer, LSTM, CNN_LSTM, MLP, FEDformer, DLSTM
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.backtestor import backtestor
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


warnings.filterwarnings('ignore')

eps = 1e-8
def sharpe(returns, freq=365, rfr=0):
    # The function that is used to caculate sharpe ratio
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)

def max_drawdown(return_list):
    # The function that is used to calculate the max drawndom
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
    if i == 0:
        return 0
    j = np.argmax(return_list[:i]) 
    return (return_list[j] - return_list[i]) / (return_list[j])


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.all_data = pd.read_csv(os.path.join(self.args.root_path,self.args.data_path))

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'FEDformer': FEDformer,
            'LSTM': LSTM,
            'CNN_LSTM': CNN_LSTM,
            'MLP': MLP,
            'DLSTM': DLSTM
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.all_data)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,label) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                label = torch.tensor(label,dtype=torch.long).to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                label = label.squeeze()
                pred = outputs.detach().cpu()
                true = label.detach().cpu()
                _, predictions = torch.max(pred, 1)
                all_targets.append(true.numpy())
                all_predictions.append(predictions.numpy())
                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        all_targets = np.concatenate(all_targets)    
        all_predictions = np.concatenate(all_predictions)
        print('accuracy_score:', accuracy_score(all_targets, all_predictions))
        print(classification_report(all_targets, all_predictions, digits=4))
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,label) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                label = torch.tensor(label,dtype=torch.long).to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    label = label.squeeze()
                    loss = criterion(outputs, label)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))
        time_now = time.time()
        test_steps = len(test_loader)
        all_targets = []
        all_predictions = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if os.path.exists(folder_path+'/result.csv'):
            self.backtest(setting, folder_path)
        else:
            self.model.eval()
            with torch.no_grad():
                iter_count = 0
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,label) in enumerate(test_loader):
                    iter_count += 1
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    label = torch.tensor(label,dtype=torch.long).to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    label = label.squeeze()
                    outputs = outputs.detach().cpu()
                    label = label.detach().cpu()
                    
                    pred = outputs  
                    true = label  
                    _, predictions = torch.max(pred, 1)
                    all_targets.append(true.numpy())
                    all_predictions.append(predictions.numpy())
                    if (i + 1) % 100 == 0:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print('\titers: {}, \tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(i+1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            
            all_targets = np.concatenate(all_targets)    
            all_predictions = np.concatenate(all_predictions)
            np.save(folder_path+'/targets.npy',all_targets)
            np.save(folder_path+'/predictions.npy',all_predictions)
            df_pred = pd.DataFrame(all_predictions)
            df_pred.to_csv(folder_path+'/result.csv',index=False)
            print('accuracy_score:', accuracy_score(all_targets, all_predictions))
            print(classification_report(all_targets, all_predictions, digits=4))
            self.backtest(setting, folder_path)
        

        return

    def backtest(self, setting,folder_path):
        
        if self.args.product == 'eth':
            test_data = self.all_data[len(self.all_data) - 2580162 - 100:len(self.all_data)]
        else:
            test_data = self.all_data[len(self.all_data) - 2590189 - 100:len(self.all_data)]
        test_data = test_data[self.args.pred_len:-(self.args.pred_len-1)]
        test_data.sort_values(by='date', inplace=True)
        test_data.set_index(keys='date', inplace=True)
        test_data.reset_index(inplace=True)
        mid_price = test_data['mid_price']
        mid_price = mid_price.values
        test_pred = pd.read_csv(folder_path+'/result.csv')
        test_pred = test_pred.to_numpy()
        bt = backtestor(mid_price,test_pred)
        bt.start_backtest()
        cpr = np.array(bt.captial_his)
        daily_culmulative_pnl= [bt.captial_his[860615]-1,bt.captial_his[860615+859559]-1,bt.captial_his[-1]-1]
        daily_culmulative_pnl = np.array(daily_culmulative_pnl)
        daily_pnl = []
        daily_pnl.append((daily_culmulative_pnl[0] -1)/1)
        for i in range(len(daily_culmulative_pnl)):
            if i == len(daily_culmulative_pnl)-1:
                break
            daily_pnl.append((daily_culmulative_pnl[i+1] -daily_culmulative_pnl[i])/daily_culmulative_pnl[i])
        daily_pnl = np.array(daily_pnl)
        print(daily_pnl)
        sr = sharpe(daily_pnl)
        mdd = max_drawdown(cpr)
        print("Sharpe ratio: ", sr)
        print("max drawdown", mdd*100)
        print("Total asset: ", cpr[-1])
        print("culmulative return: ", cpr[-1]-1)
        
        np.save(folder_path+'/cpr.npy',cpr)
