'''
@Author: Yitao Qiu
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    #def __init__(self, input_size,hidden_size,seq_len,pred_len,output_size,num_layers):
    def __init__(self, configs):
        super().__init__()
        self.input_size = configs.enc_in
        self.hidden_size = configs.hidden_size
        self.seq_len = configs.seq_len
        self.output_size = configs.c_out
        self.pred_len = configs.pred_len
        self.num_layers = configs.num_layers
        self.hidden_size = configs.hidden_size
        self.num_classes = configs.num_classes
        # LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(self.hidden_size*self.seq_len,self.pred_len*self.output_size)
        self.batch_first = True
        self.fc1 = nn.Linear(configs.c_out * configs.pred_len, self.num_classes)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        #print(x.shape)
        x_enc,_ = self.lstm(x_enc)
        batch_size, win_size, hidden_size = x_enc.shape
        x_enc = x_enc.view(batch_size, win_size*hidden_size)
        output = self.linear(x_enc)
        output = output.view(batch_size,self.pred_len,self.output_size)

        batch_size,_,c_out = output.shape
        output = output.view(batch_size,self.pred_len*c_out)

        output = self.fc1(output)
        output = torch.softmax(output, dim=1)

        return output