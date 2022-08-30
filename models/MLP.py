'''
@Author: Yitao Qiu
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_classes = configs.num_classes
        self.pred_len = configs.pred_len

        self.fc1 = nn.Linear(configs.dec_in * configs.pred_len, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.num_classes)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        batch_size,_,dec_in = x.shape
        x = x.view(batch_size,self.pred_len*dec_in)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        output = torch.softmax(x, dim=1)

        return output