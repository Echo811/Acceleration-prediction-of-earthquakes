import pandas as pd
import torch
import torch.nn as nn
from exp import train, prepare_data, valid
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import USE_CUDA

from config import seq_length, batch_size, pred_length, nhid, n_dnn_layers, n_features, device, lr, epochs, FILE_NAME

class LSTMForecaster(nn.Module):

    def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, use_cuda=False, dropout=0.2):
        '''
        n_features: 输入特征的数量（对于单变量预测，为1）
        n_hidden: 每个隐藏层中的神经元数量
        n_outputs: 每个训练样本要预测的输出数量
        n_deep_layers: LSTM 层后的隐藏稠密层数量
        sequence_len: 用于预测的步长
        dropout: 浮点数 (0 < dropout < 1)，表示隐藏层之间的 dropout 比率
        '''
        super().__init__()
        
        self.n_lstm_layers = n_lstm_layers
        self.nhid = n_hidden
        self.use_cuda = use_cuda # 设置是否使用 CUDA 加速
        
        # LSTM 层
        self.lstm = nn.LSTM(n_features,
                            n_hidden,
                            num_layers=n_lstm_layers,
                            batch_first=True) # 因为我们已经将数据转换成了这种形式
        
        # LSTM 层后的第一个稠密层
        self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden)
        
        # Dropout 层
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建隐藏稠密层（n_hidden x n_deep_layers）
        dnn_layers = []
        for i in range(n_deep_layers):
            # 最后一层 (n_hidden x n_outputs)
            if i == n_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(nhid, n_outputs))
                # 其他层 (n_hidden x n_hidden)，并可选择添加 dropout
            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(nhid, nhid))
                if dropout:
                    dnn_layers.append(nn.Dropout(p=dropout))
        # 编译隐藏稠密层
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):
        
        # 初始化隐藏状态
        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
        
        # 将隐藏状态移动到设备上
        if self.use_cuda:
            hidden_state = hidden_state.to(device)
            cell_state = cell_state.to(device)
        self.hidden = (hidden_state, cell_state)
        
        # 前向传播
        x, h = self.lstm(x, self.hidden) # LSTM 层
        x = self.dropout(x.contiguous().view(x.shape[0], -1)) # 将 LSTM 输出展平
        x = self.fc1(x) # 第一个稠密层
        return self.dnn(x) # 通过全连接的 DNN 进行进一步前向传播。

if not os.path.exists(f'./result/{FILE_NAME}'):
    os.makedirs(f'./result/{FILE_NAME}')

data = pd.read_csv('./data_normed.txt').values.tolist()
len(data)


train_loader, val_loader, test_loader= prepare_data(data, seq_length, pred_length, batch_size)

print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

# 初始化模型
model = LSTMForecaster(n_features, nhid, pred_length, seq_length, n_deep_layers=n_dnn_layers, use_cuda=USE_CUDA).to(device)

# Initialize the loss function and optimizer  
criterion = nn.MSELoss().to(device)  
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# train
train(model, train_loader, val_loader, optimizer, criterion, epochs, device)

# val
valid(model, test_loader, device)

