import torch

seq_length = 64  # 训练窗口
batch_size = 16
pred_length = 8  # 预测窗口
nhid = 96  # 隐藏层中节点的数量
n_dnn_layers = 8  # 隐藏稠密层数量
# 特征数量（因为这是单变量时间序列分析，所以将其设置为1。多变量分析将在未来加入）
n_features = 1 
# 设备选择 (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'
lr = 4e-4  
epochs = 20

FILE_NAME = f'seq_l={seq_length}_pre_l={pred_length}_epoch={epochs}'
