import torch
from torch.utils.data import Dataset, DataLoader
from config import FILE_NAME
import matplotlib.pyplot as plt

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.data = data
        self.seq_length = seq_length
        self.pred_length = pred_length

    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx+self.seq_length]
        target_seq = self.data[idx+self.seq_length:idx+self.seq_length+self.pred_length]
        return torch.tensor(input_seq), torch.tensor(target_seq)

def plot_losses(train_losses, valid_losses):
    # 绘制训练损失和验证损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='train_loss', color='blue')
    plt.plot(valid_losses, label='test_loss', color='orange')

    # 添加标题和标签
    plt.title('train_test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')

    # 添加图例
    plt.legend()

    # 显示网格线
    plt.grid(True)

    # 显示图像
    plt.savefig(f'./result/{FILE_NAME}/loss.png')
    plt.show()


'''
    使用Subset类将前 train_size 个数据点分配给训练集，
    剩余的数据点分配给测试集。这样做可以保持数据集的原始顺序，不会进行随机打乱
'''
from torch.utils.data import Dataset, Subset

def prepare_data(data, seq_length, pred_length, batch_size, shuffle=False):
    dataset = TimeSeriesDataset(data, seq_length, pred_length)
    # 计算训练集、验证集和测试集的大小
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # 分配数据集为训练集、验证集和测试集
    train_dataset = Subset(dataset, list(range(train_size)))  # 使用前70%的数据点作为训练集
    val_dataset = Subset(dataset, list(range(train_size, train_size + val_size)))  # 使用接下来的20%的数据点作为验证集
    test_dataset = Subset(dataset, list(range(train_size + val_size, len(dataset))))  # 使用剩余的10%的数据点作为测试集

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, optimizer, loss_func, epochs, device):
    # 用于存储训练和验证损失的列表
    t_losses, v_losses = [], []

    print(f'device: {device}')

    # 循环迭代训练轮次
    for epoch in range(epochs):
        train_loss, valid_loss = 0.0, 0.0
        
        # 训练步骤
        model.train()
        
        # 循环迭代训练数据集
        for x, y in train_loader:
            optimizer.zero_grad()
            
            # 将输入移动到设备上
            x = x.to(device)
            y = y.squeeze().to(device)
            
            # 前向传播
            preds = model(x).squeeze()
            loss = loss_func(preds, y)  # 计算批次损失
            train_loss += loss.item()
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()

        epoch_loss = train_loss / len(train_loader)
        t_losses.append(epoch_loss)
        
        # 验证步骤
        model.eval()
        
        # 循环迭代验证数据集
        for x, y in val_loader:
            with torch.no_grad():
                x, y = x.to(device), y.squeeze().to(device)
                preds = model(x).squeeze()
                error = loss_func(preds, y)
            valid_loss += error.item()
        
        valid_loss = valid_loss / len(val_loader)
        v_losses.append(valid_loss)
        
        if epoch % 10 == 0:
            print(f'{epoch} - 训练损失: {epoch_loss}, 验证损失: {valid_loss}')
    
    # 绘制图       
    plot_losses(t_losses, v_losses)

def valid(model, test_loader, device):
    model.eval()

    import numpy as np
    import pandas as pd
    predictions, pred_value = [], []
    with torch.no_grad():
        for data in test_loader:
            inputs = data[0].to(device)  # 获取输入数据
            outputs = model(inputs)  # 进行预测
            predictions.append(outputs.cpu().numpy())  # 将预测结果添加到列表中
    predictions = np.concatenate(predictions, axis=0)  # 将多个批次的预测结果合并成一个数组

    for data in predictions:
        pred_value.append(data.mean())

    data = pd.read_csv('./data_normed.txt').values.tolist()
    real_value = data[7379:]

    import matplotlib.pyplot as plt

    N = real_value.__len__()

    pred_value_pic1 = pred_value[:N]
    real_value_pic1 = real_value[:N]

    plt.figure(figsize=(20, 10))

    # 绘制真实值的折线图
    plt.plot(range(len(real_value_pic1)), real_value_pic1, label='True', c='cyan', linewidth=2)

    # 绘制预测值的折线图
    plt.plot(range(len(pred_value_pic1)), pred_value_pic1, label='Predicted', c='y', linewidth=2)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'./result/{FILE_NAME}/test_predict.png')
    plt.show()
#---------------------------------------------------------------------------------------------------------------
    # 更直观版 [从测试集的地方开始显示曲线，更好的看预测效果]

    # 真实数据
    real_data = pd.read_csv('./data_normed.txt').values.tolist()
    pre_data = pd.read_csv('./data_normed.txt').values.tolist()
    N = len(test_loader.dataset)
    Total = len(real_data)

    for i in range(N):
        # 从下标为：Total-1-N的位置开始改为预测值
        # print(pre_data[7378+i] == pred_value[i])
        pre_data[Total-1-N  +  i][0] = pred_value[i]

    import seaborn as sns

    plt.figure(figsize=(20, 6))

    sns.set_palette("Set1")

    M = int(Total * 0.87)  # 从M个时间点开始显示
    real_data_p2 = real_data[M:]
    pre_data_p2 = pre_data[M:]

    # ----------------------备份预测数据---------------------
    with open(f'./result/{FILE_NAME}/value(pre).txt', 'w') as f:
        for d in pre_data_p2:
            f.write(str(d)[1:-1] + "\n")
    with open(f'./result/{FILE_NAME}/value(real).txt', 'w') as f:
        for d in real_data_p2:
            f.write(str(d)[1:-1] + "\n")
    print('备份数据完毕！')
    # ----------------------备份预测数据---------------------

    # 绘制预测值的折线图
    plt.plot(range(len(pre_data_p2)), pre_data_p2, label='Predicted', linewidth=2)

    # 绘制真实值的折线图
    plt.plot(range(len(real_data_p2)), real_data_p2, label='True', linewidth=2)

    plt.title('Start with the test set section')
    plt.suptitle('Seismic acceleration time series prediction', fontsize=20)

    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()

    plt.savefig(f'./result/{FILE_NAME}/model_result.png')
    plt.show()
