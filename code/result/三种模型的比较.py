import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('../informer-lstm.csv', encoding='utf-8')

# 设置索引范围
N = 100
M = 500

# 获取需要绘制的数据
real = df['real'][N:M]
informer = df['informer'][N:M]
lstm = df['lstm'][N:M]

import seaborn as sns

plt.figure(figsize=(20, 6))

sns.set_palette("Dark2")

plt.plot(range(len(informer)), informer, label='Informer', linewidth=2)
plt.plot(range(len(lstm)), lstm, label='Lstm', linewidth=2)
plt.plot(range(len(real)), real, label='Real', linewidth=4)

plt.title('Start with the test set section', fontsize=15)
plt.suptitle('Seismic acceleration time series prediction with three models', fontsize=20)

plt.xlabel('Time', fontsize=13)
plt.ylabel('Value,',  fontsize=13)
plt.legend()
plt.savefig('three models compare.png')

plt.show()