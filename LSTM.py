pip install numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 数据读取
data = pd.read_excel(r"C:\Users\lenovo\Desktop\大创\数据\data.xlsx")

# 将日期列转换为时间格式
data['日期'] = pd.to_datetime(data['日期'])

# 选择需要的特征列
features = ['收盘(点)', '涨跌(点)', '涨跌幅（%）', '市盈率（倍）']
data = data[features]

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 构建时间序列数据
def create_dataset(dataset, look_back=1):
    x, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        x.append(a)
        y.append(dataset[i + look_back])
    return np.array(x), np.array(y)

look_back = 10
x, y = create_dataset(data_scaled)

# 数据集划分
train_size = int(len(x) * 0.8)
x_train, x_test = x[0:train_size], x[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]

# 调整数据形状
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, len(features))))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(len(features)))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# 模型预测
y_pred = model.predict(x_test)

# 反归一化
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# 绘制预测结果
plt.plot(y_test[:, 0], label='True Values')
plt.plot(y_pred[:, 0], label='Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Closing Price')
plt.title('Stock Price Prediction')
plt.legend()
plt.show()