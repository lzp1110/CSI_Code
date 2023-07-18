import os
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras as keras
from matplotlib import pyplot as plt
from CSI_Fingerprint_Rayleigh import Rayleigh_Fingerprint


def data_convert(data,num):
    DATA = data.split('\n')[:num]
    header = DATA[0].split(',')
    float_data = np.zeros((len(DATA),len(header)))
    for i,Data in enumerate(DATA):
        values = [float(x) for x in Data.split(',')]
        float_data[i][:] = values
    return float_data

def data_process(data):
    float_Data = np.zeros([30000,52])
    for i in range(2,28):
        for j in range(30000):
            float_Data[j,i-2] = data[j,i]

    for i in range(39,65):
        for j in range(30000):
            float_Data[j,i-13] = data[j,i]
    return float_Data

def dataset_construct(array1,array2,array3,data_num):
    list = [array1,array2,array3]
    time_window = int(array1.shape[0]/data_num)
    datasets = np.zeros((data_num,time_window,52,3))
    #300条数据
    for i in range(data_num):  #0-299
        for j in range(time_window): #0-99
            for k in range(52):     #0-51
                for n in range(3):
                    datasets[i][j][k][n] = list[n][i*100+j][k]

    return datasets


data_dir_jingzhi = '../Data_20221210/静止'
fname_jingzhi_8089 = os.path.join(data_dir_jingzhi,'jingzhi_8089.csv')
fname_jingzhi_8090 = os.path.join(data_dir_jingzhi,'jingzhi_8090.csv')
fname_jingzhi_8091 = os.path.join(data_dir_jingzhi,'jingzhi_8091.csv')

data_dir_yundong = '../Data_20221210/运动'
fname_yundong_8089 = os.path.join(data_dir_yundong,'yundong_8089.csv')
fname_yundong_8090 = os.path.join(data_dir_yundong,'yundong_8090.csv')
fname_yundong_8091 = os.path.join(data_dir_yundong,'yundong_8091.csv')

data_dir_wuren = '../Data_20221210/无人'
fname_wuren_8089 = os.path.join(data_dir_wuren,'wuren_8089.csv')
fname_wuren_8090 = os.path.join(data_dir_wuren,'wuren_8090.csv')
fname_wuren_8091 = os.path.join(data_dir_wuren,'wuren_8091.csv')

f = open(fname_yundong_8089)
data_y_8089 = f.read()
f.close()
f = open(fname_yundong_8090)
data_y_8090 = f.read()
f.close()
f = open(fname_yundong_8091)
data_y_8091 = f.read()
f.close()

f = open(fname_jingzhi_8089)
data_j_8089 = f.read()
f.close()
f = open(fname_jingzhi_8090)
data_j_8090 = f.read()
f.close()
f = open(fname_jingzhi_8091)
data_j_8091 = f.read()
f.close()

f = open(fname_wuren_8089)
data_w_8089 = f.read()
f.close()
f = open(fname_wuren_8090)
data_w_8090 = f.read()
f.close()
f = open(fname_wuren_8091)
data_w_8091 = f.read()
f.close()

data_j_8089 = data_convert(data_j_8089,30000)
data_j_8090 = data_convert(data_j_8090,30000)
data_j_8091 = data_convert(data_j_8091,30000)
data_y_8089 = data_convert(data_y_8089,30000)
data_y_8090 = data_convert(data_y_8090,30000)
data_y_8091 = data_convert(data_y_8091,30000)
data_w_8089 = data_convert(data_w_8089,30000)
data_w_8090 = data_convert(data_w_8090,30000)
data_w_8091 = data_convert(data_w_8091,30000)

pure_data_j_8089 = data_process(data_j_8089)
pure_data_j_8090 = data_process(data_j_8090)
pure_data_j_8091 = data_process(data_j_8091)
pure_data_y_8089 = data_process(data_y_8089)
pure_data_y_8090 = data_process(data_y_8090)
pure_data_y_8091 = data_process(data_y_8091)
pure_data_w_8089 = data_process(data_w_8089)
pure_data_w_8090 = data_process(data_w_8090)
pure_data_w_8091 = data_process(data_w_8091)

print("start preparing data")

#准备数据
#每种状态共30000个数据点，每100个数据点作为一条数据，共300条数据，200条作为训练集，共600条
#准备训练集
x_train = np.zeros((600, 100, 20))
for i in range(200):
    for j in range(100):
        for k in range(20):
            x_train[i][j][k] = data_w_8089[i*100+j][k+2]
for i in range(200,400):
    for j in range(100):
        for k in range(20):
            x_train[i][j][k] = data_j_8089[(i-200)*100+j][k+2]
for i in range(400,600):
    for j in range(100):
        for k in range(20):
            x_train[i][j][k] = data_y_8089[(i-400)*100+j][k+2]

y_train = np.zeros((600,3))
for i in range(200):
    y_train[i][0] = 1
for i in  range(200,400):
    y_train[i][1] = 1
for i in  range(400,600):
    y_train[i][2] = 1

#准备测试集
x_test = np.zeros((300, 100, 20))
for i in range(100):
    for j in range(100):
        for k in range(20):
            x_test[i][j][k] = data_w_8089[20000+i*100+j][k+2]
for i in range(100,200):
    for j in range(100):
        for k in range(20):
            x_test[i][j][k] = data_j_8089[20000+(i-100)*100+j][k+2]
for i in range(200,300):
    for j in range(100):
        for k in range(20):
            x_test[i][j][k] = data_y_8089[20000+(i-200)*100+j][k+2]

y_test = np.zeros((300,3))
for i in range(100):
    y_test[i][0] = 1
for i in  range(100,200):
    y_test[i][1] = 1
for i in  range(200,300):
    y_test[i][2] = 1

#开始训练模型
print("start create model")

model = keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(100,20)))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
model.summary()
#编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#训练模型
history = model.fit(x_train, y_train, batch_size=32, epochs=100)
print("start evaluate")
#评估模型
model.evaluate(x_test, y_test, batch_size=32)
#使用模型预测
print(model.predict(x_test))
print("end")


# 生成训练数据
# x = np.random.random((1000, 50, 20))
# y = np.random.randint(2, size=(1000,))

# # 定义输入层
# inputs = tf.keras.layers.Input(shape=(50, 20))
#
# # 定义全连接层1，有 512 个节点
# dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(inputs)
#
# # 定义全连接层2，有 3 个节点
# outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(dense1)
#
# # 创建模型
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
#
# # 编译模型
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()
# history = model.fit(x, y, batch_size=32, epochs=200)



# # 定义输入特征的维度
# input_dim = (100, 40)
#
# # 定义输出类别的数量
# output_dim = 3
#
# # 定义输入层
# inputs = tf.keras.layers.Input(shape=input_dim)
#
# # 定义隐藏层
# hidden = tf.keras.layers.Dense(64, activation='relu')(inputs)
#
# # 定义输出层
# output = tf.keras.layers.Dense(output_dim, activation='softmax')(hidden)
#
# # 创建模型
# model = tf.keras.Model(inputs=inputs, outputs=output)
#
# # 编译模型
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 打印模型的摘要信息
# model.summary()
#
# input_data = tf.random.uniform((10, 100, 40))
#
# # 使用模型预测输出
# output = model(input_data)
#
# # 打印输出的维度
# print(output.shape)