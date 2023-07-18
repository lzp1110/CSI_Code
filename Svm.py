import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_load import dataset_load,data_elm_zero
from matplotlib import pyplot as plt
from filter import Filter
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

youren_dir = './youren'
wuren_dir = './wuren'
train_data, train_label, test_data, test_label = dataset_load(youren_dir, wuren_dir)
train_data = train_data.reshape(448,100,52)
test_data = test_data.reshape(192,100,52)

pca_train = np.zeros((420,100,2))
pca_test = np.zeros((180,100,2))


temp = train_data[40,:,26]
temp_hample_filter = Filter.hample(temp)
temp_bt_filter = Filter.high_pass_filter(temp_hample_filter)
plt.plot(temp,label = "init data",color= 'black',linestyle='-')
plt.plot(temp_hample_filter,label = "hample filter data",color= 'black',linestyle='--')
plt.plot(temp_bt_filter,label = "highpass filter data",color= 'black',linestyle=':')
plt.ylabel("Amplitude")
plt.xlabel("time/s")
xticks = np.arange(0,125,25)
plt.xticks(xticks,["0","0.5","1","1.5","2"])
# plt.ylim(6,12)
plt.legend()
plt.show()


#对数据集进行PCA
for i in range(len(train_data)):
    temp = data_elm_zero(train_data[i])
    # for j in range(temp.shape[1]):
    #     temp[:,j] = Filter.hample(temp[:,j]) ##hample滤波
    temp = Filter.pca(temp,2)
    pca_train[i] = temp

for i in range(len(test_data)):
    temp = data_elm_zero(test_data[i])
    # for j in range(temp.shape[1]):
    #     temp[:,j] = Filter.hample(temp[:,j]) ##hample滤波
    temp = Filter.pca(temp,2)
    pca_test[i] = temp

#将pca后的数据输入SVM
clf = SVC(kernel='linear', C=1, random_state=0)
clf.fit(pca_train.reshape(-1, 200), train_label)

pca_pred = clf.predict(pca_test.reshape(-1, 200))
print(pca_pred)
acc = accuracy_score(test_label, pca_pred)
print("Accuracy:", acc)



##将经PCA处理后的数据输入CNN

# input_shape = (100,2)
#
# print("start create model")
#
# model = models.Sequential()
# model.add(layers.Conv1D(32, 1, activation='relu', input_shape=input_shape))
# model.add(layers.MaxPooling1D(2))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv1D(64, 1, activation='relu'))
# model.add(layers.MaxPooling1D(2))
# model.add(layers.Dropout(0.2))
# model.add(layers.Conv1D(64, 1, activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dropout(0.2))
# model.add(layers.Dense(2,activation='softmax'))
#
# model.compile(optimizer='adam',
#                 loss=tf.losses.sparse_categorical_crossentropy,
#                 metrics=['accuracy'])
# #训练模型
# history = model.fit(pca_train, train_label, batch_size=32, epochs=50)
# print("start evaluate")
#
# result = model.predict(pca_test)
#     ##保存输出的结果
# result = np.mat(result)
# result = pd.DataFrame(result)
# result.to_csv("./cnn+pca_result.csv",index=False,header=False)