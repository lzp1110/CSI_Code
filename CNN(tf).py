import numpy as np
import pandas as pd
import threading
import time
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from CSI_Fingerprint_Rayleigh import Rayleigh_Fingerprint
from data_load import dataset_load
from getCSI import ReadCSIData

# train_data, train_label, test_data, test_lable = data_load_single()
if  __name__ == '__main__':
    #导入生数据
    youren_dir = './youren'
    wuren_dir = './wuren'
    train_data,train_label,test_data,test_label = dataset_load(youren_dir,wuren_dir)
    #print(test_data.shape)


    input_shape = (100,64,1)

    print("start create model")

    model = models.Sequential()
    model.add(layers.Conv2D(32, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, 3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2,activation='softmax'))

    model.summary()
    # 编译模型
    # # 多分类
    model.compile(optimizer='adam',
                   loss=tf.losses.sparse_categorical_crossentropy,
                   metrics=['accuracy'])
    #二分类
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])

    #训练模型
    history = model.fit(train_data, train_label, batch_size=32, epochs=50)
    print("start evaluate")
    #评估模型
    #model.evaluate(test_data)
    result = model.predict(test_data)
    ##保存输出的结果
    result = np.mat(result)
    result = pd.DataFrame(result)
    result.to_csv("./cnn_result.csv",index=False,header=False)
    model.save("model.h5")



# if __name__ == "__main__":
#     Fingerprint = Rayleigh_Fingerprint()
#     a = ReadCSIData(m=64, n=100, period=0.02, port=8081, deviceName='8081 ')
#     while True:
#         raw_Data = a.CSIMatrix()
#         CSIData = data_process(raw_Data)
#         feature_matrix = Fingerprint.get_attr_matrix(CSIData)
        #print(feature_matrix)













