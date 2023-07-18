import numpy as np
import time
from getCSI import ReadCSIData
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import threading
from data_load import dataset_construct
import pandas as pd

#线程结束的标志位
end1 = 0
end2 = 0
end3 = 0
##存放获取到的数据
data_8081 = []
data_8082 = []
data_8083 = []


def th1():
    print("th1 start time:",time.time())
    global data_8081
    data_8081 = a.CSIMatrix()
    dd1 = pd.DataFrame(data_8081)
    dd1.to_csv("./online_8081.csv",index=False,header=False,mode='a')
    global end1
    print("th1 end time:", time.time())
    end1 = 1

def th2():
    print("th2 start time:",time.time())
    global data_8082
    data_8082 = b.CSIMatrix()
    ee1 = pd.DataFrame(data_8082)
    ee1.to_csv("./online_8082.csv",index=False,header=False,mode='a')
    global end2
    print("th2 end time:", time.time())
    end2 = 1
#
def th3():
    print("th3 start time:",time.time())
    global data_8083
    data_8083 = c.CSIMatrix()
    ff1 = pd.DataFrame(data_8083)
    ff1.to_csv("./online_8083.csv",index=False,header=False,mode='a')
    global end3
    print("th3 end time:", time.time())
    end3 = 1




model = models.load_model("model/model.h5")  ##加载训练好的模型
# model.summary()

# a = ReadCSIData(m=64, n=100, period=0.02, port=8081, deviceName='8081 ')
b = ReadCSIData(m=64, n=100, period=0.02, port=8081, deviceName='8082 ')
#c = ReadCSIData(m=64, n=100, period=0.02, port=8083, deviceName='8083 ')

i = 0
while True:
   # _th1 = threading.Thread(target=th1)
   _th2 = threading.Thread(target=th2)
   # _th3 = threading.Thread(target=th3)
   # _th1.start()
   _th2.start()
   # _th3.start()
   #while end1&end2&end3 == 0:
   while end2 == 0:
     pass
   end2 = 0
   # i = i + 1
   # print(i)
   dataset = np.array(data_8082[:,1:])
   print(dataset.shape)
   dataset = dataset.reshape(1,100,64,1)
   result = model.predict(dataset)
   print(result)
   # end2 = 0
   # end1 = 0
   # print("start process data")
   # #data_list = [data_8081,data_8082,data_8083]
   # data_list = [data_8083]
   # Data, _ = dataset_construct(data_list,100,1)
   # print(Data.shape)
   # result = model.predict(Data)
   # print(result)
   # result = np.mat(result)
   # gg = pd.DataFrame(result)
   # gg.to_csv("./online_result.csv",header=False,index=False,mode='a')
   #print("end process data")
