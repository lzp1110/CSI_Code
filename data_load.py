import os
import numpy as np
import cv2
from filter import Filter
from sklearn.model_selection import train_test_split

def data_convert(data):
    DATA = data.split('\n')
    DATA = DATA[:len(DATA)-1] ##-1以去除最后的空行
    header = DATA[0].split(',')
    float_data = np.zeros((len(DATA),len(header)-1))
    for i,Data in enumerate(DATA):
        values = [float(x) for x in Data.split(',')]
        float_data[i][:] = values[1:]    ##去除时间戳
    return float_data



#去除值为0的子载波
def data_elm_zero(data):
    new_data = []
    column = data.shape[1]
    for i in range(column):
        #if all(num == 0 for num in data[:,i]):
        if np.mean(data[:, i]) < 1:
            pass
        else:
            new_data.append(data[:,i])
    new_data = np.array(new_data).T
    return new_data


def dataset_construct(data_list,time_window,test_percent,youren = True):
    dataset_length = int(len(data_list[0])/time_window)
    channel_number = data_list[0].shape[1]
    #print(channel_number)
    # print(dataset_length)
    # data_num = int(dataset_length*train_percent)
    # datasets_train = np.zeros((data_num,time_window,channel_number,len(data_list)))
    # datasets_test = np.zeros((dataset_length - data_num,time_window,channel_number,len(data_list)))
    #
    datasets = np.zeros((dataset_length,time_window,channel_number,len(data_list)))
    if youren:
      labels = np.ones(dataset_length,dtype=int)
    else:
      labels = np.zeros(dataset_length,dtype=int)

    #data_num条数据
    for i in range(dataset_length):
        for j in range(time_window): #0-99
            for k in range(channel_number):
                for n in range(len(data_list)):
                        datasets[i][j][k][n] = data_list[n][i*time_window+j, k]

    data_train, data_test, labels_train, labels_test =train_test_split(datasets, labels, test_size=test_percent,random_state=1)
    return data_train, data_test, labels_train, labels_test


# def dataset_construct_single(array,data_start,data_end):
#     time_window = int(array.shape[0]/300)
#     print(time_window)
#     data_num = data_end - data_start
#     datasets = np.zeros((data_num,time_window,52,1))
#     #data_num条数据
#     for i in range(data_start,data_end):
#         for j in range(time_window): #0-99
#             for k in range(52):     #0-51
#                     if i<200:
#                        datasets[i][j][k][0] = array[i*100+j][k]
#                     else:
#                        datasets[i-200][j][k][0] = array[i*100+j][k]
#
#     return datasets
#
#
#
# def data_load():
#     data_dir_jingzhi = '../Data_20221210/静止'
#     fname_jingzhi_8089 = os.path.join(data_dir_jingzhi, 'jingzhi_8089.csv')
#     fname_jingzhi_8090 = os.path.join(data_dir_jingzhi, 'jingzhi_8090.csv')
#     fname_jingzhi_8091 = os.path.join(data_dir_jingzhi, 'jingzhi_8091.csv')
#
#     data_dir_yundong = '../Data_20221210/运动'
#     fname_yundong_8089 = os.path.join(data_dir_yundong, 'yundong_8089.csv')
#     fname_yundong_8090 = os.path.join(data_dir_yundong, 'yundong_8090.csv')
#     fname_yundong_8091 = os.path.join(data_dir_yundong, 'yundong_8091.csv')
#
#     data_dir_wuren = '../Data_20221210/无人'
#     fname_wuren_8089 = os.path.join(data_dir_wuren, 'wuren_8089.csv')
#     fname_wuren_8090 = os.path.join(data_dir_wuren, 'wuren_8090.csv')
#     fname_wuren_8091 = os.path.join(data_dir_wuren, 'wuren_8091.csv')
#
#     f = open(fname_yundong_8089)
#     data_y_8089 = f.read()
#     f.close()
#     f = open(fname_yundong_8090)
#     data_y_8090 = f.read()
#     f.close()
#     f = open(fname_yundong_8091)
#     data_y_8091 = f.read()
#     f.close()
#
#     f = open(fname_jingzhi_8089)
#     data_j_8089 = f.read()
#     f.close()
#     f = open(fname_jingzhi_8090)
#     data_j_8090 = f.read()
#     f.close()
#     f = open(fname_jingzhi_8091)
#     data_j_8091 = f.read()
#     f.close()
#
#     f = open(fname_wuren_8089)
#     data_w_8089 = f.read()
#     f.close()
#     f = open(fname_wuren_8090)
#     data_w_8090 = f.read()
#     f.close()
#     f = open(fname_wuren_8091)
#     data_w_8091 = f.read()
#     f.close()
#
#     data_j_8089 = data_convert(data_j_8089, 30000)
#     data_j_8090 = data_convert(data_j_8090, 30000)
#     data_j_8091 = data_convert(data_j_8091, 30000)
#     data_y_8089 = data_convert(data_y_8089, 30000)
#     data_y_8090 = data_convert(data_y_8090, 30000)
#     data_y_8091 = data_convert(data_y_8091, 30000)
#     data_w_8089 = data_convert(data_w_8089, 30000)
#     data_w_8090 = data_convert(data_w_8090, 30000)
#     data_w_8091 = data_convert(data_w_8091, 30000)
#
#     pure_data_j_8089 = data_process(data_j_8089)
#     pure_data_j_8090 = data_process(data_j_8090)
#     pure_data_j_8091 = data_process(data_j_8091)
#     pure_data_y_8089 = data_process(data_y_8089)
#     pure_data_y_8090 = data_process(data_y_8090)
#     pure_data_y_8091 = data_process(data_y_8091)
#     pure_data_w_8089 = data_process(data_w_8089)
#     pure_data_w_8090 = data_process(data_w_8090)
#     pure_data_w_8091 = data_process(data_w_8091)
#
#     print("start preparing data")
#
#     # 准备数据
#     # 每种状态共30000个数据点，每100个数据点作为一条数据，共300条数据，200条作为训练集，共600条
#     # 准备训练集
#     dataset_w = dataset_construct(pure_data_w_8089, pure_data_w_8090, pure_data_w_8091, 0, 200)
#     dataset_j = dataset_construct(pure_data_j_8089, pure_data_j_8090, pure_data_j_8091, 0, 200)
#     dataset_y = dataset_construct(pure_data_y_8089, pure_data_y_8090, pure_data_y_8091, 0, 200)
#     train_data = np.vstack([dataset_w, dataset_j])
#     train_data = np.vstack([train_data, dataset_y])
#
#     train_label = np.zeros((600, 1))  # 整数编码
#     for i in range(200):
#         train_label[i][0] = 0
#     for i in range(200, 400):
#         train_label[i][0] = 1
#     for i in range(400, 600):
#         train_label[i][0] = 2
#
#     # 准备测试集
#     dataset_w = dataset_construct(pure_data_w_8089, pure_data_w_8090, pure_data_w_8091, 200, 300)
#     dataset_j = dataset_construct(pure_data_j_8089, pure_data_j_8090, pure_data_j_8091, 200, 300)
#     dataset_y = dataset_construct(pure_data_y_8089, pure_data_y_8090, pure_data_y_8091, 200, 300)
#     test_data = np.vstack([dataset_w, dataset_j])
#     test_data = np.vstack([test_data, dataset_y])
#
#     test_lable = np.zeros((300, 1))
#     for i in range(100):
#         test_lable[i][0] = 0
#     for i in range(100, 200):
#         test_lable[i][0] = 1
#     for i in range(200, 300):
#         test_lable[i][0] = 2
#
#     return train_data,train_label,test_data,test_lable
#
#
# def data_load_single():
#     data_dir_wuren = './'
#     fname_wuren_8088 = os.path.join(data_dir_wuren, 'wuren_8082.csv')
#     data_dir_youren = './'
#     fname_youren_8088 = os.path.join(data_dir_youren, 'youren_8082.csv')
#
#     f = open(fname_wuren_8088)
#     data_w_8088 = f.read()
#     f.close()
#     f = open(fname_youren_8088)
#     data_y_8088 = f.read()
#     f.close()
#
#     data_w_8088 = data_convert(data_w_8088, 30000)
#     data_y_8088 = data_convert(data_y_8088, 30000)
#
#     pure_data_w_8088 = data_process(data_w_8088)
#     pure_data_y_8088 = data_process(data_y_8088)
#     #print(pure_data_y_8088.shape)
#     dataset_w = dataset_construct_single(pure_data_w_8088,0,200)
#     dataset_y = dataset_construct_single(pure_data_y_8088,0,200)
#     train_data = np.vstack([dataset_w, dataset_y])
#
#     train_lable = np.zeros((400,))
#     for i in range(200):
#         train_lable[i] = 0
#     for i in range(200, 400):
#         train_lable[i] = 1
#
#     dataset_w = dataset_construct_single(pure_data_w_8088,200,300)
#     dataset_y = dataset_construct_single(pure_data_y_8088,200,300)
#     test_data = np.vstack([dataset_w, dataset_y])
#
#     test_lable = np.zeros((200,))
#     for i in range(100):
#         test_lable[i] = 0
#     for i in range(100, 200):
#         test_lable[i] = 1
#
#
#     return train_data,train_lable,test_data,test_lable


def dataset_load(youren_dir,wuren_dir):

    time_window = 100
    test_percent = 0.3


    fName_youren = [os.path.join(youren_dir, fName) for fName in os.listdir(youren_dir)]
    fName_wuren = [os.path.join(wuren_dir,fName) for fName in os.listdir(wuren_dir)]
    data_y = []     #存放多个文件的CSI数据
    data_w = []
    # print(fName_youren,fName_wuren)
    #处理有人时的数据
    for i in range(len(fName_youren)):
        print("-------正在导入有人数据集中的第" + str(i+1) + "个文件-------")
        f = open(fName_youren[i])
        data_y.append(f.read())
        f.close()
        data_y[i] = data_convert(data_y[i])  #得到生数据array
        data_y[i] = data_elm_zero(data_y[i])

    dataset_y_train, dataset_y_test ,label_y_train,label_y_test = dataset_construct(data_y, time_window, test_percent)  ##构建样本



    #处理无人时的数据
    for i in range(len(fName_wuren)):
        print("-------正在导入无人数据集中的第" + str(i+1) + "个文件-------")
        f = open(fName_wuren[i])
        data_w.append(f.read())
        f.close()
        data_w[i] = data_convert(data_w[i])   ##将导入的csv数据转换为数组
        data_w[i] = data_elm_zero(data_w[i])

    dataset_w_train, dataset_w_test ,label_w_train,label_w_test = dataset_construct(data_w, time_window, test_percent,youren=False)   ##构建样本


    ##整合训练集和测试集

    train_data = np.vstack((dataset_y_train,dataset_w_train))
    test_data = np.vstack((dataset_y_test,dataset_w_test))
    train_label = np.hstack((label_y_train,label_w_train))
    test_label = np.hstack((label_y_test,label_w_test))


    print("未经处理的训练集和测试集的形状分别为",train_data.shape,test_data.shape,train_label.shape,test_label.shape)
    return train_data,train_label,test_data,test_label

#dataset_load("./youren","./wuren")