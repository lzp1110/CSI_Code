import os
import numpy as np
from matplotlib import pyplot as plt
from CSI_Fingerprint_Rayleigh import Rayleigh_Fingerprint
from Adaptive_Bandwidth_Estimation import ABE
from sklearn.preprocessing import MinMaxScaler
from data_load import  dataset_load
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import time



def second_difference(list,indice,k):
    beta_avr = []
    for i in range(int(len(list)/indice/k)):
        Beta = []
        for j in range(k):
            #beta.append(min(list[(i*k+j)*indice:(i*k+j+1)*indice]))
            beta,sigma = Fingerprint.para_estimate(list[(i*k+j)*indice:(i*k+j+1)*indice])
            Beta.append(beta)
        beta_avr.append(sum(Beta)/k)  #3个点取一次平均值

    difference = []
    for j in range(len(beta_avr) - 2):
        difference.append(abs((beta_avr[j] - beta_avr[j+1]) - (beta_avr[j+1] - beta_avr[j+2])))
    return difference




def Rayleigh_test():
    print("start analyze Rayleigh")

    abe_method = ABE()
    abe_method.preview_estimate_result(2,test_data[50][:,2])



    Fingerprint = Rayleigh_Fingerprint()
    y_print = np.zeros([2,train_data.shape[2]])
    w_print = np.zeros([2,train_data.shape[2]])


    start_time = time.time()
    num = int(len(train_data)/2)
    for i in range(num):
         youren = Fingerprint.get_attr_matrix(train_data[i])
         y_print += youren
         wuren = Fingerprint.get_attr_matrix(train_data[i + num])
         w_print += wuren
         print("训练进度：",i/num*100,"%")
    y_print = y_print/num
    w_print = w_print/num
    end_time = time.time()
    print("消耗时间：",end_time-start_time,"s")
    #
    # # scaler = MinMaxScaler(feature_range=(0, 1))
    # # scaler.fit_transform(np.concatenate([y_print,j_print,w_print],axis=0))
    # # y_print,j_print,w_print = map(scaler.transform,[y_print,j_print,w_print])
    #
    #
    print("start test")
    correct_w = 0
    correct_y = 0

    ceshi = []
    shiji = []

    test_num = int(len(test_data)/2)

    for i in range(test_num):
        real = Fingerprint.get_attr_matrix(test_data[i])
        # real = scaler.transform(real)
        distance1 = np.linalg.norm(real - y_print)
        distance2 = np.linalg.norm(real - w_print)
        list = [distance1,distance2]
        if distance1 == min(list):
            correct_y += 1
            ceshi.append(0)
        else:
            ceshi.append(1)
        print("有人检测进度：",i/test_num*100,"%")


    for i in range(test_num,test_num*2):
        real = Fingerprint.get_attr_matrix(test_data[i])
        # real = scaler.transform(real)
        distance1 = np.linalg.norm(real - y_print)
        distance2 = np.linalg.norm(real - w_print)
        list = [distance1,distance2]
        if distance2 == min(list):
            correct_w += 1
            ceshi.append(0)
        else:
            ceshi.append(1)
        print("无人检测进度：",i/test_num*100,"%")


    print("有人检测准确率：",correct_y/test_num*100,"%")
    print("无人检测准确率：",correct_w/test_num*100,"%")


def Pca(data):
    pca = PCA(n_components=2)
    data = data.reshape(data.shape[0]*data.shape[1],data.shape[2])
    pca_data = pca.fit_transform(data)
    ##开始聚类

    new_data = []
    for i in range(int(len(pca_data)/100)):
        kmeans = KMeans(n_clusters=1, n_init=10)
        kmeans.fit(pca_data[i*100:(i+1)*100])
        new_data.append(kmeans.cluster_centers_)
    new_data = np.array(new_data)
    new_data = new_data.reshape(new_data.shape[0],2)
    components = pca.components_.T
    return new_data,components

def try_pca_knn():
    start_time = time.time()
    all_data = np.vstack((train_data,test_data))
    pca_result, component = Pca(all_data)
    knn = KNeighborsClassifier(n_neighbors=4, p=2, metric='minkowski')
    print("start training knn")
    knn.fit(pca_result[:len(train_label)],train_label)
    print("end training knn")
    end_time = time.time()
    print("消耗时间：",end_time-start_time,"s")
    # print("start convert test_data")
    # test_pca_result = []
    # for i in range(len(test_data)):
    #     pca_cal = np.dot(test_data[i],component)
    #     kmeans = KMeans(n_clusters=1,n_init=10)
    #     kmeans.fit(pca_cal)
    #     test_pca_result.append(kmeans.cluster_centers_)
    # test_pca_result = np.array(test_pca_result)
    # test_pca_result = test_pca_result.reshape(test_pca_result.shape[0],2)

    print("start test knn")
    Z = knn.predict(pca_result[len(train_label):])
    print(Z)
    print(Z.shape)
    correct_y = 0
    correct_w = 0
    num = int(len(Z)/2)
    for j in range(num):
        if Z[j] == test_label[j]:
            correct_y = correct_y + 1
    for j in range(num,len(Z)):
        if Z[j] == test_label[j]:
            correct_w = correct_w + 1

    print("有人检测的准确率：", correct_y/num*100, "%")
    print("无人检测的准确率：", correct_w/num*100, "%")


def try_pca_OVO():
    start_time = time.time()
    all_data = np.vstack((train_data,test_data))
    pca_result, component = Pca(all_data)

    print("start training ovo")
    log_reg = LogisticRegression()
    ovo = OneVsOneClassifier(log_reg)
    ovo.fit(pca_result[:len(train_label)],train_label)
    print("end training ovo")
    end_time = time.time()
    print("消耗时间：", end_time - start_time, "s")
    # print("start convert test_data")
    # test_pca_result = []
    # for i in range(len(test_data)):
    #     pca_cal = np.dot(test_data[i],component)
    #     kmeans = KMeans(n_clusters=1,n_init=10)
    #     kmeans.fit(pca_cal)
    #     test_pca_result.append(kmeans.cluster_centers_)
    # test_pca_result = np.array(test_pca_result)
    # test_pca_result = test_pca_result.reshape(test_pca_result.shape[0],2)

    print("start test ovo")
    Z = ovo.predict(pca_result[len(train_label):])
    print(Z)
    print(Z.shape)
    correct_y = 0
    correct_w = 0
    num = int(len(Z)/2)
    for j in range(num):
        if Z[j] == test_label[j]:
            correct_y = correct_y + 1
    for j in range(num,len(Z)):
        if Z[j] == test_label[j]:
            correct_w = correct_w + 1

    print("有人检测的准确率：", correct_y/num*100, "%")
    print("无人检测的准确率：", correct_w/num*100, "%")


def try_pca_OVR():
    start_time = time.time()
    all_data = np.vstack((train_data,test_data))
    pca_result, component = Pca(all_data)

    print("start training ovr")
    log_reg = LogisticRegression()
    ovr = OneVsRestClassifier(log_reg)
    ovr.fit(pca_result[:len(train_label)],train_label)
    print("end training ovr")
    end_time = time.time()
    print("消耗时间：", end_time - start_time, "s")
    # print("start convert test_data")
    # test_pca_result = []
    # for i in range(len(test_data)):
    #     pca_cal = np.dot(test_data[i],component)
    #     kmeans = KMeans(n_clusters=1,n_init=10)
    #     kmeans.fit(pca_cal)
    #     test_pca_result.append(kmeans.cluster_centers_)
    # test_pca_result = np.array(test_pca_result)
    # test_pca_result = test_pca_result.reshape(test_pca_result.shape[0],2)

    print("start test ovr")
    Z = ovr.predict(pca_result[len(train_label):])
    print(Z)
    print(Z.shape)
    correct_y = 0
    correct_w = 0
    num = int(len(Z)/2)
    for j in range(num):
        if Z[j] == test_label[j]:
            correct_y = correct_y + 1
    for j in range(num,len(Z)):
        if Z[j] == test_label[j]:
            correct_w = correct_w + 1

    print("有人检测的准确率：", correct_y/num*100, "%")
    print("无人检测的准确率：", correct_w/num*100, "%")



def try_pca_svm():
    start_time = time.time()
    all_data = np.vstack((train_data, test_data))
    pca_result, component = Pca(all_data)

    print("start training svm")
    clf = SVC(kernel='linear', C=1, random_state=0)
    clf.fit(pca_result[:len(train_label)],train_label)
    print("end training svm")
    end_time = time.time()
    print("消耗时间：", end_time - start_time, "s")
    # print("start convert test_data")
    # test_pca_result = []
    # for i in range(len(test_data)):
    #     pca_cal = np.dot(test_data[i],component)
    #     kmeans = KMeans(n_clusters=1,n_init=10)
    #     kmeans.fit(pca_cal)
    #     test_pca_result.append(kmeans.cluster_centers_)
    # test_pca_result = np.array(test_pca_result)
    # test_pca_result = test_pca_result.reshape(test_pca_result.shape[0],2)

    print("start test svm")
    Z = clf.predict(pca_result[len(train_label):])
    print(Z)
    print(Z.shape)
    correct_y = 0
    correct_w = 0
    num = int(len(Z) / 2)
    for j in range(num):
        if Z[j] == test_label[j]:
            correct_y = correct_y + 1
    for j in range(num, len(Z)):
        if Z[j] == test_label[j]:
            correct_w = correct_w + 1

    print("有人检测的准确率：", correct_y / num * 100, "%")
    print("无人检测的准确率：", correct_w / num * 100, "%")

def try_pca_mlp():
    start_time = time.time()
    all_data = np.vstack((train_data, test_data))
    pca_result, component = Pca(all_data)

    print("start training svm")
    mlp = MLPClassifier(hidden_layer_sizes=(5, 2), activation='logistic', solver='lbfgs')
    mlp.fit(pca_result[:len(train_label)],train_label)
    print("end training svm")
    end_time = time.time()
    print("消耗时间：", end_time - start_time, "s")
    # print("start convert test_data")
    # test_pca_result = []
    # for i in range(len(test_data)):
    #     pca_cal = np.dot(test_data[i],component)
    #     kmeans = KMeans(n_clusters=1,n_init=10)
    #     kmeans.fit(pca_cal)
    #     test_pca_result.append(kmeans.cluster_centers_)
    # test_pca_result = np.array(test_pca_result)
    # test_pca_result = test_pca_result.reshape(test_pca_result.shape[0],2)

    print("start test svm")
    Z = mlp.predict(pca_result[len(train_label):])
    print(Z)
    print(Z.shape)
    correct_y = 0
    correct_w = 0
    num = int(len(Z) / 2)
    for j in range(num):
        if Z[j] == test_label[j]:
            correct_y = correct_y + 1
    for j in range(num, len(Z)):
        if Z[j] == test_label[j]:
            correct_w = correct_w + 1

    print("有人检测的准确率：", correct_y / num * 100, "%")
    print("无人检测的准确率：", correct_w / num * 100, "%")

# plt.plot(range(200),ceshi,color = "orange")
# plt.title("无人场景下测试结果")
# plt.xlabel("Time")
# #plt.show()
##检测运动和无人选用子载波2-25，静止选用子载波16-26*
#子载波1和28-38为0


youren_dir = './youren'
wuren_dir = './wuren'
train_data,train_label,test_data,test_label = dataset_load(youren_dir,wuren_dir)
train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],train_data.shape[2])
test_data = test_data.reshape(test_data.shape[0],test_data.shape[1],test_data.shape[2])
print(train_data.shape,test_data.shape)


##测试不同算法
Rayleigh_test()
#try_pca_knn()
#try_pca_OVO()
#try_pca_OVR()
#try_pca_svm()
# try_pca_mlp()










#plt.plot(range(30000),data_y_8089[:,13],label  = "yundong")
# plt.plot(range(30000),data_y_8089[:,23],label = "yundong")
# plt.plot(range(30000),data_j_8089[:,23],label = "jingzhi")
# plt.plot(range(30000),data_w_8089[:,23],label = "wuren")
# difference_j = second_difference(data_j_8089[:,13],100,3)
# difference_w = second_difference(data_w_8089[:,13],100,3)
# difference_y = second_difference(data_y_8089[:,13],100,3)


# plt.plot(range(len(difference_j[:])),difference_j[:],label = "jingzhi")
# plt.plot(range(len(difference_w[:])),difference_w[:],label = "wuren")
# plt.plot(range(len(difference_y[:])),difference_y[:],label = "yundong")
# plt.legend(loc = "upper right")
# plt.title("data")
# plt.show()











