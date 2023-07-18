#!/usr/bin/python
# encoding: utf-8
import numpy as np
import pywt
from sklearn.decomposition import PCA
import scipy.signal as signal
import pandas as pd
import os
import matplotlib.pyplot as plt


# 滤波类
class Filter(object):
    def __init__(self):
        pass

    # hample滤波
    @staticmethod
    def hample(X, nsigma=3, window_len=3):
        length = X.shape[0] - 1
        iLo = np.array([i - window_len for i in range(0, length + 1)])
        iHi = np.array([i + window_len for i in range(0, length + 1)])
        iLo[iLo < 0] = 0
        iHi[iHi > length] = length
        xmad = []
        xmedian = []
        for i in range(length + 1):
            w = X[iLo[i]:iHi[i] + 1]
            medj = np.median(w)
            mad = np.median(np.abs(w - medj))
            xmad.append(mad)
            xmedian.append(medj)
        xmad = np.array(xmad)
        xmedian = np.array(xmedian)
        scale = 1.4826  # 缩放
        xsigma = scale * xmad
        xi = ~(np.abs(X - xmedian) <= nsigma * xsigma)  # 找出离群点（即超过nsigma个标准差）

        # 将离群点替换为中为数值
        xf = X.copy()
        xf[xi] = xmedian[xi]
        return xf

    # 小波
    @staticmethod
    def wavelet_denoising(data, wavelet='db4', threshold=0.04):
        """
        :param data: 输入数据
        :param wavelet: 选用的小波，默认4阶daubechies
        :param threshold:阈值，此处用硬阈值，原文用的自适应阈值选取算法文献没看懂
        :return: 滤波后的波形
        """
        w = pywt.Wavelet(wavelet)  # 选用Daubechies4小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, wavelet, level=maxlev)  # 将信号进行小波分解
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))  # 将噪声滤波

        datarec = pywt.waverec(coeffs, wavelet)  # 将信号进行小波重构
        return datarec

    # 主成分分析(PCA), 输入维度为channel_number(行)的矩阵, 返回维度为channel_number(列)的主成分矩阵
    @staticmethod
    def pca(data, channel_number=40):
        method = PCA(n_components=channel_number)
        # 对数据进行标准化处理
        # data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        # 训练PCA模型并进行数据转换
        data = method.fit_transform(data)
        return data

    # 移动均值
    @staticmethod
    def wma(data, m=5):
        """
        :param data: 数据
        :param m: 考虑多少个点的相关性
        :return:
        """
        data_array = np.array(data)
        data_len = data_array.shape[0]
        data_ref = data_array[:]
        for i in range(data_len):
            his_data_num = min(m, i+1)
            weight = range(1, his_data_num+1)
            data_array[i] = sum([x*data_ref[i-his_data_num+x] for x in weight])/sum(weight)
        return data_array
    @staticmethod
    def high_pass_filter(data,fs = 50,fc = 5):
        b, a= signal.butter(1, fc / (fs / 2), 'highpass')
        y = signal.filtfilt(b, a, data)
        dc_signal = np.mean(data) * np.ones_like(data)
        return y + dc_signal       ##保留直流分量的滤波结果

"""
fileAddr = 'C:/Users/feng/Desktop/场景识别/data_jingzhi'
domain = os.path.abspath(fileAddr)                       # 获取文件夹的路径
fileName = os.path.join(domain, os.listdir(fileAddr)[1])                            # 将路径与文件名结合起来就是每个文件的完整路径
CSIDFMatrix = pd.read_csv(fileName, header=None)
CSIDFMatrix = np.array(CSIDFMatrix)                             # 将csv数据还原为矩阵

A = Filter()

haData = np.zeros(np.shape(CSIDFMatrix))
wmaData = np.zeros(np.shape(CSIDFMatrix))

for i in range(np.shape(CSIDFMatrix)[0]):
    haData[i] = A.hample(CSIDFMatrix[i])
    wmaData[i] = A.wma(haData[i])

pcaData = A.pca(wmaData, min(np.shape(wmaData)[0], np.shape(wmaData)[1]))

xxxxx = np.array(range(50))
plt.figure()
for i in range(64):

    # plt.plot(xxxxx, CSIDFMatrix[i], label="ini", linestyle=":")
    # plt.plot(xxxxx, haData[i], label="hample", linestyle=":")
    # plt.plot(xxxxx, wmaData[i], label="wma", linestyle=":")
    plt.plot(xxxxx, pcaData[i], label="pca", linestyle=":")
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号
plt.legend()
plt.ylim([-1, 1])
plt.xlabel("子载波通道")
plt.ylabel("数据详情")
plt.show()
"""
