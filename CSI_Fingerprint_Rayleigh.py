import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.signal import gauss_spline
import os as os
from filter import Filter

class Rayleigh_Fingerprint(object):
    def __init__(self):
        pass

    def betax(self,data, b):
        '''
        此为通过极大似然估计求得的f(β）=0的表达式
        :param data: 输入数据
        :param b: β的值
        :return: 根据输入表达式的值
        '''
        n = len(data)
        xigma = np.sqrt(sum([(x + b) ** 2 for x in data]) / 2 / n)
        re = sum([1 / (x + b) - (x + b) / xigma for x in data])
        return re, xigma

    def para_estimate(self, data, threshold=0.001, iteration_max=10):
        '''
        由于β的表达式较为复杂，难以求解，用试探+二分法进行求解
        基于假设：由瑞利分布可知，其所有取值应均大于0，改进瑞利分布应有所有取值有xi+β>0，所以β的解应分布在-Xmin右侧附近，
        由此，从Xmin右侧以0.1为步长探索，找到beta表达式过零的区间，再用二分法对该0.1长度的过零区间查找β的精确解
        :param data:输入数据
        :param threshold:表达式小于多少可以返回
        :param iteration_max:最大迭代次数
        :return:mid即为β，xigema即为σ
        '''
        left = -min(data)
        right = left + 0.1
        # 首先通过试探法找到0点左右范围
        while self.betax(data, right)[0] > 0:
            left = right
            right += 0.1
        # 通过二分法找到0点
        mid = (left + right) / 2
        val, xigma = self.betax(data, mid)
        iteration_time = 0
        while abs(val) > threshold and iteration_time < iteration_max:
            if val < 0:
                right = mid
            elif val > 0:
                left = mid
            else:
                return mid
            mid = (left + right) / 2
            val, xigma = self.betax(data, mid)
            iteration_time += 1
        return -mid, xigma

    def get_attr_matrix(self,data):
        '''
        对输入的m行n列的数据，获取其特征指纹矩阵
        !!!此处未加入第三列，动作影响系数
        :param data:
        :return:
        '''
        attr_num = 2
        m = data.shape[1]   #子载波数目
        re = np.zeros([attr_num,m])
        for i in range(m):
            d_data = Filter.wavelet_denoising(data[:,i])  #小波变换滤波
            beta,xigma = self.para_estimate(d_data)
            re[:,i]=np.array([beta,xigma])
        return re

