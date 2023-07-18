import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.signal import gauss_spline
import os as os
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']


class ABE(object):
    def __init__(self):
        self.default_band = 8
        # 估计维度，例：幅度，相位，即为2维
        self.data_dim = 1

    def Silverman_Estimation(self, data):
        '''
        大拇指法则进行采集
        :return:
        '''
        d = self.data_dim
        n = len(data)
        std = np.std(data)
        h = ((4 / (d + 2)) ** (1 / (d + 4))) * std * (n ** (-(1 / (d + 4))))
        return h

    def scott_Estimation(self, data):
        '''
        scott估计
        :return:
        '''
        n = len(data)
        std = np.std(data)
        h = 3.49 * std * n ** (-1 / 3)
        return h

    def Adaptive_Bandwidth(self, data, data_sample):
        def k_gaussian(x):
            re = np.exp(-x ** 2 / 2) / (2 * np.pi) ** (1 / 2)
            return re

        n = len(data)
        fix_h = self.Silverman_Estimation(data)
        xx = np.array(data).reshape(-1, 1)
        fix_f = KernelDensity(kernel='gaussian', bandwidth=fix_h).fit(xx)
        g = 1
        for i in range(n):
            g = g * np.exp(fix_f.score_samples(np.array([xx[i]]))[0])
        g = np.power(g, 1 / n)

        def lambdaa(x):
            lam = (g ** (-1) * np.exp(fix_f.score_samples(np.array([[x]]))[0])) ** (-0.9)
            return lam

        lambda_list = np.zeros(n)
        for i in range(n):
            lambda_list[i] = lambdaa(xx[i][0])

        def fx(data_sample):
            def get_sample_value(x):
                re = 0
                for i in range(n):
                    re += (1 / lambda_list[i] / fix_h) * k_gaussian((x - xx[i][0]) / (lambda_list[i] * fix_h))
                re /= n
                return re

            nd = len(data_sample)
            array_fx = np.zeros(nd)
            for i in range(nd):
                array_fx[i] = get_sample_value(data_sample[i])
            return array_fx

        array_fx = fx(data_sample)

        #plt.figure()
        #plt.hist(data, bins=90, density=True)
        plt.plot(data_sample, array_fx,label = '自适应核密度')
        #plt.show()

    def preview_estimate_result(self, h, data, Kernel='gaussian',c='b',type = 'wuren'):

        plt.hist(data, bins=30,density=True, label="histogram")
        pd.Series(data).plot.kde(label = "kernel density estimation")
        xx = np.array(data).reshape(-1, 1)
        data_kde = KernelDensity(kernel=Kernel, bandwidth=h).fit(xx)
        log_density = data_kde.score_samples(np.linspace(min(data) - 1, max(data) + 1, 1000)[:, np.newaxis])
        plt.plot(np.linspace(min(data) - 1, max(data) + 1, 1000)[:, np.newaxis], np.exp(log_density),color = c,label="rayleigh distribution")
        plt.legend()
        plt.title("Gaussian Kernel Density Estimation")
        plt.xlabel("Amplitude")
        plt.xlim(-1,20)
        plt.show()

        # return data_kde

def get_csi(path):
    d_files = os.listdir(path)
    data_set = []
    for name in d_files:
        file = np.loadtxt(path + '\\'+name, delimiter=',')
        data_set.append(file)
    return data_set


def load_pickle(file_name):
    t0 = pd.read_pickle(file_name)
    data = t0[t0.keys()[0]][0]
    return data

def ruili_f(data):
    data_sample = np.linspace(min(data), max(data), 1000)
    xigma = np.std(data)
    fx = np.zeros(len(data_sample))
    for i in range(len(data_sample)):
        fx[i] = data_sample[i]/xigma**2*np.exp(-(data_sample[i]**2/2/xigma**2))
    return fx,data_sample

if __name__ == "main":
    abe_method = ABE()
    data_set = get_csi('D:\workandlearn\S\CSI_study\\test')
    jingzhi_data_set_w,wuren_data_set_w,zoudong_data_set_w = data_set[0],data_set[1],data_set[2]
    jingzhi_data_set,wuren_data_set,zoudong_data_set = data_set[0][:,100:200],data_set[1][:,100:200],data_set[2][:,100:200]

    # wuren_data_set = load_pickle('D:\workandlearn\S\CSI_study\\' + 'gzy_509_shebeijiaojin_wuren' + '_Data_set.pkl')[20]
    # jingzhi_data_set = load_pickle('D:\workandlearn\S\CSI_study\\' + 'gzy_509_shebeijiaojin_youren' + '_Data_set.pkl')[20]
    # zoudong_data_set = load_pickle('D:\workandlearn\S\CSI_study\\' + 'gzy_509_shebeijiaojin_zoudong_' + '_Data_set.pkl')[20]
    # h = abe_method.Silverman_Estimation(wuren_test)
    # h2 = abe_method.scott_Estimation(wuren_test)

    # data_kde = abe_method.preview_estimate_result(h2,wuren_test)
    # data_kde2 = abe_method.preview_estimate_result(h,wuren_test)
    # abe_method.Adaptive_Bandwidth(wuren_test,np.linspace(min(wuren_test) - 1, max(wuren_test) + 1, 1000)[:, np.newaxis])
    h1_list = []
    h2_list = []
    h3_list = []
    data_see = wuren_data_set
    lables = ['静止','无人','动作']




    for zai_bo_idx in range(46,52,1):
        plt.figure()
        plt.title('无人'+str(zai_bo_idx))
        # h = abe_method.scott_Estimation(wuren_data_set[zai_bo_idx, :])
        # abe_method.preview_estimate_result(2.5*h, wuren_data_set[zai_bo_idx, :],c = 'r',type='无人')
        # h1_list.append(h)
        # h = abe_method.scott_Estimation(jingzhi_data_set[zai_bo_idx, :])
        # abe_method.preview_estimate_result(2.5*h, jingzhi_data_set[zai_bo_idx, :], c = 'g',type='静止')
        # h2_list.append(h)


        # d_data = wuren_data_set[zai_bo_idx, :]-min(wuren_data_set[zai_bo_idx, :])
        # # h = abe_method.scott_Estimation(d_data)
        # # abe_method.preview_estimate_result( h, d_data, c='r', type='带宽0.12')
        # # abe_method.preview_estimate_result(2*h, d_data, c = 'g',type='带宽0.24')
        # # abe_method.preview_estimate_result(5 * h, d_data, c='b', type='带宽0.6')
        # ruili_fx,data_sample = ruili_f(d_data)
        #
        # plt.plot(data_sample, ruili_fx, label='无人100')
        #
        #
        #
        # d_data = jingzhi_data_set[zai_bo_idx, :] - min(jingzhi_data_set[zai_bo_idx, :])
        # ruili_fx, data_sample = ruili_f(d_data)
        #
        # plt.plot(data_sample,ruili_fx,label = '静止100')



        # d_data = zoudong_data_set[zai_bo_idx, :] - min(zoudong_data_set[zai_bo_idx, :])
        # abe_method.Adaptive_Bandwidth(d_data,
        #                                np.linspace(min(d_data) - 1, max(d_data) + 1, 1000)[:, np.newaxis])
        # ruili_fx, d_sample = ruili_f(d_data)
        # h = abe_method.scott_Estimation(d_data)
        # abe_method.preview_estimate_result( h, d_data, c='r', type='拇指法则')
        # plt.hist(d_data, bins=50, density=True)
        # plt.plot(d_sample, ruili_fx, label='瑞利分布')

        # d_data = wuren_data_set_w[zai_bo_idx, :] - min(wuren_data_set_w[zai_bo_idx, :])
        #
        # ruili_fx, data_sample = ruili_f(d_data)
        #
        # plt.plot(data_sample, ruili_fx, label='无人10000')
        #
        # d_data = jingzhi_data_set_w[zai_bo_idx, :] - min(jingzhi_data_set_w[zai_bo_idx, :])
        # ruili_fx, data_sample = ruili_f(d_data)
        #
        # plt.plot(data_sample, ruili_fx, label='静止10000')
        #
        #
        #
        # d_data = zoudong_data_set_w[zai_bo_idx, :] - min(zoudong_data_set_w[zai_bo_idx, :])
        # ruili_fx, data_sample = ruili_f(d_data)
        #
        # plt.plot(data_sample, ruili_fx, label='走动10000')

        # d_data = zoudong_data_set_w[zai_bo_idx, :]
        # plt.hist(d_data, bins=90, density=True)

        d_data = wuren_data_set_w[zai_bo_idx, :]
        plt.hist(d_data, bins=200, density=True)
        plt.legend()




        plt.show()





    # abe_method.Adaptive_Bandwidth(wuren_data_set[zai_bo_idx, :],
    #                                   np.linspace(min(wuren_data_set[zai_bo_idx, :]) - 1, max(wuren_data_set[zai_bo_idx, :]) + 1, 1000)[:, np.newaxis])





