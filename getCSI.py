#!/usr/bin/python
# encoding: utf-8
import socket
import threading
import numpy as np
import time


# 毫秒级定时器类
class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self.interval = interval
        self.function = function
        self.is_running = False
        self.args = args
        self.kwargs = kwargs
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        start = int(time.time() * 1000)
        i = 0
        if not self.is_running:
            self.is_running = True

            while True:
                if not self.is_running:
                    break

                i += 1
                now = int(time.time() * 1000)
                sleep_ms = i * self.interval - (now - start)
                if sleep_ms < 0:
                    self.function(*self.args, **self.kwargs)
                else:
                    time.sleep(sleep_ms / 1000.0)
                    self.function(*self.args, **self.kwargs)

    def stop(self):
        self.is_running = False


# 读取CSI数据类
class ReadCSIData(object):
    def __init__(self, m=64, n=50, period=0.02, port=None, deviceName=None):
        print("init start")
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.soc.bind(("0.0.0.0", port))
        self.addrCSI = self.soc.recvfrom(1024)[1]              # CSI地址

        self.period = period                                   # 获取CSI数据的周期
        self.CSIData = []                                      # 获取的CSI数据

        self.deviceName = deviceName                           # 设备名
        self.m = m                                             # CSI矩阵列数
        self.n = n                                             # CSI矩阵行数
        self.CSIMatrixIni = None                               # 初始化CSI矩阵, 默认m个子载波通道、n个时序包
        self.th1 = threading.Thread(target=self.cycleTask)
        self.th1.start()
        print('{}开始:{}'.format(port , int(1000*time.time())))

    # 将16进制数据解析为带符号的10进制数据
    @staticmethod
    def parseData(data):
        CSIData = []
        for k in range(len(data)):
            CSIData.append(int.from_bytes(data[k:k + 1], byteorder='big', signed=True))
        return CSIData

    def cycleTask(self):
        RepeatedTimer(self.period*1000, self.requestCSI)  # 定时邀约数据

    # 定时发送邀约数据指令
    def requestCSI(self):
        if self.addrCSI is not None:
            self.soc.sendto(b"csi", self.addrCSI)
        # th = threading.Timer(self.period, self.requestCSI)
        # th.start()

    def CSIMatrix(self):
        num = self.n                                  # 次数计数
        while num >= 1:
            csiRawData = self.soc.recvfrom(1024)[0]   # 获取csi原始数据
            csiData = self.parseData(csiRawData)      # 解析

            # print(self.deviceName + '获取数据进度：', round((self.n-num)/self.n, 2)*100, "%")
            # 数据长度判定, 其对象为数据长度为2倍CSI矩阵行数的解析数据
            if len(csiData) == 2*self.m:
                temp = [int(time.time()*1000)]
                # 对解析数据求幅度, 并将其构成一个长度为self.m的list
                for i in range(self.m):
                    temp.append(round((csiData[2 * i] ** 2 + csiData[2 * i + 1] ** 2) ** 0.5, 2))
                self.CSIData.append(temp)                   # 添加至CSI数据大list中
                num = num - 1

        self.CSIMatrixIni = np.mat(self.CSIData)            # 将list转换为矩阵
        self.CSIData = []                                   # CSI数据list复位
        return self.CSIMatrixIni



