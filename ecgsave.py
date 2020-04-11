
# 将所有的ecg记录和标注存为txt文件

import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# 文件路径
ROOT = "E:/deeplearning/apnea-ecg-database-1.0.0/"
# 处理后文件存放路径
SEGMENTS_BASE_PATH = "E:/deeplearning/apnea-ecg-processed/"

# 训练集大小
SEGMENTS_NUMBER_TRAIN = 17045
# 测试集大小
SEGMENTS_NUMBER_TEST = 17268
# 训练集文件名称列表
TRAIN_FILENAME = [
    "a01", "a02", "a03", "a04", "a05", "a06", "a07", "a08", "a09", "a10",
    "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20",
    "b01", "b02", "b03", "b04", "b05",
    "c01", "c02", "c03", "c04", "c05", "c06", "c07", "c08", "c09", "c10"
]

# 训练集每个文件长度（对应多少个60s ecg信号）
TRAIN_LABEL_AMOUNT = [489, 528, 519, 492, 454,
                      510, 511, 501, 495, 517,
                      466, 577, 495, 509, 510,
                      482, 485, 489, 502, 510,
                      487, 517, 441, 429, 433,
                      484, 502, 454, 482, 466,
                      468, 429, 513, 468, 431]
# 测试集文件名列表
TEST_FILENAME = [
    "x01", "x02", "x03", "x04", "x05", "x06", "x07", "x08", "x09", "x10",
    "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20",
    "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30",
    "x31", "x32", "x33", "x34", "x35"
]

# 测试集每个文件长度
TEST_LABEL_AMOUNT = [523, 469, 465, 482, 505,
                     450, 509, 517, 508, 510,
                     457, 527, 506, 490, 498,
                     515, 400, 459, 487, 513,
                     510, 482, 527, 429, 510,
                     520, 498, 495, 470, 511,
                     557, 538, 473, 475, 483]
# ECG信号采样率  1秒 100个点
ECG_RAW_FREQUENCY = 100

ANN_DICT = {'A': 1, 'N': 0}


def ann2num(ann, ann_dict):
    num = []
    for a in ann:
        num.append(ann_dict[a])
    return num



# print(path)
ecg=[]
sample=[]
symbol=[]
nums=[]
print('start')
for i in range(len(TRAIN_FILENAME)):
	path=ROOT+TRAIN_FILENAME[i]
	tmpecg=wfdb.rdrecord(path).p_signal
	ann=wfdb.rdann(path,'apn')
	tmpsymbol=ann.symbol#'A','N'
	tmpnum=ann2num(tmpsymbol,ANN_DICT)#'A','N'转1,0
	tmpsample=ann.sample#标注  对应  ecg信号的起点
	for j in range(TRAIN_LABEL_AMOUNT[i]):#将1分钟的ecg信号放入数组
		if tmpsample[j]+6000<=len(tmpecg) and j<TRAIN_LABEL_AMOUNT[i]:
			ecg.append(tmpecg[tmpsample[j]:tmpsample[j]+6000])
			nums.append(tmpnum[j])

# print(ecg)
print(len(ecg[0]),len(ecg[1]))
print(len(nums))
print('end')
plt.plot(ecg[1][:600])
plt.show()
