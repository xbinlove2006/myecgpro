"""
参数说明：
path   文件路径+文件名（不包含后缀）
功能设计：
1.读取文件、标注
2.数据分片
3.标注转换0,1
"""
import wfdb
class Ecg():
    def __init__(self,path):
        self.path=path
        self.ecg=None
        self.ann=[]
    def rdecg(self):
        #读取ecg数据
        self.ecg=wfdb.rdrecord(self.path).p_signal
        return self.ecg
    def rdann(self):
        #读取标注
        ann=wfdb.rdann(self.path,'apn')
        self.ann.append(ann.sample)
        self.ann.append(ann.symbol)
        return self.ann

def ann2num(ann,ann_dict):
    num=[]
    for a in ann:
        num.append(ann_dict[a])
    return num

def div_sig(signal,lenth,width=6000):
    divided_sigs=[]
    for i in range(lenth):
        divided_sigs.append(signal[i*width:(i+1)*width])
    return divided_sigs

