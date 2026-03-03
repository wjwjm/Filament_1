import numpy as np
import scipy.io

data = np.load('khzfil_out.npz')          # 加载
dic = {k: data[k] for k in data}   # 转成 dict
scipy.io.savemat('./matlab保存数据/71706.mat', dic)  # 存成 v7 格式 .mat