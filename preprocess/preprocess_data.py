import scipy.io
import numpy as np
from numpy import *
import glob
from scipy.signal import firwin, lfilter
import matplotlib.pyplot as plt

dataDir = '../data/rawdata/'
files = sorted(glob.glob(dataDir + "*.mat"))

data_X = []
for f in files:
    mat_data = scipy.io.loadmat(f[:-4] + ".mat")
    data_X = mat_data['X'].squeeze()
    print len(data_X)
ratio_train = 0.8
N = len(data_X[0])
print N

all_data = []
for i in range(N):
    temp = []
    for j in range(8):
       temp.append(data_X[j][i])
    all_data.append(temp)

all_data = np.array(all_data)
# all_data = all_data.astype('float32')
print all_data

train, val = all_data[0:3133],all_data[3133:N]

scipy.io.savemat('../data/preprocessdata/trainset.mat', mdict={'trainset':train})
scipy.io.savemat('../data/preprocessdata/valset.mat', mdict={'valset':val})
