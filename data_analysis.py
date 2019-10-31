from scipy.io import loadmat
import torch
import matplotlib.pyplot as plt
import numpy as np


path_to_QT = 'C:/Users\Justin Wang\Documents\Tsinghua\IntermResults\w_1.mat'
path_to_AE = 'C:/Users\Justin Wang\Documents\Tsinghua\IntermResults\HSAEC2\\1'

#Load in QT
qt = loadmat(path_to_QT)['w']
qt = qt.transpose(2, 0, 1)

#Load in model results
ae = torch.load(path_to_AE, map_location='cpu')

#bins = np.linspace(-100, 100, 500)

plt.hist(ae.view(-1), bins=500, alpha=0.5, label='AE')
plt.hist(qt.reshape(-1), bins=500, alpha=0.5, label='SC')
plt.legend(loc='upper right')


