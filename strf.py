import matplotlib.pyplot as plt
import torch

import_nets = [1,1,1]
nets = []

for i in range(len(import_nets)):
    if import_nets[i]:
        nets.append(torch.load('net' + str(i + 1), map_location='cpu'))


def calc(nets, num_layers=2):
    
    import numpy as np
    import torch
    from scipy.signal import convolve2d
    
    bn = [24, 128, 256]
    bs = [10, 10, 10]
    s1 = [2, 2, 1]
    
    base = nets[0].f1.weight.data
    strf = []
    strf.append(torch.reshape(base, [-1, bs[0], bs[0]]).permute(1, 2, 0))
    s = bs[0] + bs[0] * s1[0]
    strf.append(torch.zeros(s, s, bn[1]))
    s = s + bs[0] * np.prod(s1[0:2])
    strf.append(torch.zeros(s, s, bn[2]))
    
    for l in range (1, num_layers):
        print("Layer %d:" % (l + 1))
        base = nets[l].f1.weight.data
        base = torch.reshape(base, [bn[l], bn[l-1], bs[l], bs[l]])
        base = base.permute([2, 3, 1, 0])
        for i in range (bn[l]):
            for j in range (bn[l-1]):
                baseEx = np.kron(base[:, :, j, i], torch.ones(np.prod(s1[0:l]),np.prod(s1[0:l])))
                npad = ((0,1), (0,1))
                baseEx = np.pad(baseEx, pad_width = npad, mode='constant', constant_values=0)
                strf[l][:, :, i] = strf[l][:, :, i] + torch.as_tensor(convolve2d(baseEx, strf[l-1][:, :, j], mode='full'))
            if (i + 1)%10 == 0:
                print("%.2f%%" % (100 * (i + 1) / bn[l]))
    
    return strf
    
def visualize(strf, layer=0, save=False, out_dir=''):
    #Visualize single STRF
    fig = plt.figure(figsize=(40, 40))
    for i in range(strf.shape[2]):
        ax = fig.add_subplot(16, 16, i + 1)
        ax.imshow(strf[:,:,i], cmap='jet')
    fig.savefig('BaseSTRF3.png')
    plt.close(fig)

                