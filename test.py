import model as m
import samples as s
import torch


data_dir = 'D:/Data/TIMIT'
samples = s.get_from_spectro(data_dir, num_samples = 100, partial = False)
#samples = s.PCA_whitening(samples)
net = m.AE(100, 64)

net.train_lbfgs(1, samples, 2000, './', decov=0, weight_decay=3e-4, sparsity_lambda=3)

import matplotlib.pyplot as plt
weights = list(net.parameters())[0].detach().to('cpu')
fig = plt.figure(figsize=(40, 40))
for i in range(weights.shape[0]):
    filt = torch.reshape(weights[i,:], [10, 10])
#    filt = filt / torch.sqrt(torch.sum(filt**2))
    ax = fig.add_subplot(8, 8, i + 1)
    ax.imshow(filt, cmap='jet', interpolation = 'nearest')
    
fig.savefig('./filtersTest64.png')
#fig.savefig('../../LAE/filtersGreyWTAFacesDeconv' + str(patch_size) + 'in-batch' + str(batch_size) + '.png')
plt.close(fig)