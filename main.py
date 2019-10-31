import model as m
import torch
import samples as s
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
iteration = 2
train_net    = [0, 1, 0, 0, 0, 0]
conti_net    = [0, 0, 0, 0, 0, 0]
infer_net    = [0, 0, 0, 0, 0, 0]
train_epochs = [5000, 8000, 50000, 50000, 50000, 50000]

use_partial = [False, False, False, False, False, False]
num_samples = [4000, 2000, 2000, 1000, 1000, 1000]
sparsity_lambda = [3, 0, 3, 3, 3, 3]
weight_decay = [0, 0, 0, 3e-4, 3e-4, 3e-4]
epsilon = [-1, -1, -1, -1, -1, -1]

filter_shapes = [
        [100, 1, 10, 10],
        [200, 100, 10, 10],
        [400, 200, 10, 10],
        [500, 400, 10, 10],
        [500, 500, 10, 10],
        [500, 500, 10, 10]
        ]

filter_strides = [2, 2, 1, 1, 1, 1]

nets = []
for filter_shape in filter_shapes:
    nets.append(m.AE(np.prod(filter_shape[1:4]), filter_shape[0]).to(device))

spectro_dir = '../../Data/TIMIT/'
root_dir = '../../intermDat/FHSAE' + str(iteration) + '/'
#root_dir = '../../intermDat/FHSAE1/'

#spectro_dir for first layer, Conv1-5 for layers 2-6
data_dirs = [spectro_dir] + [root_dir + 'Conv' + str(i) for i in range(1, 6)]
data_funcs = [s.get_from_spectro] + [s.get for i in range(1, 6)]

out_dirs = data_dirs[1:] + [root_dir + 'Conv6']

for i in range(len(nets)):
    print("===== NET #%d =====" % (i + 1))
    
    if train_net[i]:
        if conti_net[i]:
            nets[i] = torch.load(os.path.join(out_dirs[i], 'net' + str(iteration)), map_location=device)
            #Make backup
            torch.save(nets[i], os.path.join(out_dirs[i], 'tempnet' + str(iteration)))
        #Get normalized samples
        print("Loading Training Data")
        samples = data_funcs[i](data_dirs[i], num_samples=num_samples[i], partial=use_partial[i], device=device)
        print("Initializing Training")
        if i == 0:
            nets[i].train_lbfgs(iteration, samples, train_epochs[i], out_dirs[i], sparsity_lambda=sparsity_lambda[i], device=device, cont=conti_net[i], weight_decay=weight_decay[i])
        else:
            torch.save(samples, os.path.join(data_dirs[i], 'samples'))
#            nets[i].train_lbfgs(iteration, samples, train_epochs[i], out_dirs[i], sparsity_lambda=sparsity_lambda[i], device=device, cont=conti_net[i], weight_decay=weight_decay[i])
            nets[i].train_adam(iteration, samples, train_epochs[i], out_dirs[i], sparsity_lambda=sparsity_lambda[i], device=device, cont=conti_net[i], weight_decay=weight_decay[i])
    if infer_net[i]:
        nets[i] = torch.load(os.path.join(out_dirs[i], 'net' + str(iteration)), map_location=device)
        print("Initializing Inferring")
        if i == 0:
            nets[i].infer_from_spectro(data_dirs[i], out_dirs[i], filter_shapes[i], filter_strides[i], epsilon=epsilon[i], device=device)
        else:
            nets[i].infer(data_dirs[i], out_dirs[i], filter_shapes[i], filter_strides[i], epsilon=epsilon[i], device=device)
        
        
        
        
        
        
        
        
    