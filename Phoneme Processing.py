import os
import math
import pickle
from skimage.io import imread
import torch

class Layer():
    def __init__(self, kernel_size, stride):
        super(Layer, self)
        self.kernel_size = kernel_size
        self.stride = stride
        
    #Immitates dimentiality reduction of CNN or Pooling layer given bounds
    def __call__(self, x, max_len):        
        return [min(math.floor(x[0] / self.stride), max_len - 1), max(math.ceil((x[1] - self.kernel_size) / self.stride) + 1, 1)]
        
    def real_conv(self, x):
        return math.floor((x - self.kernel_size) / self.stride) + 1

class Downsample():
    def __init__(self, scaler):
        super(Downsample, self)
        self.scaler = scaler
        
    #Downsample using scalar to imitate descimation of spectrogram
    def __call__(self, x, y):
        return [math.floor(x[0] / self.scaler), math.ceil(x[1] / self.scaler)]
    
    def real_conv(self, x):
        return math.ceil(x / self.scaler)

datapath = '../../Data/TIMIT/'
used_path = '../../intermDat/FHSAE1/Conv1/'

#Spectrogram decimation parameter
scale = 160

#Layers
layers = [
        Downsample(scale), #Raw to Spectro
        Layer(10, 2), #Conv1
        Layer(2, 1), #Pool1
        Layer(10, 2), #Conv2
        Layer(2, 1), #Pool2
        Layer(10, 1), #Conv3
        Layer(2, 1), #Pool3
        Layer(10, 1), #Conv4
        Layer(2, 1), #Pool4
        Layer(10, 1), #Conv5
        Layer(2, 1), #Pool5
        Layer(10, 1), #Conv6
        Layer(2, 1), #Pool6
        ]

used_dirs = [file for file in os.listdir(used_path) if str.isdigit(file)]
subdirs = [sub_dir for sub_dir in os.listdir(datapath) if str.isdigit(sub_dir) and sub_dir in used_dirs]
for subdir in subdirs:
    
    #Grab phoneme list
    temppath = os.path.join(datapath, subdir)
    with open(os.path.join(temppath, [file for file in os.listdir(temppath) if file.endswith('.PHN')][0]), 'r') as in_file, \
         open(os.path.join(temppath, 'procPhonemes.pickle'), 'wb') as out_file:
        
        outdat = []
        
        max_len = torch.load(os.path.join(temppath, 'spectro'), map_location='cpu').shape[1]
             
        for line in in_file:
            #Input parsing
            linedat = line.split()
            phoneme = linedat[2]
            linedat = linedat[:2]
            linedat = list(map(int, linedat))
            
            #Output preparation
            idxdat = [[linedat[0], linedat[1]]]
            
            temp_bounds = [linedat[0], linedat[1]]
            temp_max = max_len
            first = True
            for layer in layers:
                if not first:
                    temp_max = layer.real_conv(temp_max)
                if temp_max < 0:
                    print(temp_max)
                    print(temppath)
                temp_bounds = layer(temp_bounds, temp_max)
                first = False
                if temp_bounds[0] >= temp_bounds[1]:
                    temp_bounds[1] = temp_bounds[0] + 1
                idxdat.append(temp_bounds)
                
            outdat.append((phoneme, idxdat))
        
        pickle.dump(outdat, out_file)
