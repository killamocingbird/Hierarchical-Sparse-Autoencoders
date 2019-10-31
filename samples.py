import os
from matplotlib.pyplot import imread
from random import randint
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def get_from_spectro(data_dir, patch_size=10, num_samples=1, partial=False, file_name='spectro', device='cpu'):
    
    samples = []
    
    #Assume folders are integers
    dirs = [x for x in os.listdir(data_dir) if str.isdigit(x) and os.path.exists(os.path.join(data_dir, x, 'spectro'))]
    num_files = min(num_samples, len(dirs)) if partial else len(dirs)

    for sub_dir_idx in range(num_files):
        #Read in image
        img = torch.load(os.path.join(data_dir, dirs[sub_dir_idx], file_name))
        if img.shape[0] > img.shape[1]:
            continue
        rand_x = randint(0, img.shape[0] - patch_size)
        rand_y = randint(0, img.shape[1] - patch_size)
        samples.append(np.reshape(img[rand_x:rand_x + patch_size, rand_y:rand_y + patch_size], [-1]))
        
        if (sub_dir_idx + 1)%100 == 0:
            print("%.2f%% Loaded" % ((sub_dir_idx + 1) / num_files * 100))
    
    return normalize(torch.as_tensor(samples, device=device))

def get(data_dir, patch_size=10, num_samples=1, partial=False, device='cpu'):
    
    try:
        samples = torch.load(os.path.join(data_dir, 'samples'), map_location=device)
        return samples
    except:
        samples = None
    
    #Assume files are integers
    files = [x for x in os.listdir(data_dir) if str.isdigit(x)]
    num_files = min(num_samples, len(files)) if partial else len(files)
    
    done_first = False
    
    for file_idx in range(num_files):
        X = torch.load(os.path.join(data_dir, files[file_idx]), map_location=device)
        #Input is of size [Channels, x, y]
        rand_x = randint(0, X.shape[1] - patch_size)
        rand_y = randint(0, X.shape[2] - patch_size)
        X = torch.reshape(X[:, rand_x:rand_x + patch_size, rand_y:rand_y + patch_size], [-1])
        if not done_first:
            samples = torch.Tensor(num_files, X.numel())
            done_first = True
        samples[file_idx, :] = X
    
    return normalize(samples)

#Do nothing because data will come normalized
def normalize(x):
    return x


# =============================================================================
# def normalize(samples):
#     #Subtract mean
#     for i in range(samples.shape[0]):
#         samples[i,:] -= samples[i,:].mean()
#     
#     #Scale by STD
#     std_dev = 3 * torch.std(samples, 0)
#     samples = torch.max(torch.min(samples, std_dev), -std_dev) / std_dev
#     
#     #Rescale from [-1 1] to [0.1 0.9]
#     samples = (samples + 1) * 0.4 + 0.1
#     
#     return samples
# =============================================================================

def normalize_single_sample(sample):
    #Assuming dimensions are 1xfeatures
    sample -= sample.mean()
    std_dev = torch.std(sample)
    sample = torch.max(torch.min(sample, std_dev), -std_dev) / std_dev
    
    sample = (sample + 1) * 0.4 + 0.1
    
    return sample


def PCA_whitening(samples, epsilon=1e-5):
    #Assuming sampes is of nxm, where each row is a sample
    x = samples.t()
    #Now each column is a sample
    sigma = torch.matmul(x, x.t()) / x.shape[1]
    u, s, v = torch.svd(sigma)
    
    xZCwhite = torch.matmul(torch.matmul(torch.matmul(u, torch.diag(1 / torch.sqrt(s + epsilon))), u.t()), x)
    
    return xZCwhite.t()

            