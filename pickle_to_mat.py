

import torch, scipy.io, os

data_path = 'C:/Users/Justin Wang/Documents/Data/TIMIT/'
output_path = 'C:/Users/Justin Wang/Documents/Data/TIMIT_MAT/'

#Extract folders
dirs = os.listdir(data_path)
dirs = [dirs[i] for i in range(len(dirs)) if dirs[i].isnumeric()]

for i in range(len(dirs)):
    folder_name = dirs[i]
    temp_dat_path = os.path.join(data_path, folder_name, 'spectro')
    temp_dat = torch.load(temp_dat_path)
    scipy.io.savemat(output_path + 'x_' + folder_name + '.mat', mdict = {'x': temp_dat})
    
    if (i + 1)%100 == 0:
        print("%.2f%% Done" % (100 * (i + 1) / len(dirs)))
    