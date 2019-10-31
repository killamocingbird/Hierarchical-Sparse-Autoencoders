import torch
import os
import pickle
import tictoc as t
from scipy.stats import ranksums
from scipy.stats import ttest_ind
from sklearn.cluster import AgglomerativeClustering
import F_ratio as f_ratio
import matplotlib.pyplot as plt
import numpy as np

iteration = 1

response_calc = [0, 0, 0, 0, 1, 0]
PSIs_calc = [0, 0, 0, 0, 1, 0]
plot_PSIs = [0, 0, 0, 0, 1, 0]
sparse_calc = [0, 0, 0, 0, 0, 0]

break_early = False
break_num = 2000

num_channels = [24, 128, 256, 512, 500, 500]

data_dir = '../../Data/TIMIT/'
root_dir = '../../intermDat/FHSAE' + str(iteration)
act_map_dirs = [root_dir + '/Conv' + str(i) for i in range(1, 7)]

phoneme_map_dir = '../../Data/TIMIT/Phoneme Mapping.txt'

tic = t.tic()
for i in range(len(num_channels)):
    print("===== NET #%d =====" % (i + 1))
    if response_calc[i]:
        #Create empty dictionary for responses
        phoneme_distribs = dict()
        with open(phoneme_map_dir, 'r') as phoneme_map:
            for line in phoneme_map:
                phoneme = line.split()[1]
                phoneme_distribs[phoneme] = [[] for i in range(num_channels[i])]
        
        files = [file for file in os.listdir(act_map_dirs[i]) if str.isdigit(file)]
        print("Calculating Responses")
        tic()
        for file_idx in range(len(files)):
            
            if break_early and file_idx > break_num:
                break
            
            #Load in activation map
            act_map = torch.load(os.path.join(act_map_dirs[i], files[file_idx]), map_location='cpu')
            
            #Load in phoneme duration data
            phoneme_dat = pickle.load(open(os.path.join(data_dir, files[file_idx], 'procPhonemes.pickle'), 'rb'))
        
            for phoneme in phoneme_dat:
                if phoneme[0] in phoneme_distribs:
                    phoneme_dur = phoneme[1][2 * i + 3]
                    for j in range(num_channels[i]):
                        phoneme_distribs[phoneme[0]][j].append(float(torch.max(torch.abs(act_map[j, :, phoneme_dur[0]:phoneme_dur[1] + 1]))))
            
            if file_idx % 10 == 9:
                if break_early:
                    print("%.2f%% completed" % (100 * file_idx / break_num))
                else:
                    print("%.2f%% completed" % (100 * file_idx / len(files)))
        torch.save(phoneme_distribs, os.path.join(act_map_dirs[i], 'phonemeDistribs'))
        tic.toc()
    if PSIs_calc[i]:
        #Load in phoneme distributions
        phoneme_distribs = torch.load(os.path.join(act_map_dirs[i], 'phonemeDistribs'), map_location='cpu')
        
        #Determine which units are deemed active
        print("Determining Active Units")
        tic()
        active_units = []
        for j in range(num_channels[i]):
            not_silence = []
            for phoneme in phoneme_distribs:
                if phoneme != 'h#':
                    not_silence.extend(phoneme_distribs[phoneme][j])
            ttest = ttest_ind(not_silence, phoneme_distribs['h#'][j])
            if ttest[0] > 0 and ttest[1] < 0.001:
                active_units.append(j)
        tic.toc()
                
        print("Calculating PSIs")
        tic()
        PSIs = [[] for _ in range(len(active_units))]
        for j in range(len(active_units)):
            for x in phoneme_distribs:
                if x == 'h#':
                    continue
                temp = 0
                for y in phoneme_distribs:
                    if x != y and y != 'h#':
                        rtest = ranksums(phoneme_distribs[x][active_units[j]], phoneme_distribs[y][active_units[j]])
                        if rtest[0] > 0 and rtest[1] < 0.01:
                            temp += 1
                PSIs[j].append(temp)
#            if j%10 == 9:
#                print("%.2f%% completed" % (100 * (j + 1) / len(active_units[i])))
        torch.save(PSIs, os.path.join(act_map_dirs[i], 'PSIs'))
        tic.toc()
    if plot_PSIs[i]:
        #Load in PSIs
        PSIs = torch.load(os.path.join(act_map_dirs[i], 'PSIs'))
        cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
        cluster.fit_predict(PSIs)
        temp = list(zip(cluster.labels_, PSIs))
        temp.sort()
        
        print("F-Ratio: ", f_ratio.calc(temp).item())
        
        temp2 = []
        for j in range(len(temp)):
            temp2.append(temp[j][1])
            
        PSIImg = np.array(temp2)
        PSIImg = PSIImg.transpose()
        
        plt.figure(figsize=(20, 20))
        plt.imshow(PSIImg, cmap='Greys')
        plt.savefig(act_map_dirs[i] + '/PSIPlots' + str(i) + '.png')
        
    if sparse_calc[i]:
        print("Calculating Lifetime Sparseness")
        
        sparse_num = torch.zeros(num_channels[i])
        sparse_dem = torch.zeros(num_channels[i])

        files = [file for file in os.listdir(act_map_dirs[i]) if str.isdigit(file)]
        for file_idx in range(len(files)):
            
            if break_early and file_idx > break_num:
                break
            
            #Load in activation map
            act_map = torch.load(os.path.join(act_map_dirs[i], files[file_idx]), map_location='cpu')
        
            for j in range(num_channels[i]):
                sparse_num[j] += torch.mean(act_map[j, :, :])**2
                sparse_dem[j] += torch.mean(act_map[j, :, :]**2)
            
            if file_idx % 100 == 99:
                if break_early:
                    print("%.2f%% completed" % (100 * file_idx / break_num))
                else:
                    print("%.2f%% completed" % (100 * file_idx / len(files)))
        
        sparse_vals = sparse_num / sparse_dem
        sparse_vals = 1 - sparse_vals
        print("Lifetime Sparsity %.2f" % (sparse_vals.mean()))
        
        torch.save(sparse_vals, os.path.join(act_map_dirs[i], 'sparse'))
        
        
        
        
        
