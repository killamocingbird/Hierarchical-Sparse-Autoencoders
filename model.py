import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import samples as s
import math
from torch.optim.lr_scheduler import MultiStepLR

class AE(nn.Module):
    def __init__(self, in_features, h_features):
        super(AE, self).__init__()
        self.in_features = in_features
        self.h_features = h_features
        self.f1 = nn.Linear(in_features, h_features)
        self.f2 = nn.Linear(h_features, in_features)
    
    def forward(self, x):
        x = self.f1(x)
        encoded = torch.sigmoid(x)
        decoded = torch.sigmoid(self.f2(encoded))
        return (encoded, decoded)
    
    def train_lbfgs(self, iteration, samples, epochs, out_dir, rho=0.01, sparsity_lambda=3, weight_decay=0.0001, decov=0, print_freq=10, device='cpu', cont=False):
        #### INPUT ####
        #samples is a Tensor of dimensions Features X Observations
        #epochs defines number of iterations of training
        #out_dir is directory of net and optim output
        #rho is input used in equation for KL Divergence, input -1 to not use KL Divergence
        #sparsity_lambda is constant to balance sparsity to recon error
        #weight_decay is constant to control weight decay regulation
        #print_freq is frequency of loss print out and net/optim saving
        
        #Cast everything to proper device
        self.to(device)
        samples = samples.float().to(device)
        
        #Using LBFGS, not using minibatches
        if cont and False:
            optimizer = torch.load(os.path.join(out_dir, 'optim' + str(iteration)), map_location=device)
            #Make backup in case of failure
            torch.save(optimizer, os.path.join(out_dir, 'tempoptim' + str(iteration)))
        else:
            optimizer = optim.LBFGS(self.parameters(), max_iter=1, lr=1e-2)
#        scheduler = StepLR(optimizer, step_size=3_000, gamma=10)
        min_loss = 1000000000
        running_mse_loss = 0.0
        running_kl_loss = 0.0
        running_l2_norm = 0.0
        running_decov_loss = 0.0
        for epoch in range(epochs):
            num_evals = 0
            #Closure for optimization
            def closure():
                
                nonlocal min_loss
                nonlocal running_mse_loss
                nonlocal running_kl_loss
                nonlocal running_l2_norm
                nonlocal running_decov_loss
                nonlocal num_evals
                
                optimizer.zero_grad()
                encoded, decoded = self(samples)
                
                #Calculate Recon Error
                mse_loss = torch.sum((decoded - samples)**2) / samples.shape[0] / 2
                
                #Calculate KL Divergence
                rho_hat = encoded.mean(0)
                kl_div = torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))
                kl_loss = sparsity_lambda * kl_div
                
                #Calculate Weight Decay
                l2_reg = 0.0
                for param in self.parameters():
                    #Apply weight decay only to weights and not bias
                    if param.numel() == self.in_features * self.h_features:
                        l2_reg += torch.sum(param**2)
                l2_norm = weight_decay * l2_reg / 2
                
                #Calculate Decov Loss
                decov_loss = decov * DeCov_loss(encoded)
                
                #Calculate Total Loss
                loss = mse_loss + kl_loss + l2_norm + decov_loss
#                loss = mse_loss + decov_loss
                
                #Add onto running loss
                running_mse_loss += mse_loss.item()
                running_kl_loss += kl_loss.item()
                running_l2_norm += l2_norm.item()
                running_decov_loss += decov_loss.item()
                
                loss.backward()
                
                num_evals += 1
                if (epoch + 1)%print_freq == 0 and num_evals == 1:
                    latest_loss = (running_mse_loss + running_kl_loss + running_l2_norm + running_decov_loss) / print_freq / num_evals
                    print("[%d]: %.10f" % (epoch + 1, latest_loss))
                    print("M: %.8f \t K: %.8f D: %.8f" % (running_mse_loss / print_freq / num_evals, running_kl_loss / print_freq / num_evals, running_decov_loss / print_freq / num_evals))
                    if latest_loss < min_loss:
                        torch.save(self, os.path.join(out_dir, 'net' + str(iteration)))
                        torch.save(optimizer, os.path.join(out_dir, 'optim' + str(iteration)))
                        min_loss = latest_loss
                    
                    running_mse_loss = 0
                    running_kl_loss = 0
                    running_l2_norm = 0
                    running_decov_loss = 0
                    num_evals = 0
                
                return loss
            
            optimizer.step(closure)
#            scheduler.step()
            
    def train_adam(self, iteration, samples, epochs, out_dir, batch_size=64, lr = 1e-4, rho=0.01, sparsity_lambda=1e-2, weight_decay=0.0001, print_freq=10, device='cpu', cont=False):
        self.to(device)
        samples = samples.float().to(device)
        
        #Using Adam
        
        if cont:
            optimizer = optim.Adam(self.parameters(), lr=lr, amsgrad=True)
            optimizer.load_state_dict(torch.load(os.path.join(out_dir, 'optim' + str(iteration)), map_location=device))
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            #Make backup in case of failure
            torch.save(optimizer.state_dict(), os.path.join(out_dir, 'tempoptim' + str(iteration)))
        else:
            optimizer = optim.Adam(self.parameters(), lr=lr, amsgrad=True)
#            scheduler = MultiStepLR(optimizer, milestones = [100], gamma=0.1)
        
        batch_size = len(samples) if batch_size == -1 else batch_size
        min_loss = 1000000000
        running_mse_loss = 0.0
        running_kl_loss = 0.0
        running_l2_norm = 0.0
        loop_num = 0
        
        for epoch in range(epochs):
            for i in range(math.floor(samples.shape[0] / batch_size)):
                X = samples[i * batch_size:(i + 1) * batch_size].float()
                encoded, decoded = self(X.float())
                
                #Calculate Recon Error
                mse_loss = torch.sum((decoded - X)**2) / X.shape[0] / 2
                
                #Calculate KL Divergence
                rho_hat = encoded.mean(0)
                kl_div = torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))
                kl_loss = sparsity_lambda * kl_div
                
                #Calculate Weight Decay
                l2_reg = 0.0
                for param in self.parameters():
                    #Apply weight decay only to weights and not bias
                    if param.numel() == self.in_features * self.h_features:
                        l2_reg += torch.sum(param**2)
                l2_norm = weight_decay * l2_reg / 2
                
                #Calculate Total Loss
                loss = mse_loss + kl_loss + l2_norm
                
                #Add onto running loss
                running_mse_loss += mse_loss.item()
                running_kl_loss += kl_loss.item()
                running_l2_norm += l2_norm.item()
                loop_num += 1
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (epoch + 1)%print_freq == 0 and i == 0:
                    latest_loss = (running_mse_loss + running_kl_loss + running_l2_norm) / loop_num
                    print("[%d]: %.10f" % (epoch + 1, latest_loss))
#                    print("M: %.8f \t K: %.8f \t RK: %.8f" % (running_mse_loss / loop_num, running_kl_loss / loop_num, running_kl_loss / loop_num / sparsity_lambda))
                    print("M: %.8f \t K: %.8f \t RK: %.8f" % (running_mse_loss / loop_num, running_kl_loss / loop_num, running_kl_loss / loop_num / 1))
                    if latest_loss < min_loss:
                        torch.save(self, os.path.join(out_dir, 'net' + str(iteration)))
                        torch.save(optimizer.state_dict(), os.path.join(out_dir, 'optim' + str(iteration)))
                        min_loss = latest_loss
                    
                    loop_num = 0
                    running_mse_loss = 0
                    running_kl_loss = 0
                    running_l2_norm = 0
#            if not cont:
#                scheduler.step()
                    
    def infer(self, data_dir, out_dir, shape, stride, device='cpu', epsilon=-1):
        #Assume that the data_dir will contain pickled files which are 1 indexed
        #Shape will be desired shape of filters
        #Each file will be of shape [channels, x, y]
        #Perform convolution then pooling
        filters = self.f1.weight.data
        
        #Prune weights
        if epsilon != -1:
            filters = prune_weights(filters, epsilon)
            
        filters = torch.reshape(filters, shape).to(device)
        files = [x for x in os.listdir(data_dir) if str.isdigit(x)]
        for file in files:
            X = torch.load(os.path.join(data_dir, file), map_location=device)
            X_shape = X.shape
            #Normalize input
            X = s.normalize_single_sample(X.view(-1).unsqueeze(0))
            X = torch.reshape(X, [1, X_shape[0], X_shape[1], X_shape[2]]).float()
            #Convolve
            out_map = F.conv2d(X, filters, stride=stride)
            #Pooling
            out_map, _ = F.max_pool2d_with_indices(out_map, 2, stride=1)
            
            #Reshape to remove batch dimension
            out_map = out_map.squeeze(0)
            torch.save(out_map, os.path.join(out_dir, file))
            
    def infer_from_spectro(self, data_dir, out_dir, shape, stride, device='cpu', epsilon=-1):
        filters = self.f1.weight.data
        
        #Prune weights
        if epsilon != -1:
            filters = prune_weights(filters, epsilon)
        
        filters = torch.reshape(filters, [-1, 1, shape[2], shape[3]]).to(device)
        
        from skimage.io import imread
        
        #Extract folder names with assumption that all folder names are integers
        dirs = [x for x in os.listdir(data_dir) if str.isdigit(x)]
        
        for i in range(len(dirs)):
            folder_name = dirs[i]
            #Read in image
            img = torch.load(os.path.join(data_dir, folder_name, 'spectro'))
            if img.shape[0] > img.shape[1]:
                continue
            img = torch.as_tensor(img)
            img_shape = img.shape
            img = s.normalize_single_sample(img.view(-1).unsqueeze(0))
            img = torch.reshape(img, [1, 1, img_shape[0], img_shape[1]]).float()
            img = img.to(device)
            
            #Convolve
            out_map = F.conv2d(img, filters, stride=stride)
            #Pooling
            out_map, _ = F.max_pool2d_with_indices(out_map, 2, stride=1)
            
            #Reshape to remove batch dimension
            out_map = out_map.squeeze(0)
            
            #Save
            torch.save(out_map, os.path.join(out_dir, folder_name))  
            
def DeCov_loss(h):
    #Assume h is of size, [batch, features]
    C_temp = torch.sum(h, 0) - torch.mean(h, 0)
    C = torch.ger(C_temp, C_temp) / h.shape[0]
    return ((C**2).sum() - (C.diag()**2).sum()) / 2

def prune_weights(weights, epsilon):
    #weights will have dimensions filtersXfeatures
    included = [True for _ in range(len(weights))]
    for i in range(len(weights)):
        for j in range(len(weights)):
            if included[i] and i != j:
                if torch.sum(torch.abs(weights[i] - weights[j])) < epsilon:
                    included[j] = False
                    
    return weights[torch.as_tensor(included)==True, :]
        
        
                
                
        
        
        
        
        
        
    
        
