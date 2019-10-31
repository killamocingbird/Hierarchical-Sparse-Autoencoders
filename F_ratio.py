import torch

def calc(layer_PSIs, groups=6):
    #Precalculation of averages
    av = torch.zeros(len(layer_PSIs[0][1]),)
    group_avs = [torch.zeros(len(av),) for i in range(groups)]
    group_nums = [0 for i in range(groups)]
    for i in range(len(layer_PSIs)):
        PSI = layer_PSIs[i]
        av += torch.as_tensor(PSI[1]).float()
        group_avs[PSI[0]] += torch.as_tensor(PSI[1]).float()
        group_nums[PSI[0]] += 1
    av /= len(layer_PSIs)
    for i in range(groups):
        group_avs[i] /= group_nums[i]
    
    #Calculation of F-ratio
    #Numerator
    num = 0
    #Denominator
    dem = 0
    
    for i in range(groups):
        
        num += group_nums[i] * L2(group_avs[i] - av)**2
        
        tempdem = 0
        for j in range(len(layer_PSIs)):
            if layer_PSIs[j][0] == i:
                tempdem += L2(torch.as_tensor(layer_PSIs[j][1]).float() - group_avs[i])**2
        dem += tempdem
        
    num /= (groups - 1)
    dem /= (len(layer_PSIs) - groups)
    
    return num / dem
        
def L2(x):
    return torch.sum(x**2)**(0.5)
    
    