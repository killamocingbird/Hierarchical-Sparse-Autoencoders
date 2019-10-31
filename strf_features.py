import torch
import numpy as np

def strf_features(strf):
    s = strf.shape[0]
# =============================================================================
#     u, _, v = np.linalg.svd(strf)
#     u = torch.as_tensor(u)
#     v = torch.as_tensor(v)
# =============================================================================
    u, _, v = torch.svd(strf)
    
    T, ind = torch.max(u[0, :], 0)
    Dura = 0
    while torch.norm(u[0, max(0, ind-Dura):min(ind+Dura, s)]) < 0.9 * torch.norm(u[0, :]):
        Dura = Dura + 1
    
    Dura = Dura * 2
    
    F, ind = torch.max(v[:, 0], 0)
    Band = 0
    while torch.norm(v[0, max(0, ind-Band):min(ind+Band, s)]) < 0.9 * torch.norm(v[0, :]):
        Band = Band + 1
    
    Band = Band * 2
    
    return [T, F, Dura, Band]


def gen_graphs(strfs):
    
    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt
    xlabels = ['Best T', 'Center F', 'Duration', 'Bandwidth']
    
    feats = []
    for i in range(strfs.shape[2]):
        feats.append(strf_features(strfs[:,:,i]))
        
    feats = np.array(feats)
    
    fig = plt.figure(figsize=(5,5))
    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.hist(feats[:,i], bins = 10)
        ax.set_title(xlabels[i])
        plt.axis('off')
    
    
# =============================================================================
#     lbls = dict()
#     for i in range(strfs.shape[2]):
#         anal = strf_features(strfs[:,:,i])[feat]
#         if anal in lbls:
#             lbls[anal] += 1
#         else:
#             lbls[anal] = 1
# #        print(i)
#         
#     minLbl = 1_000_000_000
#     maxLbl = -1
#     
#     for x in lbls:
#         minLbl = min(minLbl, x)
#         maxLbl = max(maxLbl, x)
#     
#     plotlbls = [i for i in range(minLbl, maxLbl + 1)]
#     plotdat = [0 for i in range(minLbl, maxLbl + 1)]
#     for x in lbls:
#         plotdat[x - minLbl] = lbls[x]
#     
# 
#     
#     plt.bar(plotlbls, plotdat, align='center', alpha=0.5)
#     plt.ylabel('Usage')
#     plt.title('Programming language usage')
#     
#     plt.show()
# =============================================================================

        