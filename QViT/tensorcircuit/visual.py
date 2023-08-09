import itertools
from pennylane import matrix
import torch
import numpy as np
import matplotlib.pyplot as  plt
def plot_opm(op,args,wire_order=None):
    m_ = matrix(op,wire_order=wire_order)(*args).real
    if torch.is_tensor(m_):
        m_=m_.detach().numpy()
    q_n = int( np.log2(m_.shape[0]))
    fig = plt.figure(figsize=[m_.shape[0]/5,m_.shape[0]/5])
    plt.imshow(m_ ,cmap='RdBu',vmin=-1,vmax=1)
    labels =tick_label_gen(q_n)
    labels_y= [ '|'+labels[l]+'>' if np.abs(m_[l,0])>.01 else '' for l in range(len(labels)) ]
    labels_x= [ '|'+labels[l]+'>' if (np.abs(m_[:,l])>.01).any() else '' for l in range(len(labels)) ]
    
    ticks = [i for i in range(2**q_n)]
    plt.xticks(ticks,labels_x ,rotation='vertical')
    plt.yticks(ticks,labels_y)
    plt.colorbar()
    
def tick_label_gen(n):
    return [str(i)[1:-1] for i in itertools.product(*[[0,1] for i in range(n)])]