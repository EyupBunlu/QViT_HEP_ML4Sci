import torch
import torch.nn as nn
from math import sqrt
import math
from .parametrizations import convert_array
from .circuits import *
from torch.utils.data import Dataset

class simple_dataset(Dataset):
    def __init__(self,data,target,transform=None):
        self.data = data
        self.target = target
        self.len = self.data.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self,idx):
        sample = {'input':self.data[idx],'output':self.target[idx]}
        return sample



class AttentionHead(nn.Module):
    def __init__(self,d_t,d_k):
        super(AttentionHead,self).__init__()
        self.Q = nn.Linear(d_k,d_k)
        self.V = nn.Linear(d_k,d_k)
        self.K = nn.Linear(d_k,d_k)
        self.norm = nn.LayerNorm(d_k)
        self.attention = lambda Q,V,K : torch.bmm(nn.Softmax(dim=-1)(torch.bmm(Q,torch.transpose(K,1,2) )/math.sqrt(d_k)),V)
    def forward(self,input1):

        return self.norm(self.attention(self.Q(input1),self.V(input1),self.K(input1))+input1)



class AttentionHead_Hybrid1(nn.Module):
    def __init__(self,d_t,d_k):
        super(AttentionHead_Hybrid1,self).__init__()

        self.V = nn.Linear(d_k,d_k)
        self.norm = nn.LayerNorm(d_k)
        # len_phi = (d_k**2-d_k)//2
        len_phi = 2*d_k-3
        self.A = circuit_to_layer(compute_attention_element,list(range(d_k)),{'phi':len_phi})
        self.attention = lambda V,A : torch.bmm(nn.Softmax(dim=-1)(A/math.sqrt(d_k)),V)

    def forward(self,input1):

        input2 = convert_array(input1)
        input3= ((input1)**2).sum(axis=-1).sqrt()

        V = self.V(input1)

        A = compute_attention(input2,input3,self.A)

        return self.norm(self.attention(V,A)+input1)

class AttentionHead_Hybrid2(nn.Module):
    def __init__(self,d_t,d_k):
        super(AttentionHead_Hybrid2,self).__init__()

        self.V = nn.Linear(d_k,d_k)
        self.norm = nn.LayerNorm(d_k)
        wires = list(range(d_k))
        self.Q = circuit_to_layer(measure_query_key,wires=wires,pars={'phi': d_k*4})
        self.K = circuit_to_layer(measure_query_key,wires=wires,pars={'phi': d_k*4})
        
        self.attention = lambda A,V : torch.bmm(nn.Softmax(dim=-1)(A/math.sqrt(d_k)),V)

    def forward(self,input1):

        V = self.V(input1)

        Q = self.Q(input1)
        K = self.K(input1)
        A = torch.empty((*input1.shape[:-1],input1.shape[-2]),device=input1.device)
        for j in range(input1.shape[-2]):

            A[...,j] = torch.exp(-(Q-K[...,j][...,None])**2)
        return self.norm(self.attention(A,V)+input1)    

class MultiHead(nn.Module):
    def __init__(self,d_t,d_k,n_h,attention_type):
        super(MultiHead,self).__init__()
        attention_dict = {'classic':AttentionHead,'hybrid1':AttentionHead_Hybrid1,'hybrid2':AttentionHead_Hybrid2}
    
        self.heads =  nn.ModuleList(attention_dict[attention_type](d_t,d_k) for i in range(n_h))

        self.merger = nn.Linear(n_h,1)

    def forward(self,input1):
        res = torch.empty(*input1.shape,len(self.heads),device = self.merger.weight.device)
        for i,m in enumerate(self.heads):
            res[...,i] =m(input1)
        return self.merger(res).squeeze(-1)


class Transformer(nn.Module):
    def __init__(self,d_t,d_k,n_h,n_layers,attention_type):
        super(Transformer,self).__init__()
        self.embedder = nn.LazyLinear(d_k)
        self.pos_embedding = torch.tensor([ [math.sin(i/1000**((j-1))/d_k) ] if i%2==1 else [math.cos(i/1000**((j-1))/d_k)] for i in range(d_t) for j in range(d_k)  ])
        self.attention_layers = nn.ModuleList( MultiHead(d_t+1,d_k,n_h,attention_type) for i in range(n_layers))
        self.class_token = nn.parameter.Parameter(torch.normal(torch.zeros(d_k),1/d_k))
        self.norm = nn.LayerNorm(d_k)
        
    def forward(self,input1):

        input1_ = torch.cat( (self.class_token.broadcast_to([input1.shape[0],1,self.class_token.shape[0]]), self.embedder(input1)),axis=1)
        + self.pos_embedding[None,:,:]
        input1__ = self.norm(input1_)

        temp = []
        for i,m in enumerate(self.attention_layers):
            if i!=0: temp.append(m(temp[-1]))
            else: temp.append(m(input1__))

        return temp[-1][:,-1,:]


class HViT(nn.Module):
    def __init__(self,d_t,d_k,n_h,n_layers,FC_layers,attention_type):
        super(HViT,self).__init__()
        self.transformer = Transformer(d_t,d_k,n_h,n_layers,attention_type)
        self.classifier = construct_FNN(FC_layers)
    def forward(self,input1):

        return self.classifier(self.transformer(input1))


def construct_FNN(layers,activation=nn.ReLU,output_activation=None,Dropout = None):
    layer = [j for i in layers for j in [nn.LazyLinear(i),activation()] ][:-1]
    if Dropout:
        layer.insert(len(layer)-2,nn.Dropout(Dropout))
    if output_activation is not None:
        layer.append(output_activation)
    return nn.Sequential(*layer)
