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
def construct_FNN(layers,activation=nn.ReLU,output_activation=None,Dropout = None):
    layer = [j for i in layers for j in [nn.LazyLinear(i),activation()] ][:-1]
    if Dropout:
        layer.insert(len(layer)-2,nn.Dropout(Dropout))
    if output_activation is not None:
        layer.append(output_activation)
    return nn.Sequential(*layer)

class AttentionHead(nn.Module):
    def __init__(self,d_t,d_k):
        super(AttentionHead,self).__init__()
        self.Q = nn.Linear(d_k,d_k)
        self.V = nn.Linear(d_k,d_k)
        self.K = nn.Linear(d_k,d_k)
        
        self.soft = nn.Softmax(dim=-1)

    def attention(self,Q,K,V):
        return torch.bmm(self.soft(torch.bmm(Q,K.permute(0,2,1) )/math.sqrt(Q.shape[-1])),V)
    def forward(self,input1):

        Q = self.Q(input1)
        K = self.K(input1)
        V = self.V(input1)

        return self.attention(Q,K,V)

class MultiHead(nn.Module):
    def __init__(self,d_t,d_k,n_h,attention_type):
        super(MultiHead,self).__init__()
        self.d_h = d_k//n_h
        self.heads =  nn.ModuleList([AttentionHead(d_t,self.d_h) for i in range(n_h)])
    def forward(self,input1):

        
        return torch.cat(  [m(input1[...,(i*self.d_h):( (i+1)*self.d_h)]) for i,m in enumerate(self.heads)],dim=-1)

class EncoderLayer(nn.Module):
    def __init__(self,d_t,d_k,n_h,attention_type):
        super(EncoderLayer,self).__init__()
        self.norm1 = nn.LayerNorm([d_k])
        self.norm2 = nn.LayerNorm([d_k])
        self.MHA = MultiHead(d_t,d_k,n_h,attention_type)
        self.merger = construct_FNN([d_k,d_k],activation=nn.GELU)
    def forward(self,input1):
      
        input1_norm = self.norm1(input1)
        res = self.MHA(input1_norm)+input1

        return self.merger(self.norm2(res))+res
        
class Transformer(nn.Module):
    def __init__(self,d_t,d_k,n_h,n_layers,attention_type):
        super(Transformer,self).__init__()

        self.pos_embedding = nn.parameter.Parameter(torch.tensor( [ math.sin(1/10000**((i-1)/d_k))  if i%2==1 else math.cos(i/10000**((i-1)/d_k)) for i in range(d_k) ]))
        self.pos_embedding.requires_grad = False
        self.encoder_layers = nn.ModuleList([ EncoderLayer(d_t+1,d_k,n_h,attention_type) for i in range(n_layers)])
        self.class_token = nn.parameter.Parameter(torch.normal(torch.zeros(d_k),1/math.sqrt(d_k)))
        self.embedder = nn.Linear(d_k,d_k)

    def forward(self,input1):
        input1_ = torch.cat( (self.class_token.repeat(input1.shape[0],1,1), self.embedder(input1)+ self.pos_embedding[None,None,:]),axis=1)#

        # temp = torch.empty(len(self.encoder_layers)+1,*input1_.shape,device=input1.device)
        temp =[input1_]
        for i,m in enumerate(self.encoder_layers):temp.append( m(temp[i]))


        return temp[-1].flatten(start_dim=1)


class HViT(nn.Module):
    def __init__(self,d_t,d_k,n_h,n_layers,FC_layers,attention_type):
        super(HViT,self).__init__()
        self.transformer = Transformer(d_t,d_k,n_h,n_layers,attention_type)
        self.classifier = construct_FNN(FC_layers,activation=nn.LeakyReLU)
    def forward(self,input1):
        
        return self.classifier(self.transformer(input1))

def construct_FNN(layers,activation=nn.ReLU,output_activation=None,Dropout = None):
    layer = [j for i in layers for j in [nn.LazyLinear(i),activation()] ][:-1]
    if Dropout:
        layer.insert(len(layer)-2,nn.Dropout(Dropout))
    if output_activation is not None:
        layer.append(output_activation)
    return nn.Sequential(*layer)
