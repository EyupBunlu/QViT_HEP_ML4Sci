import torch
import torch.nn as nn
from math import sqrt
import math
from .parametrizations import convert_array
from .circuits import compute_attention
# from QViT.Parametrizations import convert_array

class AttentionHead(nn.Module):
    def __init__(self,d_t,d_k):
        super(AttentionHead,self).__init__()

        self.V = nn.Linear(d_k,d_k)
        self.norm = nn.LayerNorm(d_k)
        self.phi = nn.parameter.Parameter(torch.normal(0.,1/sqrt(d_t)*torch.ones( (d_t**2-d_t)//2)) )
        self.attention = lambda V,A : torch.bmm(nn.Softmax(dim=-1)(A.type(torch.float32)/math.sqrt(d_k)),V)

    def forward(self,input1):
        
        input2 = convert_array(input1)
        input3= ((input1)**2).sum(axis=-1).sqrt()

        V = self.V(input1)
        
        A = compute_attention(input2,self.phi,input3,wires=list(range(input1.shape[1])))
        
        
        return self.norm(self.attention(V,A)+input1)
        
class MultiHead(nn.Module):
    def __init__(self,d_t,d_k,n_h):
        super(MultiHead,self).__init__()
        self.heads =  nn.ModuleList(AttentionHead(d_t,d_k) for i in range(n_h))
        self.merger = nn.Linear(n_h,1)
    def forward(self,input1):
        res = torch.empty(*input1.shape,len(self.heads))
        for i,m in enumerate(self.heads):
            res[...,i] =m(input1)
        return self.merger(res).squeeze(-1)
class Transformer(nn.Module):
    def __init__(self,d_t,d_k,n_h,n_layers):
        super(Transformer,self).__init__()
        self.embedder = nn.LazyLinear(d_k)
        self.pos_embedding = torch.tensor([ [math.sin(i/1000**((j-1))/d_k) ] if i%2==1 else [math.cos(i/1000**((j-1))/d_k)] for i in range(d_t) for j in range(d_k)  ])
        self.attention_layers = nn.ModuleList( MultiHead(d_t+1,d_k,n_h) for i in range(n_layers))
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
    
    
class ViT(nn.Module):
    def __init__(self,d_t,d_k,n_h,n_layers,
                 FC_layers,FC_activation=nn.ReLU,FC_final_activation=nn.Softmax(dim=1)):
        super(ViT,self).__init__()
        self.transformer = Transformer(d_t,d_k,n_h,n_layers)
        layers = [ [nn.LazyLinear(par),FC_activation()] for i,par in enumerate(FC_layers) ]
        layers = [ l  for L in layers for l in L ]
        self.classifier = nn.Sequential(  *(layers+[FC_final_activation]) )
        
    def forward(self,input1):

        return self.classifier(self.transformer(input1))