import torch
import torch.nn as nn
from math import sqrt
import math
from .parametrizations import convert_array
from .circuits import *
from torch.utils.data import Dataset
import warnings

#################### 1st Hybrid Approach
class AttentionHead_Hybrid1(nn.Module):
    def __init__(self,d_t,d_k):
        super(AttentionHead_Hybrid1,self).__init__()

        self.V = nn.Linear(d_k,d_k)
        self.norm = nn.LayerNorm(d_k,elementwise_affine=False)
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
class MultiHead_hybrid1(nn.Module):
    def __init__(self,d_t,d_k,n_h,attention_type):
        super(MultiHead_hybrid1,self).__init__()
        self.d_h = d_k//n_h
        self.heads =  nn.ModuleList([AttentionHead_Hybrid1(d_t,self.d_h) for i in range(n_h)])
    def forward(self,input1):

        
        return torch.cat(  [m(input1[...,(i*self.d_h):( (i+1)*self.d_h)]) for i,m in enumerate(self.heads)],dim=-1)

class EncoderLayer_hybrid1(nn.Module):
    def __init__(self,d_t,d_k,n_h,attention_type,embed_dim = None,ff_dim = None):
        super(EncoderLayer_hybrid1,self).__init__()
        self.norm1 = nn.LayerNorm([d_k],elementwise_affine=False)
        self.norm2 = nn.LayerNorm([d_k],elementwise_affine=False)
        self.MHA = MultiHead_hybrid1(d_t,d_k,n_h,attention_type)
        self.merger = construct_FNN([d_k,d_k],activation=nn.GELU)
    def forward(self,input1):
      
        input1_norm = self.norm1(input1)
        res = self.MHA(input1_norm)+input1

        return self.merger(self.norm2(res))+res        

#################### 2nd Hybrid Approach ######################
class EncoderLayer_hybrid2(nn.Module):
    def __init__(self,d_t,d_k,n_h,attention_type,embed_dim,ff_dim=None):
        if ff_dim is not None:warnings.warn("ff_dim is not utilized since no ff")
        super(EncoderLayer_hybrid2,self).__init__()
        self.d_h = d_k//n_h
        self.heads =  nn.ModuleList([AttentionHead_Hybrid2(d_t,self.d_h) for i in range(n_h)]) 
        self.merger = nn.Linear(n_h,n_h)
        self.norm1 = nn.LayerNorm([d_k],elementwise_affine=False)
    def forward(self,input1):
        
        input1_norm = self.norm1(input1)
        head_result = torch.stack(  [m(input1_norm[...,(i*self.d_h):( (i+1)*self.d_h)]) for i,m in enumerate(self.heads)],dim=-1)
        res = self.merger(head_result).flatten(start_dim=-2)+input1
        return res


        
        

class AttentionHead_Hybrid2(nn.Module):
    def __init__(self,d_t,d_k):
        super(AttentionHead_Hybrid2,self).__init__()

        self.d_k = d_k
        
        
        self.norm = nn.LayerNorm(d_k)

        self.V = QLayer(measure_value,[3*d_k],int(d_k))
        self.Q = QLayer(measure_query_key,[3*d_k+1],int(d_k))
        self.K = QLayer(measure_query_key,[3*d_k+1],int(d_k))
        
        self.attention = lambda A,V : torch.bmm(nn.Softmax(dim=-1)(A/d_k**.5),V)
        self.flattener = lambda A: A.flatten(0,1)

    def forward(self,input1):

        flat_input  = self.flattener(input1)

        V = self.V(flat_input).reshape(input1.shape).type(input1.dtype)
        Q = self.Q(flat_input).reshape(*input1.shape[:2])
        K = self.K(flat_input).reshape(*input1.shape[:2])
        A = torch.empty((*input1.shape[:-1],input1.shape[-2]),device=input1.device)
        for j in range(input1.shape[-2]):
            A[...,j] = -(Q-K[...,j][...,None])**2
        return self.attention(A,V)
        
        

#################### Classical Approach ######################
class AttentionHead(nn.Module):
    def __init__(self,d_t,embed_per_head_dim):
        super(AttentionHead,self).__init__()
        self.Q = nn.Linear(embed_per_head_dim,embed_per_head_dim)
        self.V = nn.Linear(embed_per_head_dim,embed_per_head_dim)
        self.K = nn.Linear(embed_per_head_dim,embed_per_head_dim)
        
        self.soft = nn.Softmax(dim=-1)

    def attention(self,Q,K,V):
        return torch.bmm(self.soft(torch.bmm(Q,K.permute(0,2,1) )/math.sqrt(Q.shape[-1])),V)
    def forward(self,input1):

        Q = self.Q(input1)
        K = self.K(input1)
        V = self.V(input1)

        return self.attention(Q,K,V)

class MultiHead(nn.Module):
    def __init__(self,d_t,embed_dim,n_h,attention_type):
        super(MultiHead,self).__init__()
        self.d_h = embed_dim//n_h
        self.heads =  nn.ModuleList([AttentionHead(d_t,self.d_h) for i in range(n_h)])
    def forward(self,input1):

        
        return torch.cat(  [m(input1[...,(i*self.d_h):( (i+1)*self.d_h)]) for i,m in enumerate(self.heads)],dim=-1)

class EncoderLayer(nn.Module):
    def __init__(self,d_t,embed_dim,n_h,attention_type,ff_dim):
        super(EncoderLayer,self).__init__()
        self.norm1 = nn.LayerNorm([embed_dim],elementwise_affine=False)
        self.norm2 = nn.LayerNorm([embed_dim],elementwise_affine=False)
        self.MHA = MultiHead(d_t,embed_dim,n_h,attention_type)
        self.merger = construct_FNN([ff_dim,embed_dim],activation=nn.GELU)
    def forward(self,input1):
      
        input1_norm = self.norm1(input1)
        res = self.MHA(input1_norm)+input1

        return self.merger(self.norm2(res))+res



#
############################### Shared Functions for all transformer architectures used here ####################
class Transformer(nn.Module):
    def __init__(self,d_t,d_k,n_h,n_layers,embed_dim,ff_dim,pos_embedding,classifying_type,attention_type):
        super(Transformer,self).__init__()
        self.cls_type = classifying_type
        self.embedding = pos_embedding
        
        self.pos_embedding = nn.parameter.Parameter(torch.tensor( [ math.sin(1/10000**((i-1)/d_k))  if i%2==1 else math.cos(i/10000**((i-1)/d_k)) for i in range(embed_dim) ]))
        self.pos_embedding.requires_grad = False
        attention_dict={'hybrid2':EncoderLayer_hybrid2,'classic':EncoderLayer,'hybrid1':EncoderLayer_hybrid1}
        if self.cls_type=='cls_token':
            self.encoder_layers = nn.ModuleList([ attention_dict[attention_type](d_t+1,embed_dim,n_h,attention_type,ff_dim) for i in range(n_layers)])
        else: self.encoder_layers = nn.ModuleList([ attention_dict[attention_type](d_t,embed_dim,n_h,attention_type,ff_dim) for i in range(n_layers)])
        if self.cls_type == "cls_token":self.class_token = nn.parameter.Parameter(torch.rand(embed_dim)/math.sqrt(embed_dim))
        
        self.embedder = nn.Linear(d_k,embed_dim)
        if self.cls_type == 'max':
            self.final_act =  lambda temp: temp[-1].max(axis=1).values
        if self.cls_type == 'mean':
            self.final_act =  lambda temp: temp[-1].mean(axis=1)
        if self.cls_type == 'sum':
            self.final_act =  lambda temp:temp[-1].sum(axis=1)
        if self.cls_type == 'cls_token':
            self.final_act =  lambda temp:temp[-1][:,0]

    def forward(self,input1):
        if self.cls_type == "cls_token": 
            input1_ = torch.cat( (self.class_token.repeat(input1.shape[0],1,1), self.embedder(input1)),axis=1)
        else: 
            input1_ = self.embedder(input1)
        if self.embedding: temp =[input1_+self.pos_embedding[None,None,:]]
        else: temp = [input1_]
        
        
        for i,m in enumerate(self.encoder_layers):temp.append( m(temp[i]))

        return self.final_act(temp)

class HViT(nn.Module):
    def __init__(self,d_t,d_k,n_h,n_layers,FC_layers,attention_type,pos_embedding,classifying_type,embed_dim,ff_dim):
        super(HViT,self).__init__()
        self.transformer = Transformer(d_t,d_k,n_h,n_layers,embed_dim,ff_dim,pos_embedding,classifying_type,attention_type)
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
