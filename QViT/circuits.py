import torch.nn as nn
import numpy as np
import tensorcircuit as tc
import torch
from jax import pmap
from jax.numpy import array
####################################### Shared Func

# Wrapper

class QLayer(nn.Module):
    
    def circuit_to_func(self,K,func,nqubits):
        def f(inputs,phi):
            return func(inputs,phi,nqubits)

        f_vmap = K.vmap(f, vectorized_argnums=0)
        # f_pmap = pmap(f,in_axes=(0, None))


        f_batch = tc.interfaces.torch_interface(f_vmap, jit=True)

        
        return f_batch

    def __init__(self,func,par_sizes,nqubits):
        super(QLayer,self).__init__()
        self.K = tc.set_backend("jax")
        self.w = nn.Parameter(torch.normal(0,1/par_sizes[-1]**.5*torch.ones(par_sizes)) )
        self.f = self.circuit_to_func(self.K,func,nqubits)
    def forward(self,input1):
        return self.f(input1,self.w)



########################################### Circuits in the first method
# # # # # # Circuit Architectures given in the QViT Paper. Unutilized in the project but given for the completeness

def loader_bs(X):
    qml.PauliX(wires=0)
    for i,x in enumerate(X):
        # if X[i]!=X.max():
        qml.Beamsplitter(X[i]/X.max(),0,[i,i+1])
def mmult_bs(phi,X,length=3):

    k=0
    loader_bs(X)
    for i in range(2*length-2):
        j = length-abs(length-1-i)
        
        if i%2: 
            for _ in range(j):
                if _%2==0:
                    qml.Beamsplitter(phi[k], 0, [_,_+1])
                    k+=1
        else:
            for _ in range(j): 
                if _%2:
                    qml.Beamsplitter(phi[k], 0, [_,_+1])
                    k+=1
    return qml.expval(qml.PauliZ([1]))



# # # # # # # # # Circuit Architectures utilized in the project


def rbs(wires,th):
# Performs rbs operation described in the "" paper on wires with theta = th angles.
    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])
    qml.CZ(wires)
    qml.RY( th,wires[0])
    qml.RY(-th,wires[1])
    qml.CZ(wires)
    qml.Hadamard(wires[0])
    qml.Hadamard(wires[1])



def vector_loader(alphas,wires=None,is_x=True,is_conjugate=False):
# Loads the vector to the given wires
# alpha: the parametrized data. Parametrization can be achieved using convert_array function.
# wires: indicates which wires the operation is performed on. The default is [0,1,2,...,len(alpha)].
# is_x:  Whether to apply hadamard gate to the first gate or not. The default is True
# is_conjugate : is True if conjugate is being applied. The default is False

    if type(wires)==type(None): wires = [ i for i in range(alphas.shape[-1]+1)]
    if is_x and not(is_conjugate):qml.PauliX(wires=wires[0])
    if is_conjugate:
        for i in reversed(range(alphas.shape[-1])):
            rbs([wires[i],wires[i+1]],-alphas[...,i])
    else: 
        for i in range(alphas.shape[-1]):
            rbs([wires[i],wires[i+1]],alphas[...,i])
    if is_x and is_conjugate:qml.PauliX(wires=wires[0])


# Loads the matrix parametrized by mag_alphas and alphas to the mag_wires and wires.
def matrix_loader(mag_alphas,alphas,mag_wires,wires,is_conjugate=False):
    
# mag_alphas: The parametrized form of the rows norms of the matrix
# alphas: The parametrized form of the rows of the matrix
# mag_wires: The wires where norms of the rows are stored on.
# mag_wires: The wires where the rows are stored on.
    if not(is_conjugate):
        
        vector_loader(mag_alphas,mag_wires)
        for i in range(len(mag_wires)):
            qml.CNOT([mag_wires[i],wires[0]])
            vector_loader(alphas[i],wires,is_x=False)
            if i != len(mag_alphas):vector_loader(alphas[i+1],wires,is_x=False,is_conjugate=True)
    else:
        
        for i in reversed(range(len(mag_wires))):
            if i != len(mag_alphas):vector_loader(alphas[i+1],wires,is_x=False,is_conjugate=False)
            
            vector_loader(alphas[i],wires,is_x=False,is_conjugate=True)
            qml.CNOT([mag_wires[i],wires[0]])
            
        vector_loader(mag_alphas,mag_wires,is_conjugate=True)


def mmult(phi,wires=None,length=None):
    
    if type(length)==type(None): length = len(wires)
    if type(wires)==type(None): wires = [ i for i in range(length)]
    k=0

    for i in range(2*length-2):
        j = length-abs(length-1-i)
        
        if i%2: 
            for _ in range(j):
                if _%2==0:
                    rbs([wires[_],wires[_+1]],phi[k])
                    k+=1
        else:
            for _ in range(j): 
                if _%2:
                    rbs([wires[_],wires[_+1]],phi[k])
                    k+=1


def mmult_x(phi,wires=None,length=None):
    
    if type(length)==type(None): length = len(wires)
    if type(wires)==type(None): wires = [ i for i in range(length)]
    k=0

    for i in range(len(wires)-1):
        j = len(wires)-2-i
        
        if i==j:
            rbs([wires[j],wires[j+1]],phi[k])
            k+=1
        else:
            rbs([wires[i],wires[i+1]],phi[k])
            k+=1
            rbs([wires[j],wires[j+1]],phi[k])
            k+=1
                    
                    
def compute_attention_element(inputs,phi):
    alphas_i,alphas_j = torch.split(inputs,inputs.shape[-1]//2,dim=-1)
    wires = list(range(alphas_i.shape[-1]+2))
    qml.PauliX(wires[0])
    rbs(wires[:2],torch.pi/4)
    vector_loader(alphas_j,wires[1:],is_x=False)
    mmult(phi,wires=wires[1:])
    vector_loader(alphas_i,wires[1:],is_conjugate=True,is_x=False)
    rbs(wires[:2],torch.pi/4)
    return qml.expval(qml.PauliZ([wires[1]]))



def compute_attention(alphas,norms,compute_element):
    yhat=[]
    n=norms.shape[1]

    n_items = alphas.shape[0]
    
    for n_i in range(n_items):
                
        res= compute_element( torch.stack([alphas[n_i,[i,j]].flatten()   for j in range(n) for i in range(n)],dim=0)  )
        e1 = (-res.reshape(n,n)/2+1/2+1e-10).sqrt()
        wij = e1*2-1
        yhat.append(wij*torch.outer(norms[n_i],norms[n_i]) )
    yhat = torch.stack(yhat,dim=0)
    return yhat




################################################################################# Circuits used in the second method


def encode_token(c,data,nqubits):
    for i in range(nqubits):
        c.H(i)
        c.rx(i,theta = data[i])
        
        
# def qkv_ansatz(c,data,phi,nqubits):

#     for i in range(nqubits):
#         c.rx(i,theta=phi[0,i])
#     for i in range(nqubits):
#         c.ry(i,theta=phi[1,i])
#     for i in range(nqubits-1):
#         c.cnot(i,i+1)
        
        
        
# def qkv_ansatz(c,data,phi,nqubits):
#     for _ in range(0,phi.shape[0]//3,3)
#     for i in range(nqubits):
#         c.rx(i,theta=phi[_,i])
#     for i in range(nqubits):
#         c.ry(i,theta=phi[_+1,i])
        
#     for i in range(nqubits-1):
#         c.cnot(i,i+1)
#     c.cnot(0,nqubits-1)

def qk_ansatz(c,data,phi,nqubits):
    for i in range(nqubits):
        c.rx(i,theta=phi[i])
    for i in range(nqubits):
        c.ry(i,theta=phi[nqubits+i])

    for _ in range(2,phi.shape[0]//nqubits+1):


        for i in range(nqubits-1):
            c.cnot(i,i+1)
        c.cnot(nqubits-1,0)
        if _ != (phi.shape[0]//nqubits):
            for i in range(nqubits):
                c.ry(i,theta=phi[nqubits*(_)+i])
        else:
            
            c.ry(0,theta=phi[nqubits*(_)])
                
def v_ansatz(c,data,phi,nqubits):
    for i in range(nqubits):
        c.rx(i,theta=phi[i])
    for i in range(nqubits):
        c.ry(i,theta=phi[nqubits+i])

    for _ in range(2,phi.shape[0]//nqubits):


        for i in range(nqubits-1):
            c.cnot(i,i+1)
        c.cnot(nqubits-1,0)

        for i in range(nqubits):
            c.ry(i,theta=phi[nqubits*(_)+i])

        
def measure_query_key(data,phi,nqubits):
    c=tc.Circuit(nqubits)
    encode_token(c,data,nqubits)
    qk_ansatz(c,data,phi,nqubits)
    return (c.expectation_ps(z=[0]) ).real

def measure_value(data,phi,nqubits):
    c=tc.Circuit(nqubits)
    encode_token(c,data,nqubits)
    v_ansatz(c,data,phi,nqubits)
    return array([c.expectation_ps(z=[i]).real for i in range(nqubits)])