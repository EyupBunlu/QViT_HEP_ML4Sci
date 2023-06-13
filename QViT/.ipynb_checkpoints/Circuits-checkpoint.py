import pennylane as qml
import torch
import numpy as np



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

    if type(wires)==type(None): wires = [ i for i in range(len(alphas)+1)]
    if is_x and not(is_conjugate):qml.PauliX(wires=wires[0])
    if is_conjugate:
        for i in reversed(range(len(alphas))):
            rbs([wires[i],wires[i+1]],-alphas[i])
    else: 
        for i,x in enumerate(alphas):
            rbs([wires[i],wires[i+1]],x)
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


def mmult(phi,alpha_x,wires=None):
    
    if type(wires)==type(None): wires = [ i for i in range(len(alphas)+1)]
    k=0
    length = len(alpha_x)+1
    
    if type(alpha_x)!='NoneType': vector_loader(alpha_x)
    for i in range(2*length-2):
        j = length-abs(length-1-i)
        
        if i%2: 
            for _ in range(j):
                if _%2==0:
                    rbs([_,_+1],phi[k])
                    k+=1
        else:
            for _ in range(j): 
                if _%2:
                    rbs([_,_+1],phi[k])
                    k+=1

# # # # # # # # # # Parametrization functions.

# Converts the array to the parameters
def convert_array(X):
    alphas = torch.zeros(X.shape[0]-1)
    temp = X/(X**2).sum().sqrt()
    for i,x in enumerate(temp[:-1]):
        if i==0:
            alphas[i] = torch.acos(x)
         
        else:

            alphas[i] = torch.acos(x/(1-(temp[:i]**2).sum()).sqrt() )
            if torch.isnan(alphas[i]): alphas[i] = torch.acos(torch.ones(1))
    return alphas


# Converts the matrix to the parameters
def convert_matrix(X):

    mag_alphas = convert_array( (X**2).sum(axis=1).sqrt() )
    
    alphas = torch.stack([convert_array(x) for x in X])

    return mag_alphas,alphas
