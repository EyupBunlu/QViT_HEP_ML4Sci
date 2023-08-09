import torch
# Converts the array to the parameters, supports batch transforms



def convert_array(X):
    alphas = torch.zeros(*X.shape[:-1],X.shape[-1]-1)
    X_normd = X.clone()/(X**2).sum(axis=-1)[...,None].sqrt()
    for i in range(X.shape[-1]-1):
        if i==0:
            alphas[...,i] = torch.acos(X_normd[...,i])
         
        elif i<(X.shape[-1]-2):

            alphas[...,i] = torch.acos(X_normd[...,i]/torch.prod(torch.sin(alphas[...,:i]),dim=-1) )

        else:
            alphas[...,i] = torch.atan2(input=X_normd[...,-1],other=X_normd[...,-2] )
    return alphas
# Converts the matrix to the parameters, doesn't support batch transforms
def convert_matrix(X):

    mag_alphas = convert_array( (X**2).sum(axis=1).sqrt() )
    
    alphas = convert_array(X)

    return mag_alphas,alphas