import torch
# Converts the array to the parameters, supports batch transforms
def convert_array(X):
    alphas = torch.zeros(*X.shape[:-1],X.shape[-1]-1)
    temp = X/(X**2).sum(axis=-1)[...,None].sqrt()
    for i in range(X.shape[-1]-1):
        if i==0:
            alphas[...,i] = torch.acos(X[...,i])
         
        else:

            alphas[...,i] = torch.acos(X[...,i]/(1-(X[...,:i]**2).sum(axis=-1)[...,None]).sqrt() )
            alphas[...,i] = torch.where(torch.isnan(alphas[...,i]),torch.acos(torch.ones(1)),alphas[...,i] )
    return alphas


# Converts the matrix to the parameters, doesn't support batch transforms
def convert_matrix(X):

    mag_alphas = convert_array( (X**2).sum(axis=1).sqrt() )
    
    alphas = torch.stack([convert_array(x) for x in X])

    return mag_alphas,alphas