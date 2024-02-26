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

def patcher_with_color(data,sh):
    r,c = sh

    rmax = (data.shape[-3]//r)
    cmax = (data.shape[-2]//c)

    patched = torch.empty(*data.shape[:-3],2*rmax*cmax,r*c,device=data.device).type(torch.float32)
    n=0
    for i in range(rmax):
        for j in range(cmax):
            for k in range(2):
                
                patched[...,n,:] = data[...,(i*r):(i*r+r),(j*c):(j*c+c),k].flatten(start_dim = -2)
                n+=1
    return patched