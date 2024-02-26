from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import numpy as np
import torch
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
def patcher(data,sh):
    r,c = sh

    rmax = (data.shape[-2]//r)
    cmax = (data.shape[-1]//c)
    
    patched = torch.empty(*data.shape[:-2],rmax*cmax,r*c,device=data.device)
    for i in range(rmax):
        for j in range(cmax):
            patched[...,(i*cmax)+j,:] = data[...,(i*r):(i*r+r),(j*c):(j*c+c)].flatten(start_dim = -2)
    return patched
def patcher_with_color(data,sh):
    r,c = sh

    rmax = (data.shape[-3]//r)
    cmax = (data.shape[-2]//c)

    patched = torch.empty(*data.shape[:-3],rmax*cmax,r*c*2,device=data.device)
    for i in range(rmax):
        for j in range(cmax):
            patched[...,(i*cmax)+j,:] = data[...,(i*r):(i*r+r),(j*c):(j*c+c),:].flatten(start_dim = -3)
    return patched


def train(model,tr_dl,val_dl,loss_fn,optim,n_epochs,device='cuda'):
    try:
        min_loss = np.inf
        bar_epoch = tqdm(range(n_epochs))
        history = {'tr':[],'val':[],'tr_acc':[],'val_acc':[]}
        for epoch in bar_epoch:
            loss =0
            val_loss = 0
        
            total_samples = 0
            bar_batch = tqdm(tr_dl)
            model.train()
            pred_tr = []
            real_tr = []
            pred_val = []
            real_val = []
            for i in bar_batch:
                optim.zero_grad()
                yhat = model(i['input'].to(device))
                y = i['output']
                loss_ = loss_fn(yhat,y.to(device))
        
                loss_.sum().backward()
        
                optim.step()
                loss += loss_.sum().item()
                total_samples += y.shape[0]
                if len(yhat.shape)==1 or yhat.shape[-1]==1:
                    pred_tr.append((torch.sigmoid(yhat.detach())>.5).cpu())
                    real_tr.append(y.detach().cpu().unsqueeze(-1))
                else:
                    pred_tr.append(yhat.detach().argmax(axis=-1).cpu())
                    real_tr.append(y.detach().cpu())

                bar_batch.set_postfix_str(f'loss:{loss/total_samples}')
                            

            
            model.eval()
            for i in val_dl:
                with torch.no_grad():
                    yhat = model(i['input'].to(device))
                    y = i['output']
                    val_loss_ = loss_fn(yhat,y.to(device))
                    val_loss += val_loss_.sum().item()
                    if len(yhat.shape)==1 or yhat.shape[-1]==1:
                        pred_val.append((torch.sigmoid(yhat.detach())>.5).cpu())
                        real_val.append(y.detach().cpu().unsqueeze(-1))
                    else:
                        pred_val.append(yhat.detach().argmax(axis=-1).cpu())
                        real_val.append(y.detach().cpu())

            history['tr_acc'].append((torch.cat(pred_tr)==torch.cat(real_tr)).sum()/total_samples )
            history['val_acc'].append((torch.cat(pred_val)==torch.cat(real_val)).sum()/len(val_dl.dataset) )
            history['val'].append(val_loss/len(val_dl.dataset))
            history['tr'].append(loss/total_samples)
            bar_epoch.set_postfix_str(f'loss:{loss/total_samples}, v.loss:{val_loss/len(val_dl.dataset)},\
            tr_acc:{history["tr_acc"][-1] }, val_acc:{ history["val_acc"][-1] }')
            if history['val'][-1]<min_loss:
                min_loss = history['val'][-1]
                torch.save(model.state_dict(),'best_state_on_training_loss')
            if history['val_acc'][-1]==max(history['val_acc']):
                min_loss = history['val'][-1]
                torch.save(model.state_dict(),'best_state_on_training_acc')
            torch.save(history,'temp_history')
        return history
    except KeyboardInterrupt:
        return history