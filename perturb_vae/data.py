import torch
import numpy as np
import pandas as pd
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CombinedDataset(Dataset):
    '''
    Dataset for the combined data
    '''
    def __init__(self, X, u, t, c):
        '''
        Args:
            X: numpy array of shape (n_samples, n_features)
            u: one-hot encoded numpy array of shape (n_samples, num_perturbations)
            t: one-hot encoded numpy array of shape (n_samples, num_celltypes) or None
            c: one-hot encoded numpy array of shape (n_samples, batch_dim) or None
        '''
        super(CombinedDataset, self).__init__()
        self.X = torch.tensor(X).float() #.to(device)
        self.u = torch.tensor(u).float() #.to(device)
        if c is not None:
            self.c = torch.tensor(c).float() #.to(device)
        else:
            self.c = torch.zeros(X.shape[0]).float() #.to(device)
        if t is not None:
            self.t = torch.tensor(t).float() #.to(device)
        else:
            self.t = torch.zeros(X.shape[0]).float() #.to(device)
        self.len = X.shape[0]
        
        if self.c.ndimension() == 1:
            self.c = self.c.unsqueeze(1)
        if self.u.ndimension() == 1:
            self.u = self.u.unsqueeze(1)
        if self.t.ndimension() == 1:
            self.t = self.t.unsqueeze(1)
        
        # self.X.requires_grad = True
        # self.t.requires_grad = True
        # self.c.requires_grad = True
        # self.u.requires_grad = True
    
    def __len__(self):
        '''
        Returns:
            the number of samples in the dataset
        '''
        return self.len
    
    def __getitem__(self, index):
        '''
        Args:
            index
        
        Returns:
            the sample at the given index
        '''
        x_sample = self.X[index]
        u_sample = self.u[index]
        c_sample = self.c[index]
        t_sample = self.t[index]
        return x_sample,u_sample,t_sample,c_sample

class LatentDataset(Dataset):
    '''
    Dataset for the combined data
    '''
    def __init__(self, z, c):
        '''
        Args:
            z: numpy array of shape (n_samples, latemt_dim)
            c: one-hot encoded numpy array of shape (n_samples, batch_dim) or None
        '''
        super(LatentDataset, self).__init__()
        # self.zd = torch.tensor(zd).float() #.to(device)
        # self.zi = torch.tensor(zi).float()
        self.z = torch.tensor(z).float()
        if c is not None:
            self.c = torch.tensor(c).float() #.to(device)
        else:
            self.c = torch.zeros(z.shape[0]).float() #.to(device)
        self.len = z.shape[0]
        if self.c.ndimension() == 1:
            self.c = self.c.unsqueeze(1)
    
    def __len__(self):
        '''
        Returns:
            the number of samples in the dataset
        '''
        return self.len
    
    def __getitem__(self, index):
        '''
        Args:
            index
        
        Returns:
            the sample at the given index
        '''
        # zd_sample = self.zd[index]
        # zi_sample = self.zd[index]
        z_sample = self.z[index]
        c_sample = self.c[index]
        return z_sample,c_sample    

class LabelDataset(Dataset):
    '''
    Dataset for the combined data
    '''
    def __init__(self, u, t):
        super(LabelDataset, self).__init__()
        # self.zd = torch.tensor(zd).float() #.to(device)
        # self.zi = torch.tensor(zi).float()
        self.u = torch.tensor(u).float()
        if t is not None:
            self.t = torch.tensor(t).float() #.to(device)
        else:
            self.t = torch.zeros(u.shape[0]).float() #.to(device)
        self.len = u.shape[0]
        if self.t.ndimension() == 1:
            self.t = self.t.unsqueeze(1)
    
    def __len__(self):
        '''
        Returns:
            the number of samples in the dataset
        '''
        return self.len
    
    def __getitem__(self, index):
        '''
        Args:
            index
        
        Returns:
            the sample at the given index
        '''
        # zd_sample = self.zd[index]
        # zi_sample = self.zd[index]
        u_sample = self.u[index]
        t_sample = self.t[index]
        return u_sample,t_sample    
