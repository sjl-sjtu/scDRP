import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

class EarlyStopping:
    def __init__(self, patience=25, delta=0.0, verbose=False):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model, path="best_model.pt"):
        self.path = path
        # torch.save(model.state_dict(), self.path)
        # if self.verbose:
        #     print(f"Validation loss decreased, model saved to {self.path}")


def train_model(device, writer, train_dataset, valid_dataset, model, epoch_num, batch_size, 
                num_batch, lr=1e-6, accumulation_steps=1, adaptlr = True, count_data=False,
                early_stopping = True, patience=25):
    """
    Train the model
    
    Args:
        device: device to run the model
        writer: tensorboard writer
        train_dataset: training dataset
        model: model to train
        epoch_num: number of epochs
        batch_size: batch size
        num_batch: number of batches
        lr: learning rate
        accumulation_steps: number of steps to accumulate gradients
        adaptlr: whether to adapt learning rate
    """
    # load data
    train_data = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=True,num_workers=4,pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if adaptlr==True:
        scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =  epoch_num * num_batch)
    
    model = model.to(device)    
    if early_stopping:
        early_stopping = EarlyStopping(patience=patience, verbose=True) 
    for epoch in range(epoch_num):
        model.train()
        total_loss,recon_loss,kl_loss,ind_loss,l1_norm,l1_latent = 0,0,0,0,0,0
        # total_samples = 0
        for step, (x,u,t,c) in enumerate(train_data):
            model.train()
            x,u,t,c = x.to(device),u.to(device),t.to(device),c.to(device)
            x.requires_grad = True
            t.requires_grad = True
            c.requires_grad = True
            u.requires_grad = True
            # torch.autograd.set_detect_anomaly(True)
            if count_data:
                _,_,_,_, _, _, _, _, _, _, loss, loss_dict = model(x,u,t,c)
            else:
                _,_,_,_, _, _, _, loss, loss_dict = model(x,u,t,c)
            # with torch.autograd.detect_anomaly():
            #     loss.backward()
            loss.backward()  
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
            if (step + 1) % accumulation_steps == 0:
                optimizer.step() 
                optimizer.zero_grad()  
                
                if (writer is not None) & (adaptlr == True):
                    writer.add_scalar("lr/train",scheduler.get_last_lr()[0],epoch*num_batch+step)
                if adaptlr == True:    
                    scheduler.step()
            
            total_loss+=loss.item()
            recon_loss+=loss_dict['recon_loss']
            kl_loss+=loss_dict['kl_loss']
            ind_loss+=loss_dict['ind_loss']
            l1_norm+=loss_dict['l1_norm']
            l1_latent+=loss_dict['l1_latent']
            
            if writer is not None:
                writer.add_scalar("Loss/train", loss.item(), epoch*num_batch+step+1)
                writer.add_scalar("Recon_Loss/train", loss_dict['recon_loss'], epoch*num_batch+step+1)
                writer.add_scalar("KL_Loss/train", loss_dict['kl_loss'], epoch*num_batch+step+1)
                writer.add_scalar("Ind_Loss/train", loss_dict['ind_loss'], epoch*num_batch+step+1)
                writer.add_scalar("L1_norm/train", loss_dict['l1_norm'], epoch*num_batch+step+1)
                writer.add_scalar("L1_latent/train", loss_dict['l1_latent'], epoch*num_batch+step+1)
            
            # total_samples += x.size(0)
        if writer is not None:    
            writer.add_scalar("Loss_epoch/train", total_loss/num_batch, epoch+1)
            writer.add_scalar("Recon_Loss_epoch/train", recon_loss/num_batch, epoch+1)
            writer.add_scalar("KL_Loss_epoch/train", kl_loss/num_batch, epoch+1)
            writer.add_scalar("Ind_Loss_epoch/train", ind_loss/num_batch, epoch+1)
            writer.add_scalar("L1_norm_epoch/train", l1_norm/num_batch, epoch+1)
            writer.add_scalar("L1_latent_epoch/train", l1_latent/num_batch, epoch+1)

        if (epoch + 1) % 10 == 0:
            print("epoch {}: loss = {:.4f}, recon_loss = {:.4f}, kl_loss = {:.4f}, ind_loss = {:.4f}, l1_norm = {:.4f}, l1_latent = {:.4f}".format(
                epoch+1,total_loss/num_batch,recon_loss/num_batch,kl_loss/num_batch,ind_loss/num_batch,l1_norm/num_batch,l1_latent/num_batch))
        
        if early_stopping:
            validate_loss = validate_model(device, valid_dataset, model, batch_size, count_data)
            early_stopping(validate_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
    # zd,zi,rho = inference_model(device, train_dataset, model, batch_size)
    # return zd,zi,rho

def validate_model(device, validate_dataset, model, batch_size, count_data):
    model.eval()
    validate_data = DataLoader(validate_dataset,batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    total_loss= 0
    for _, (x,u,t,c) in enumerate(validate_data):
        x,u,t,c = x.to(device),u.to(device),t.to(device),c.to(device)
        model.eval()
        if count_data:
            _,_,_,_, _, _, _, _, _, _, loss, _ = model(x,u,t,c)
        else:
            _,_,_,_, _, _, _, loss, _ = model(x,u,t,c)
        total_loss+=loss.item()    
    return loss

def inference_model(device, infer_dataset, model, batch_size, count_data=False):
    """
    Do inference using the trained model
    
    Args:
        device: device to run the model
        infer_dataset: inference dataset
        model: trained model to inference
        batch_size: batch size
        
    Returns:
        zd: latent representation that depends on perturbation
        zi: latent representation that is independent from perturbation
        mu_d: mean of latent representation that depends on perturbation
        mu_i: mean of latent representation that is independent from perturbation
        rho: mean expression level
        dispersion: dispersion parameter
        pi: zero-inflation parameter
        s: library size
    """
    model = model.to(device) 
    model.eval()
    infer_data = DataLoader(infer_dataset,batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
    zd_list = []
    zi_list = []
    mud_list = []
    mui_list = []
    logvard_list = []
    logvari_list = []
    rho_list = []
    dispersion_list = []
    pi_list = []
    s_list = []
    if count_data:
        for _, (x,u,t,c) in enumerate(infer_data):
            x,u,t,c = x.to(device),u.to(device),t.to(device),c.to(device)
            model.eval()
            zd,zi,mu_d,mu_i,logvar_d,logvar_i,rho,dispersion, pi,s = model(x,u,t,c,train=False)
            
            zd_list.append(zd.detach().cpu().numpy())
            zi_list.append(zi.detach().cpu().numpy())
            mud_list.append(mu_d.detach().cpu().numpy())
            mui_list.append(mu_i.detach().cpu().numpy())
            logvard_list.append(logvar_d.detach().cpu().numpy())
            logvari_list.append(logvar_i.detach().cpu().numpy())
            rho_list.append(rho.detach().cpu().numpy())
            dispersion_list.append(dispersion.detach().cpu().numpy())
            pi_list.append(pi.detach().cpu().numpy())
            s_list.append(s.detach().cpu().numpy())
        zd = np.concatenate(zd_list, axis=0)
        zi = np.concatenate(zi_list, axis=0)
        mu_d = np.concatenate(mud_list, axis=0)
        mu_i = np.concatenate(mui_list, axis=0)
        logvar_d = np.concatenate(logvard_list, axis=0)
        logvar_i = np.concatenate(logvari_list, axis=0)
        rho = np.concatenate(rho_list, axis=0)
        dispersion = np.mean(dispersion_list, axis=0) # dispersion is gene-based
        pi = np.concatenate(pi_list, axis=0)
        library_size = np.concatenate(s_list, axis=0)
        return zd, zi, mu_d, mu_i, logvar_d, logvar_i, rho, dispersion, pi, library_size
    else:
        for _, (x,u,t,c) in enumerate(infer_data):
            x,u,t,c = x.to(device),u.to(device),t.to(device),c.to(device)
            model.eval()
            zd,zi,mu_d,mu_i,logvar_d,logvar_i,rho = model(x,u,t,c,train=False)
            
            zd_list.append(zd.detach().cpu().numpy())
            zi_list.append(zi.detach().cpu().numpy())
            mud_list.append(mu_d.detach().cpu().numpy())
            mui_list.append(mu_i.detach().cpu().numpy())
            logvard_list.append(logvar_d.detach().cpu().numpy())
            logvari_list.append(logvar_i.detach().cpu().numpy())
            rho_list.append(rho.detach().cpu().numpy())
        zd = np.concatenate(zd_list, axis=0)
        zi = np.concatenate(zi_list, axis=0)
        mu_d = np.concatenate(mud_list, axis=0)
        mu_i = np.concatenate(mui_list, axis=0)
        logvar_d = np.concatenate(logvard_list, axis=0)
        logvar_i = np.concatenate(logvari_list, axis=0)
        rho = np.concatenate(rho_list, axis=0)
        return zd, zi, mu_d, mu_i, logvar_d, logvar_i, rho
