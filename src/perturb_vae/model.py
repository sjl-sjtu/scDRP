import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from .loss import ZINBLoss,HSICloss,klLoss_prior,conditional_HISCloss
from torch.nn import functional as F
from functorch import make_functional_with_buffers
import scipy as sp
# import ot

class Encoder(nn.Module):
    """
    Encoder network
    """
    def __init__(self, device, input_dim = 3000, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.2):
        """
        Args:
            input_dim: int, input dimension
            layer_dims: list of int, hidden layer dimensions
            latent_dim: int, latent dimension
            dropout_rate: float, dropout rate in MLP
        """
        super(Encoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # q(z|x,c)
        layers_zxy = []
        current_dim = input_dim
        for dim in layer_dims:
            layers_zxy.append(nn.Linear(current_dim, dim))
            layers_zxy.append(nn.BatchNorm1d(dim))
            layers_zxy.append(nn.LeakyReLU())
            layers_zxy.append(nn.Dropout(dropout_rate))
            current_dim = dim
        self.zxy_encoder = nn.Sequential(*layers_zxy)
        
        self.mu_layer = nn.Linear(layer_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(layer_dims[-1], latent_dim)
        
    def forward(self,x):
        """
        Args:
            x: torch.Tensor, input data (batch_size x input_dim)
        
        Returns:
            z: torch.Tensor, latent variable
            mu: torch.Tensor, mean of the latent variable
            logvar: torch.Tensor, log variance of the latent variable
        """
        h = self.zxy_encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick
        
        Args:
            mu: torch.Tensor, mean of the latent variable
            logvar: torch.Tensor, log variance of the latent variable
        
        Returns:
            z: torch.Tensor, latent variable
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class Decoder(nn.Module):
    """
    Decoder network
    """
    def __init__(self, device, input_dim = 3000, covariate_dim = 1, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.2, library_size_strategy="observed"):
        """
        Args:
            input_dim: int, input dimension
            covariate_dim: int, covariate dimension
            layer_dims: list of int, hidden layer dimensions
            latent_dim: int, latent dimension
            dropout_rate: float, dropout rate in MLP
        """
        super(Decoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.covariate_dim = covariate_dim
        self.library_size_strategy = library_size_strategy
        
        # p(x|z,c)
        layers_xz = []
        current_dim =  latent_dim + covariate_dim
        for dim in reversed(layer_dims):
            layers_xz.append(nn.Linear(current_dim, dim))
            layers_xz.append(nn.BatchNorm1d(dim))
            layers_xz.append(nn.LeakyReLU())
            layers_xz.append(nn.Dropout(dropout_rate))
            current_dim = dim
        
        self.decoder = nn.Sequential(*layers_xz)  
        if self.library_size_strategy == "original": 
            self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                            # nn.Softmax(dim=-1)
                                            nn.Softplus()
                                            )
        else:
            self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                        nn.Softmax(dim=-1)
                                        # nn.Softplus()
                                        )
        # gene-cell dispersion
        self.dispersion_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                              nn.Softplus())
        
        # gene dispersion
        self.dispersion = torch.nn.Parameter(torch.randn(input_dim))
        
        self.dropout_layer = nn.Sequential(
            nn.Linear(layer_dims[0], input_dim),
            nn.Sigmoid())
        # self.dropout_layer = nn.Linear(layer_dims[0], input_dim)
        
    def forward(self,z,c,dispersion_strategy = "gene"):
        """
        Args:
            z: torch.Tensor, latent variables (batch_size x latent_dim)
            c: torch.Tensor, covariate data (batch_size x covariate_dim)
            dispersion_strategy: str, strategy to specify dispersion factor (we have two options: gene-wise and gene-cell wise but currently use gene-wise only)
        
        Returns:
            rho: torch.Tensor, mean of the negative binomial distribution
            dispersion: torch.Tensor, dispersion of the negative binomial distribution
            pi: torch.Tensor, zero-inflation parameter
        """
        # z = torch.cat([zd,zi],dim=-1)
        if self.covariate_dim > 0:
            z = torch.cat([z, c],dim=-1)
        h = self.decoder(z)
        rho = self.mean_layer(h)  # proportion
        # # rho = rho * 10000 / rho.sum(dim=-1, keepdim=True) 
        # rho = rho * 10000
        if dispersion_strategy == "gene":
            dispersion = torch.exp(self.dispersion) # Ensure positive outputs # gene-wise
        elif dispersion_strategy == "gene-cell":
            dispersion = self.dispersion_layer(h) # gene-cell wise
        pi = self.dropout_layer(h) 
        return rho, dispersion, pi

class MSEDecoder(nn.Module):
    """
    Decoder network
    """
    def __init__(self, device, input_dim = 3000, covariate_dim = 1, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.2):
        """
        Args:
            input_dim: int, input dimension
            covariate_dim: int, covariate dimension
            layer_dims: list of int, hidden layer dimensions
            latent_dim: int, latent dimension
            dropout_rate: float, dropout rate in MLP
        """
        super(MSEDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.covariate_dim = covariate_dim
        
        # p(x|z,c)
        layers_xz = []
        current_dim =  latent_dim + covariate_dim
        for dim in reversed(layer_dims):
            layers_xz.append(nn.Linear(current_dim, dim))
            layers_xz.append(nn.BatchNorm1d(dim))
            layers_xz.append(nn.LeakyReLU())
            layers_xz.append(nn.Dropout(dropout_rate))
            current_dim = dim
        
        self.decoder = nn.Sequential(*layers_xz)    
        self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                        # nn.Softmax(dim=-1)
                                        nn.Softplus()
                                        )
        # # gene-cell dispersion
        # self.dispersion_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
        #                                       nn.Softplus())
        
        # # gene dispersion
        # self.dispersion = torch.nn.Parameter(torch.randn(input_dim))
        
        # self.dropout_layer = nn.Sequential(
        #     nn.Linear(layer_dims[0], input_dim),
        #     nn.Sigmoid())
        # self.dropout_layer = nn.Linear(layer_dims[0], input_dim)
        
    def forward(self,z,c,dispersion_strategy = "gene"):
        """
        Args:
            z: torch.Tensor, latent variable (batch_size x latent_dim)
            c: torch.Tensor, covariate data (batch_size x covariate_dim)
            dispersion_strategy: str, strategy to specify dispersion factor (we have two options: gene-wise and gene-cell wise but currently use gene-wise only)
        
        Returns:
            rho: torch.Tensor, mean of the negative binomial distribution
            dispersion: torch.Tensor, dispersion of the negative binomial distribution
            pi: torch.Tensor, zero-inflation parameter
        """
        # z = torch.cat([zd,zi],dim=-1)
        if self.covariate_dim > 0:
            z = torch.cat([z, c],dim=-1)
        h = self.decoder(z)
        rho = self.mean_layer(h)  # proportion
        # # # rho = rho * 10000 / rho.sum(dim=-1, keepdim=True) 
        # rho = rho * 10000
        # if dispersion_strategy == "gene":
        #     dispersion = torch.exp(self.dispersion) # Ensure positive outputs # gene-wise
        # elif dispersion_strategy == "gene-cell":
        #     dispersion = self.dispersion_layer(h) # gene-cell wise
        # pi = self.dropout_layer(h) 
        return rho #, dispersion, pi

class PerturbNet(nn.Module):
    """
    scPerturb model
    """
    def __init__(self, device, input_dim, covariate_dim = 1, celltype_num = 1, perturbation_num = 2, 
                 layer_dims=[500,100], 
                 latent_dep_dim=20, latent_ind_dim = 10, dropout_rate = 0.2, lambda_sparse = 0, l1_latent = 0,
                 beta = 10, count_data=False, encoder_covariates = False, library_size_strategy="observed",eps=1e-10):
        """
        Args:
            input_dim: int, input dimension
            covariate_dim: int, covariate dimension (default: 1)
            celltype_num: int, number of cell types (default: 1)
            perturbation_num: int, number of perturbations (default: 2)
            layer_dims: list of int, hidden layer dimensions (default: [500,100])
            latent_dep_dim: int, latent dimension for dependent variable (default: 20)
            latent_ind_dim: int, latent dimension for independent variable (default: 10)
            dropout_rate: float, dropout rate in MLP (default: 0.2)
            lambda_sparse: float, sparsity penalty (default: 0)
            beta: float, KL divergence weight (default: 10)
            encoder_covariates: boolean, whether to include covariates in encoders (default: False)
            eps: float, small value to prevent numerical instability (default: 1e-10)
        """
        super(PerturbNet, self).__init__()
        
        self.device = device
        self.eps = eps
        self.input_dim = input_dim
        self.latent_dep_dim = latent_dep_dim
        self.latent_ind_dim = latent_ind_dim
        self.covariate_dim = covariate_dim
        self.perturbation_num = perturbation_num
        self.lambda_sparse = lambda_sparse
        self.celltype_num = celltype_num
        self.beta = beta
        self.encoder_covariates = encoder_covariates
        self.count_data = count_data
        self.l1_latent = l1_latent
        self.library_size_strategy = library_size_strategy
        
        # + self.celltype_num + self.perturbation_num
        self.encoder_dep = Encoder(device, self.input_dim + self.celltype_num + self.perturbation_num + self.covariate_dim * self.encoder_covariates, 
                                   layer_dims, self.latent_dep_dim, dropout_rate)
        self.encoder_ind = Encoder(device, self.input_dim + self.celltype_num + self.covariate_dim * self.encoder_covariates, 
                                   layer_dims, self.latent_ind_dim, dropout_rate)
        if self.count_data:
            self.decoder = Decoder(device, self.input_dim, self.covariate_dim, layer_dims, 
                                self.latent_dep_dim + self.latent_ind_dim, dropout_rate,library_size_strategy=self.library_size_strategy)
        else:
            self.decoder = MSEDecoder(device, self.input_dim, self.covariate_dim, layer_dims, 
                                self.latent_dep_dim + self.latent_ind_dim, dropout_rate)
        
        self.prior_zd = nn.Sequential(
            nn.Linear(self.perturbation_num+self.celltype_num, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 2 * self.latent_dep_dim)  # Output prior mean and log variance
        )
        if self.celltype_num > 0:
            self.prior_zi = nn.Sequential(
                nn.Linear(celltype_num, 5),
                nn.LeakyReLU(),
                nn.Linear(5, 2 * self.latent_ind_dim)  # Output prior mean and log variance
            )
        
        self.m_d = nn.Parameter(torch.ones(self.latent_dep_dim, device=self.device))
        self.m_i = nn.Parameter(torch.ones(self.latent_ind_dim, device=self.device))
        
    def forward(self,x,u,t,c,train=True):
        """
        Args:
            x: torch.Tensor, input data (batch_size, input_dim)
            u: torch.Tensor, perturbation data (batch_size, perturbation_num)
            t: torch.Tensor, cell type data (batch_size, celltype_num)
            c: torch.Tensor, covariate data (batch_size, covariate_dim)
            train:
        
        Returns:
            z_d: torch.Tensor, latent representation that depends on perturbation, (batch_size, latent_dep_dim)
            z_i: torch.Tensor, latent representation that is independent from perturbation, (batch_size, latent_ind_dim)
            mu_d: torch.Tensor, mean of latent representation that depends on perturbation, (batch_size, latent_dep_dim)
            mu_i: torch.Tensor, mean of latent representation that is independent from perturbation, (batch_size, latent_ind_dim)
            rho: torch.Tensor, mean of the negative binomial distribution, (batch_size, input_dim)
            dispersion: torch.Tensor, dispersion of the negative binomial distribution, (input_dim,) when gene-wise
            pi: torch.Tensor, zero-inflation parameter, (batch_size, input_dim)
            s: torch.Tensor, library size, (batch_size, 1)
            loss: torch.Tensor, total loss
            loss_dict: dict, dictionary of losses
        """
        x_original = x
        if self.count_data:
            x = torch.log1p(x)
        
        # if t.shape[1]==1:
        #     t = F.one_hot(t.to(torch.int64),num_classes=self.celltype_num).squeeze().view(-1,self.celltype_num).float()
        #     t.requires_grad = True
        # if u.shape[1]==1:
        #     u = F.one_hot(u.to(torch.int64),num_classes=self.perturbation_num).squeeze().view(-1,self.perturbation_num).float()
        #     u.requires_grad = True
        
        if self.encoder_covariates:
            if self.celltype_num > 0:
                z_d, mu_d, logvar_d = self.encoder_dep(torch.cat([x,u,t,c],dim=-1))
                z_i, mu_i, logvar_i = self.encoder_ind(torch.cat([x,t,c],dim=-1))
            else:
                z_d, mu_d, logvar_d = self.encoder_dep(torch.cat([x,u,c],dim=-1))
                z_i, mu_i, logvar_i = self.encoder_ind(torch.cat([x,c],dim=-1))
        else:
            if self.celltype_num > 0:
                z_d, mu_d, logvar_d = self.encoder_dep(torch.cat([x,u,t],dim=-1))
                z_i, mu_i, logvar_i = self.encoder_ind(torch.cat([x,t],dim=-1))
            else:
                z_d, mu_d, logvar_d = self.encoder_dep(torch.cat([x,u],dim=-1))
                z_i, mu_i, logvar_i = self.encoder_ind(torch.cat([x],dim=-1))
                
        # m_d = self.gumbel_softmax(self.m_d, temperature=0.5)
        # m_i = self.gumbel_softmax(self.m_i, temperature=0.5)
        m_d = torch.sigmoid(5.0*self.m_d) #(torch.sigmoid(self.m_d) > 0.5).float()
        m_i = torch.sigmoid(5.0*self.m_i) #(torch.sigmoid(self.m_i) > 0.5).float()
        # z = torch.cat([z_d * self.m_d, z_i * self.m_i],dim=-1)
        if self.l1_latent > 0:
            mu_d, logvar_d = mu_d * m_d, logvar_d * m_d
            mu_i, logvar_i = mu_i * m_i, logvar_i * m_i
        z_d = self.reparameterize(mu_d, logvar_d)
        z_i = self.reparameterize(mu_i, logvar_i)
        # z_d = z_d * m_d
        # z_i = z_i * m_i
        z = torch.cat([z_d, z_i],dim=-1)
        
        if self.count_data:
            rho, dispersion, pi = self.decoder(z, c)
            s = self.sample_sequencing_depth(x_original, strategy=self.library_size_strategy) # library size
            # recon loss
            recon_loss = ZINBLoss()(x_original, rho, dispersion, pi, s, eps = self.eps)
        else:
            rho = self.decoder(z, c)
            # loss
            recon_loss = nn.MSELoss(reduction="sum")(x_original, rho)/x_original.shape[0]
        
        # prior of zd, zi
        if self.celltype_num > 0:
            prior_out_zd = self.prior_zd(torch.cat([u,t],dim=-1))
        else:
            prior_out_zd = self.prior_zd(u)
        mu_prior_zd, logvar_prior_zd = prior_out_zd.chunk(2, dim=-1)
        if self.celltype_num > 0:
            prior_out_zi = self.prior_zi(t)
            mu_prior_zi, logvar_prior_zi = prior_out_zi.chunk(2, dim=-1)
        else:
            mu_prior_zi, logvar_prior_zi = torch.zeros(x.shape[0],self.latent_ind_dim,device=self.device), torch.ones(x.shape[0],self.latent_ind_dim,device=self.device)
        # if self.celltype_num > 1:
        #     prior_out_zd = self.prior_zd(torch.cat([u,t],dim=-1))
        #     mu_prior_zd, logvar_prior_zd = prior_out_zd.chunk(2, dim=-1)
        #     prior_out_zi = self.prior_zi(t)
        #     mu_prior_zi, logvar_prior_zi = prior_out_zi.chunk(2, dim=-1)
        # else:
        #     prior_out_zd = self.prior_zd(u)
        #     mu_prior_zd, logvar_prior_zd = prior_out_zd.chunk(2, dim=-1)
        #     mu_prior_zi, logvar_prior_zi = torch.zeros(x.shape[0],self.latent_ind_dim,device=self.device), torch.ones(x.shape[0],self.latent_ind_dim,device=self.device)
        
        if train:
            # KL loss
            kl_loss = klLoss_prior(mu_d, logvar_d, mu_prior_zd, logvar_prior_zd) + klLoss_prior(mu_i, logvar_i, mu_prior_zi, logvar_prior_zi)
            ind_loss = conditional_HISCloss(z_d,z_i,t) # zd indep zi given t

            #sparsity on z
            # latent_l1 = torch.sum(torch.abs(torch.concat([mu_d,mu_i],dim=-1))) #torch.sum(torch.abs(z_d)) + torch.sum(torch.abs(z_i)) # 
            latent_l1 = torch.sum(torch.abs(torch.concat([m_d,m_i], dim=-1)))
            
            if self.lambda_sparse != 0:
                self.decoder.eval()
                params = {k: v for k, v in self.decoder.named_parameters()} 
                buffers = {k: v for k, v in self.decoder.named_buffers()}
                def fmodel(params, buffers, z, c): #functional version of model
                    # z_d = z_d.unsqueeze(0) if z_d.dim() == 1 else z_d
                    # z_i = z_i.unsqueeze(0) if z_i.dim() == 1 else z_i
                    if self.count_data:
                        z = z.unsqueeze(0) if z.dim() == 1 else z
                        if self.covariate_dim > 0:
                            c = c.unsqueeze(0).detach() if c.dim() == 1 else c.detach()
                        return torch.func.functional_call(self.decoder, (params, buffers), (z, c))[0]
                    else:
                        z = z.unsqueeze(0) if z.dim() == 1 else z
                        if self.covariate_dim > 0:
                            c = c.unsqueeze(0).detach() if c.dim() == 1 else c.detach()
                        return torch.func.functional_call(self.decoder, (params, buffers), (z, c))
                jacobian_z = torch.func.vmap(
                    torch.func.jacrev(fmodel, argnums=2), in_dims=(None, None, 0, 0) 
                )(params, buffers,z, c)
                jacobian_penalty = torch.sum(torch.abs(jacobian_z))/x.shape[0]
                self.decoder.train()
            else:
                jacobian_penalty = torch.tensor(0, device=self.device)
                
            # lambda_hsic = 1 #(recon_loss.detach().mean() + kl_loss.detach().mean()) / ind_loss.detach().mean()
            loss = recon_loss + self.beta * kl_loss + ind_loss + self.lambda_sparse * jacobian_penalty + self.l1_latent * latent_l1
            loss_dict = {'total_loss':loss.item(), 'recon_loss':recon_loss.item(), 'kl_loss':kl_loss.item(),
                        'ind_loss':ind_loss.item(), 'l1_norm': jacobian_penalty.item(), 'l1_latent': latent_l1.item()}
            if self.count_data:
                return z_d, z_i, mu_d, mu_i, logvar_d, logvar_i, rho, dispersion, pi, s, loss, loss_dict
            else:
                return z_d, z_i, mu_d, mu_i, logvar_d, logvar_i, rho, loss, loss_dict
        else:
            if self.count_data:
                return z_d, z_i, mu_d, mu_i, logvar_d, logvar_i, rho, dispersion, pi, s
            else:
                return z_d, z_i, mu_d, mu_i, logvar_d, logvar_i, rho
        
    def sample_sequencing_depth(self, x, strategy="observed"):
        """
        Sample sequencing depth
        
        Args:
            x: torch.Tensor, observed data
            strategy: str, strategy to sample sequencing depth. We have two options: batch_sample and observed, but will use observed only currently
        
        Returns:
            s: torch.Tensor, library size
        """
        if strategy=="batch_sample":
            # batch empirically sample
            mu_s = torch.log(x.sum(dim=-1) + 1.0).mean()
            sigma_s = torch.log(x.sum(dim=-1) + 1.0).std()
            log_s = mu_s + sigma_s * torch.randn_like(sigma_s)
            s = torch.exp(log_s).reshape(-1,1)
            # s = s.detach()
        elif strategy == "observed":
            log_s = torch.log(x.sum(dim=-1)).unsqueeze(1)
            s = torch.exp(log_s).reshape(-1,1)
            # s = s/10000
            # s = s.detach()
        elif strategy == "original":
            s = torch.ones(x.shape[0],device=x.device).reshape(-1,1)
        return s
    
    # def get_latent(self,x,u,c,m,std):
    #     z_d, z_i, _, _, _,_,_,_ = self.forward(x,u,c,m,std)
    #     return z_d, z_i
    
    # def get_rho(self,x,u,c,m,std):
    #     _,_,rho,_,_,_,_,_ = self.forward(x,u,c,m,std)
    #     return rho
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick
        
        Args:
            mu: torch.Tensor, mean of the latent variable
            logvar: torch.Tensor, log variance of the latent variable
        
        Returns:
            z: torch.Tensor, latent variable
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def gumbel_softmax(self, logits, temperature=1.0):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = logits + gumbel_noise
        return F.softmax(y / temperature, dim=-1)