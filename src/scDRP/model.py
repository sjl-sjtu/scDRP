import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from .loss import *
from torch.nn import functional as F
from functorch import make_functional_with_buffers
import scipy as sp
# import ot

class HardConcreteGate(nn.Module):
    """
    Hard Concrete Gate for L0 regularization

    Args:
        size: int, size of the gate
        beta: float, temperature parameter. Default is 2/3
        gamma: float, left stretch parameter. Default is -0.1
        zeta: float, right stretch parameter. Default is 1.1
    """
    def __init__(self, size, beta=2/3, gamma=-0.1, zeta=1.1):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(size))  # learnable logit
        self.beta = beta # temperature
        self.gamma = gamma # left stretch (<0)
        self.zeta = zeta # right stretch (>1)

    def sample_gate(self, training=True):
        '''
        Sample gate values using Hard Concrete distribution

        Args:
            training: bool, whether in training mode

        Returns:
            gate: torch.Tensor, sampled gate values
        '''
        if training:
            u = torch.rand_like(self.log_alpha)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
            s = s * (self.zeta - self.gamma) + self.gamma # stretch to (gamma, zeta)
            gate = torch.clamp(s, 0, 1)
        else:
            # deterministic: use sigmoid(alpha) as probability
            s = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
            gate = torch.clamp(s, 0, 1)
        return gate

    def regularization_loss(self):
        '''
        Compute the expected L0 penalty

        Returns:
            prob_nonzero: a tensor representing the expected L0 penalty (prob of being nonzero)
        '''
        # expected L0 penalty (prob of being nonzero)
        prob_nonzero = torch.sigmoid(self.log_alpha - self.beta * np.log(-self.gamma / self.zeta))
        return prob_nonzero.sum()

    def forward(self, x, training=True):
        '''
        Forward pass with gating

        Args:
            x: input tensor
            training: whether the model is in training mode

        Returns:
            output: gated output tensor
            gate: sampled gate values
        '''
        gate = self.sample_gate(training=training)
        return x * gate, gate


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
        # z = torch.cat([zd,zu],dim=-1)
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

class NBDecoder(nn.Module):
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
        super(NBDecoder, self).__init__()
        
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
        
        # self.dropout_layer = nn.Sequential(
        #     nn.Linear(layer_dims[0], input_dim),
        #     nn.Sigmoid())
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
        # z = torch.cat([zd,zu],dim=-1)
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
        # pi = self.dropout_layer(h) 
        pi = torch.zeros_like(rho)
        return rho, dispersion, pi

class MSEDecoder(nn.Module):
    """
    Decoder network
    """
    def __init__(self, device, input_dim = 3000, covariate_dim = 1, layer_dims = [500,100], latent_dim = 20,
                 dropout_rate = 0.2, distribution = "Normal"):
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
        # self.positive_output = positive_output
        
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
        if distribution == "Normal_possitive":
            self.mean_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
                                            # nn.Softmax(dim=-1)
                                            nn.Softplus()
                                            )
        else:
            self.mean_layer = nn.Linear(layer_dims[0], input_dim)
        # # gene-cell dispersion
        # self.dispersion_layer = nn.Sequential(nn.Linear(layer_dims[0], input_dim),
        #                                       nn.Softplus())
        
        # # gene dispersion
        # self.dispersion = torch.nn.Parameter(torch.randn(input_dim))
        
        # self.dropout_layer = nn.Sequential(
        #     nn.Linear(layer_dims[0], input_dim),
        #     nn.Sigmoid())
        # self.dropout_layer = nn.Linear(layer_dims[0], input_dim)
        
    def forward(self, z, c, dispersion_strategy = "gene"):
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
        # z = torch.cat([zd,zu],dim=-1)
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
    def __init__(self, device, input_dim, covariate_dim = 0, celltype_num = 0, perturbation_num = 2, 
                 layer_dims=[500,100], 
                 latent_dep_dim = 50, latent_ind_dim = 50, dropout_rate = 0.2, lambda_sparse = 0, l0_latent = 0.001,
                 beta = 1, lambda_hsic = 0.2, distribution = "ZINB", #count_data=False, positive_output=True,
                 encoder_covariates = False, library_size_strategy="observed",eps=1e-10):
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
        # self.count_data = count_data
        self.l0_latent = l0_latent
        self.library_size_strategy = library_size_strategy
        # self.positive_output = positive_output
        self.distribution = distribution
        self.lambda_hsic = lambda_hsic
        
        # + self.celltype_num + self.perturbation_num
        self.encoder_dep = Encoder(device, self.input_dim + self.celltype_num + self.perturbation_num + self.covariate_dim * self.encoder_covariates, 
                                   layer_dims, self.latent_dep_dim, dropout_rate)
        self.encoder_ind = Encoder(device, self.input_dim + self.celltype_num + self.covariate_dim * self.encoder_covariates, 
                                   layer_dims, self.latent_ind_dim, dropout_rate)
        if self.distribution == "ZINB":
            self.decoder = Decoder(device, self.input_dim, self.covariate_dim, layer_dims, 
                                self.latent_dep_dim + self.latent_ind_dim, dropout_rate,library_size_strategy=self.library_size_strategy)
        elif self.distribution == "NB":
            self.decoder = NBDecoder(device, self.input_dim, self.covariate_dim, layer_dims,
                                self.latent_dep_dim + self.latent_ind_dim, dropout_rate,library_size_strategy=self.library_size_strategy)
        else:
            self.decoder = MSEDecoder(device, self.input_dim, self.covariate_dim, layer_dims, 
                                self.latent_dep_dim + self.latent_ind_dim, dropout_rate, self.distribution)
        
        self.prior_zd = nn.Sequential(
            nn.Linear(self.perturbation_num+self.celltype_num, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 2 * self.latent_dep_dim)  # Output prior mean and log variance
        )
        if self.celltype_num > 0:
            self.prior_zu = nn.Sequential(
                nn.Linear(celltype_num, 5),
                nn.LeakyReLU(),
                nn.Linear(5, 2 * self.latent_ind_dim)  # Output prior mean and log variance
            )
        
        # self.m_d = nn.Parameter(5.0*torch.ones(self.latent_dep_dim, device=self.device))
        # self.m_u = nn.Parameter(5.0*torch.ones(self.latent_ind_dim, device=self.device))
        # self.m_d = nn.Parameter(torch.zeros(self.latent_dep_dim, device=self.device))
        # self.m_u = nn.Parameter(torch.zeros(self.latent_ind_dim, device=self.device))

        self.gate_zd = HardConcreteGate(self.latent_dep_dim)
        self.gate_zu = HardConcreteGate(self.latent_ind_dim)
        
    def forward(self,x,a,t,c,train=True):
        """
        Args:
            x: torch.Tensor, input data (batch_size, input_dim)
            a: torch.Tensor, perturbation data (batch_size, perturbation_num)
            t: torch.Tensor, cell type data (batch_size, celltype_num)
            c: torch.Tensor, covariate data (batch_size, covariate_dim)
            train:
        
        Returns:
            z_d: torch.Tensor, latent representation that depends on perturbation, (batch_size, latent_dep_dim)
            z_u: torch.Tensor, latent representation that is independent from perturbation, (batch_size, latent_ind_dim)
            mu_d: torch.Tensor, mean of latent representation that depends on perturbation, (batch_size, latent_dep_dim)
            mu_u: torch.Tensor, mean of latent representation that is independent from perturbation, (batch_size, latent_ind_dim)
            rho: torch.Tensor, mean of the negative binomial distribution, (batch_size, input_dim)
            dispersion: torch.Tensor, dispersion of the negative binomial distribution, (input_dim,) when gene-wise
            pi: torch.Tensor, zero-inflation parameter, (batch_size, input_dim)
            s: torch.Tensor, library size, (batch_size, 1)
            loss: torch.Tensor, total loss
            loss_dict: dict, dictionary of losses
        """
        x_original = x
        if self.distribution in ["ZINB","NB"]: # self.count_data:
            x = torch.log1p(x)
        
        # if t.shape[1]==1:
        #     t = F.one_hot(t.to(torch.int64),num_classes=self.celltype_num).squeeze().view(-1,self.celltype_num).float()
        #     t.requires_grad = True
        # if u.shape[1]==1:
        #     u = F.one_hot(u.to(torch.int64),num_classes=self.perturbation_num).squeeze().view(-1,self.perturbation_num).float()
        #     u.requires_grad = True
        
        if self.encoder_covariates:
            if self.celltype_num > 0:
                z_d, mu_d, logvar_d = self.encoder_dep(torch.cat([x,a,t,c],dim=-1))
                z_u, mu_u, logvar_u = self.encoder_ind(torch.cat([x,t,c],dim=-1))
            else:
                z_d, mu_d, logvar_d = self.encoder_dep(torch.cat([x,a,c],dim=-1))
                z_u, mu_u, logvar_u = self.encoder_ind(torch.cat([x,c],dim=-1))
        else:
            if self.celltype_num > 0:
                z_d, mu_d, logvar_d = self.encoder_dep(torch.cat([x,a,t],dim=-1))
                z_u, mu_u, logvar_u = self.encoder_ind(torch.cat([x,t],dim=-1))
            else:
                z_d, mu_d, logvar_d = self.encoder_dep(torch.cat([x,a],dim=-1))
                z_u, mu_u, logvar_u = self.encoder_ind(torch.cat([x],dim=-1))
                
        # m_d = self.gumbel_softmax(self.m_d, temperature=0.5)
        # m_u = self.gumbel_softmax(self.m_u, temperature=0.5)
        # m_d = torch.sigmoid(self.m_d) # F.gumbel_softmax(self.m_d, tau=1.0, hard=False, dim=-1) #  #(torch.sigmoid(self.m_d) > 0.5).float()
        # m_u = torch.sigmoid(self.m_u) # F.gumbel_softmax(self.m_u, tau=1.0, hard=False, dim=-1) #  #(torch.sigmoid(self.m_u) > 0.5).float()
        # z = torch.cat([z_d * self.m_d, z_u * self.m_u],dim=-1)
        # if self.l0_latent > 0:
        #     mu_d, logvar_d = mu_d * m_d, logvar_d * m_d
        #     mu_u, logvar_u = mu_u * m_u, logvar_u * m_u
        z_d = self.reparameterize(mu_d, logvar_d)
        z_u = self.reparameterize(mu_u, logvar_u)
        if self.l0_latent > 0:
            # z_d = z_d * m_d
            # z_u = z_u * m_u
            z_d, _ = self.gate_zd(z_d, training=train)
            z_u, _ = self.gate_zu(z_u, training=train)
        z = torch.cat([z_d, z_u],dim=-1)
        
        if self.distribution == "ZINB":
            rho, dispersion, pi = self.decoder(z, c)
            s = self.sample_sequencing_depth(x_original, strategy=self.library_size_strategy) # library size
            # recon loss
            recon_loss = ZINBLoss()(x_original, rho, dispersion, pi, s, eps = self.eps)
        elif self.distribution == "NB":
            rho, dispersion, pi = self.decoder(z, c)
            s = self.sample_sequencing_depth(x_original, strategy=self.library_size_strategy) # library size
            # recon loss
            recon_loss = ZINBLoss()(x_original, rho, dispersion, pi, s, eps = self.eps)
        else:
            rho = self.decoder(z, c)
            # loss
            recon_loss = nn.MSELoss(reduction="sum")(x_original, rho)/x_original.shape[0]
        
        # prior of zd, zu
        if self.celltype_num > 0:
            prior_out_zd = self.prior_zd(torch.cat([a,t],dim=-1))
        else:
            prior_out_zd = self.prior_zd(a)
        mu_prior_zd, logvar_prior_zd = prior_out_zd.chunk(2, dim=-1)
        if self.celltype_num > 0:
            prior_out_zu = self.prior_zu(t)
            mu_prior_zu, logvar_prior_zu = prior_out_zu.chunk(2, dim=-1)
        else:
            mu_prior_zu, logvar_prior_zu = torch.zeros(x.shape[0],self.latent_ind_dim,device=self.device), torch.ones(x.shape[0],self.latent_ind_dim,device=self.device)
        # if self.celltype_num > 1:
        #     prior_out_zd = self.prior_zd(torch.cat([u,t],dim=-1))
        #     mu_prior_zd, logvar_prior_zd = prior_out_zd.chunk(2, dim=-1)
        #     prior_out_zu = self.prior_zu(t)
        #     mu_prior_zu, logvar_prior_zu = prior_out_zu.chunk(2, dim=-1)
        # else:
        #     prior_out_zd = self.prior_zd(u)
        #     mu_prior_zd, logvar_prior_zd = prior_out_zd.chunk(2, dim=-1)
        #     mu_prior_zu, logvar_prior_zu = torch.zeros(x.shape[0],self.latent_ind_dim,device=self.device), torch.ones(x.shape[0],self.latent_ind_dim,device=self.device)
        
        if train:
            # KL loss
            kl_loss = klLoss_prior(mu_d, logvar_d, mu_prior_zd, logvar_prior_zd) + klLoss_prior(mu_u, logvar_u, mu_prior_zu, logvar_prior_zu)
            
            # independence loss by HSIC
            # ind_loss = conditional_HISCloss(z_d,z_u,t) # zd indep zu given t
            if self.celltype_num == 0:
                ind_loss = HSICLoss()(z_d,z_u)
            else:
                ind_loss = UnnormalizedHSCICLoss()(z_d,z_u,t)

            # sparsity on dim(z)
            # latent_l1 = torch.sum(torch.abs(torch.concat([mu_d,mu_u],dim=-1))) #torch.sum(torch.abs(z_d)) + torch.sum(torch.abs(z_u)) # 
            # latent_l1 = torch.sum(torch.abs(torch.concat([m_d,m_u], dim=-1)))
            if self.l0_latent > 0:
                latent_l1 = self.gate_zd.regularization_loss() + self.gate_zu.regularization_loss()
            else:
                latent_l1 = torch.tensor(0, device=self.device)
            
            if self.lambda_sparse != 0:
                self.decoder.eval()
                params = {k: v for k, v in self.decoder.named_parameters()} 
                buffers = {k: v for k, v in self.decoder.named_buffers()}
                def fmodel(params, buffers, z, c): #functional version of model
                    # z_d = z_d.unsqueeze(0) if z_d.dim() == 1 else z_d
                    # z_u = z_u.unsqueeze(0) if z_u.dim() == 1 else z_u
                    if self.distribution in ["ZINB","NB"]:
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
                
            # lambda_hsic = 5 #0.01 * (recon_loss.detach().mean() + kl_loss.detach().mean()) / ind_loss.detach().mean()
            loss = recon_loss + self.beta * kl_loss  + self.lambda_sparse * jacobian_penalty + self.l0_latent * latent_l1  + self.lambda_hsic * ind_loss
            loss_dict = {'total_loss':loss.item(), 'recon_loss':recon_loss.item(), 'kl_loss':kl_loss.item(),
                        'ind_loss': ind_loss.item(),
                        'l1_norm': jacobian_penalty.item(), 'l0_latent': latent_l1.item()}
            if self.distribution in ["ZINB","NB"]:
                return z_d, z_u, mu_d, mu_u, logvar_d, logvar_u, rho, dispersion, pi, s, loss, loss_dict
            else:
                return z_d, z_u, mu_d, mu_u, logvar_d, logvar_u, rho, loss, loss_dict
        else:
            if self.distribution in ["ZINB","NB"]:
                return z_d, z_u, mu_d, mu_u, logvar_d, logvar_u, rho, dispersion, pi, s
            else:
                return z_d, z_u, mu_d, mu_u, logvar_d, logvar_u, rho
        
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
    #     z_d, z_u, _, _, _,_,_,_ = self.forward(x,u,c,m,std)
    #     return z_d, z_u
    
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
        
    # def gumbel_softmax(self, logits, temperature=1.0):
    #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    #     y = logits + gumbel_noise
    #     return F.softmax(y / temperature, dim=-1)