import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel

class ZINBLoss(nn.Module):
    '''
    Zero-Inflated Negative Binomial Loss
    '''
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, rho, dispersion, pi, s, eps=1e-8):
        '''
        Args:
            x: observed data
            rho: mean gene expression (mean of the negative binomial distribution equals rho * s)
            dispersion: dispersion of the negative binomial distribution
            pi: zero-inflation parameter
            s: scale parameter (library size)
            eps: small value to prevent numerical instability
        
        Returns:
            loss: negative log likelihood
        '''
        # P_NB(x; mu,r) = Gamma(x+r)/[Gamma(r)Gamma(x+1)] * [r/(r+mu)]^r * [mu/(r+mu)]^x
        # logP_NB(x) = logGamma(x+r) - logGamma(r) - logGamma(x+1) + rlog(r) - rlog(r+mu) + xlog(mu) - xlog(r+mu)
        # -logP_NB(x) = [-logGamma(x+r) + logGamma(r) + logGamma(x+1)] + [- rlog(r) - xlog(mu) + (r+x)log(r+mu)]
        mean = torch.clamp(rho * s, min=eps)
        dispersion = torch.clamp(dispersion, min=eps)
        
        # negative likelihood of NB
        # t1 = -logGamma(x+r) + logGamma(r) + logGamma(x+1)
        t1 = torch.lgamma(dispersion) + torch.lgamma(x + 1.0) - torch.lgamma(x + dispersion) 
        # t2 = - rlog(r) - xlog(mu) + (r+x)log(r+mu)
        t2 = - dispersion * torch.log(dispersion) - x * torch.log(mean) + (dispersion + x) * torch.log(dispersion + mean)
        nb_final = t1 + t2
        
        # zero-inflation
        zero_nb = torch.exp(dispersion * (torch.log(dispersion) - torch.log(dispersion + mean)))  # P_{NB}(x=0) = [r/(r+mu)]^r
        # zero_nb = torch.pow(dispersion / (dispersion + mean), dispersion) 
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)   # loss when x=0: -log[pi + (1-pi)*P_{NB}(x=0)]
        nb_case = nb_final - torch.log(1.0 - pi + eps)      # loss when x>0: -log[1-pi]-log[P_{NB}(x)]

        loss = torch.where(x <= eps, zero_case, nb_case)
        return torch.mean(torch.sum(loss,dim=1),dim=0)

def klLoss(mu, logvar):
    '''
    KL divergence between the latent distribution and the prior
    
    Args:
        mu: mean of the latent distribution
        logvar: log variance of the latent distribution
        Returns:
        kl: KL divergence
        
    Returns:
        kl: KL divergence
    '''
    kl = torch.mean(
        -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=1),dim=0)
    return kl

def klLoss_prior(mu_q, logvar_q, mu_p, logvar_p):
    """ 
    Compute KL(q || p) for two Gaussians q(z|x) ~ N(mu_p, exp(logvar_p)) and p(z) ~ N(mu_q, exp(logvar_q))
    
    Args:
        mu_q: mean of q
        logvar_q: log variance of q
        mu_p: mean of p
        logvar_p: log variance of p
    
    Returns:
        kl: KL divergence
    """
    # Compute KL divergence
    kl = -0.5 * torch.sum(
        - torch.exp(logvar_q - logvar_p)  # sigma_q^2 / sigma_p^2
        - ((mu_q - mu_p).pow(2)) * torch.exp(-logvar_p)  # (mu_q - mu_p)^2 / sigma_p^2
        + 1
        - logvar_p  # log(sigma_p^2)
        + logvar_q,  # - log(sigma_q^2)
        dim=1  # Sum over latent dimensions
    )
    return torch.mean(kl,dim=0)  # mean over batch 
    # return -0.5 * torch.sum(1 + logvar_q - logvar_p - 
    #                         (mu_q - mu_p).pow(2) / torch.exp(logvar_p) - 
    #                         torch.exp(logvar_q) / torch.exp(logvar_p))


def rbf_kernel(X, Y, sigma=1.0):
    """
    Compute the RBF (Gaussian) kernel matrix
    
    Args:
        X: input samples 2 (batch_size, dim)
        Y: input samples 2 (batch_size, dim)
        sigma: RBF kernel width
    
    Returns:
        K: RBF kernel matrix
    """
    pairwise_sq_dists = torch.cdist(X, Y, p=2) ** 2 
    K = torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))
    return K

def HSICloss(X, Y, sigma=1.0):
    """
    Compute Hilbert-Schmidt Independence Criterion (HSIC) between two sets of samples
    
    Args:
        X: input samples 1 (batch_size, dim)
        Y: input samples 2 (batch_size, dim)
        sigma: RBF kernel width
    
    Returns:
        HSIC: Hilbert-Schmidt Independence Criterion (smaller HSIC means more independent)

    """
    n = X.shape[0]
    K = rbf_kernel(X, X, sigma) #rbf_kernel(X, X, gamma=1 / (2 * sigma**2))
    L = rbf_kernel(Y, Y, sigma) #rbf_kernel(Y, Y, gamma=1 / (2 * sigma**2))
    H = torch.eye(n).to(X.device) - (1.0 / n) * torch.ones((n, n)).to(X.device)
    # HSIC: Tr(KHLH) / (n-1)^2
    # HSIC = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
    HSIC = torch.trace(K @ H @ L @ H)  / ((n - 1) ** 2)
    return HSIC

# def pairwise_distances(x):
#     #x should be two dimensional
#     instances_norm = torch.sum(x**2,-1).reshape((-1,1))
#     return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

# def GaussianKernelMatrix(x, sigma=1):
#     pairwise_distances_ = pairwise_distances(x)
#     return torch.exp(-pairwise_distances_ /sigma)

# def HSIC(x, y, s_x=1, s_y=1):
#     m,_ = x.shape #batch size
#     K = GaussianKernelMatrix(x,s_x)
#     L = GaussianKernelMatrix(y,s_y)
#     H = torch.eye(m) - 1.0/m * torch.ones((m,m))
#     H = H.double().cuda()
#     HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
#     return HSIC 

def center_kernel(K, Kz):
    """Centers the kernel matrix K with respect to the conditioning variable Z.

    Args:
        K (numpy.ndarray): Kernel matrix of shape (n, n).
        Kz (numpy.ndarray): Kernel matrix for Z of shape (n, n).

    Returns:
        numpy.ndarray: Centered kernel matrix of shape (n, n).
    """
    Kz_inv = torch.pinverse(Kz) #np.linalg.pinv(Kz)  # Compute the pseudo-inverse of Kz
    return K - Kz @ Kz_inv @ K

def conditional_HISCloss(X, Y, Z, sigma=1.0):
    """Computes the Conditional Hilbert-Schmidt Independence Criterion (CHSIC).

    Args:
        X (numpy.ndarray): Sample matrix for variable X, shape (n, d_x).
        Y (numpy.ndarray): Sample matrix for variable Y, shape (n, d_y).
        Z (numpy.ndarray): Sample matrix for conditioning variable Z, shape (n, d_z).
        sigma (float, optional): Bandwidth parameter for the RBF kernel. Defaults to 1.0.

    Returns:
        float: The conditional HSIC value, representing the dependency between X and Y given Z.
    """
    n = X.shape[0]

    # Step 1: Remove the influence of Z by computing the residuals
    # Fit a linear model (or another model) conditioned on Z for both X and Y.
    X_residual = X - torch.matmul(Z, torch.pinverse(Z) @ X)
    Y_residual = Y - torch.matmul(Z, torch.pinverse(Z) @ Y)

    # Step 2: Compute RBF kernels for the residuals of X and Y
    K = rbf_kernel(X_residual, X_residual, sigma)
    L = rbf_kernel(Y_residual, Y_residual, sigma)

    # Step 3: Centering matrix
    H = torch.eye(n).to(X.device) - (1.0 / n) * torch.ones((n, n)).to(X.device)

    # Step 4: Compute the conditional HSIC
    HSIC_conditioned = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
    return HSIC_conditioned

    # n = len(X)    
    # # Compute kernel matrices
    # Kx = rbf_kernel(X, X, sigma) #rbf_kernel(X, X, gamma=1 / (2 * sigma**2))
    # Ky = rbf_kernel(Y, Y, sigma) #rbf_kernel(Y, Y, gamma=1 / (2 * sigma**2))
    # Kz = rbf_kernel(Z, Z, sigma) #rbf_kernel(Z, Z, gamma=1 / (2 * sigma**2))
    # # Center kernel matrices
    # Kx_c = center_kernel(Kx, Kz)
    # Ky_c = center_kernel(Ky, Kz)
    # # Compute CHSIC
    # return torch.trace(Kx_c @ Ky_c) / (n - 1) ** 2  # Normalization
