import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import rbf_kernel
from typing import Optional, Union, Literal

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




class HSICLoss(nn.Module):
    """Hilbert-Schmidt Independence Criterion (HSIC) for Independence Testing.
    
    Measures the dependence between X and Y. HSIC is zero if and only if X and Y
    are independent (assuming characteristic kernels like Gaussian RBF).
    
    Based on: Gretton et al. "Measuring Statistical Dependence with Hilbert-Schmidt Norms" (2005)
    
    Args:
        sigma_x (float, optional): Fixed bandwidth for X. If None, uses automatic selection.
        sigma_y (float, optional): Fixed bandwidth for Y. If None, uses automatic selection.
        sigma_method (str): Method for automatic sigma selection when sigma_x/sigma_y is None.
            - 'median': Median heuristic (default, recommended)
            - 'scott': Scott's rule
        sigma_scale (float): Additional scaling factor for computed sigma. Default: 1.0
    
    Example:
        >>> # Automatic sigma selection (recommended)
        >>> loss_fn = HSICLoss()
        >>> X = torch.randn(100, 5)
        >>> Y = torch.randn(100, 3)
        >>> loss = loss_fn(X, Y)
        >>> 
        >>> # Manual sigma specification
        >>> loss_fn = HSICLoss(sigma_x=1.0, sigma_y=0.5)
        >>> loss = loss_fn(X, Y)
    """
    
    def __init__(
        self,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_method: Literal['median', 'scott'] = 'median',
        sigma_scale: float = 1.0
    ):
        """Initializes the HSIC loss.
        
        Args:
            sigma_x: Fixed bandwidth for X. If None, auto-computed per forward pass.
            sigma_y: Fixed bandwidth for Y. If None, auto-computed per forward pass.
            sigma_method: Method for automatic bandwidth selection.
            sigma_scale: Global scaling factor applied to all computed sigmas.
        """
        super(HSICLoss, self).__init__()
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_method = sigma_method
        self.sigma_scale = sigma_scale

    def _median_heuristic(self, x: torch.Tensor) -> float:
        """Computes sigma using median heuristic.
        
        sigma^2 = median(||x_i - x_j||^2) * sigma_scale
        
        Args:
            x: Tensor of shape [batch_size, dim].
        
        Returns:
            Optimal sigma value.
        """
        n = x.shape[0]
        
        # For efficiency, subsample if dataset is large
        if n > 1000:
            idx = torch.randperm(n, device=x.device)[:1000]
            x = x[idx]
            n = 1000
        
        dist_sq = torch.cdist(x, x, p=2) ** 2
        
        # Extract upper triangular distances (excluding diagonal)
        triu_indices = torch.triu_indices(n, n, offset=1, device=x.device)
        distances = dist_sq[triu_indices[0], triu_indices[1]]
        
        if distances.numel() == 0:
            return 1.0
        
        median_dist_sq = torch.median(distances)
        sigma = torch.sqrt(torch.clamp(median_dist_sq, min=1e-6))
        
        return sigma.item() * self.sigma_scale

    def _scotts_rule(self, x: torch.Tensor) -> float:
        """Computes sigma using Scott's rule.
        
        sigma = n^(-1/(d+4)) * std(X) * sigma_scale
        
        Args:
            x: Tensor of shape [batch_size, dim].
        
        Returns:
            Optimal sigma value.
        """
        n, d = x.shape
        std = x.std(dim=0).mean()
        sigma = n ** (-1.0 / (d + 4)) * std
        return max(sigma.item(), 1e-6) * self.sigma_scale

    def _compute_sigma(self, x: torch.Tensor, fixed_sigma: Optional[float]) -> float:
        """Computes or returns sigma for a given tensor.
        
        Args:
            x: Input tensor.
            fixed_sigma: Pre-specified sigma value. If None, auto-compute.
        
        Returns:
            Sigma value to use.
        """
        if fixed_sigma is not None:
            return fixed_sigma
        
        if self.sigma_method == 'median':
            return self._median_heuristic(x)
        elif self.sigma_method == 'scott':
            return self._scotts_rule(x)
        else:
            raise ValueError(f"Unknown sigma_method: {self.sigma_method}")

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Computes Gaussian RBF kernel matrix.
        
        K(x, y) = exp(-||x - y||^2 / (2sigma^2))
        
        Args:
            x: Tensor of shape [batch_size, dim_x].
            y: Tensor of shape [batch_size, dim_y].
            sigma: Bandwidth parameter.
        
        Returns:
            Kernel matrix of shape [batch_size, batch_size].
        """
        dist_sq = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-0.5 * dist_sq / (sigma ** 2))

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Computes HSIC between X and Y.
        
        Args:
            X: Input tensor of shape [batch_size, dim_x].
            Y: Input tensor of shape [batch_size, dim_y].
        
        Returns:
            HSIC value (scalar). Zero indicates independence.
        
        Raises:
            ValueError: If batch sizes don't match.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Batch size mismatch: X={X.shape[0]}, Y={Y.shape[0]}")
        
        n = X.shape[0]
        
        # Compute or use fixed sigmas
        sigma_x = self._compute_sigma(X, self.sigma_x)
        sigma_y = self._compute_sigma(Y, self.sigma_y)
        
        # Compute kernel matrices
        K = self._gaussian_kernel(X, X, sigma_x)
        L = self._gaussian_kernel(Y, Y, sigma_y)
        
        # Centering matrix H = I - (1/n)11^T
        H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
        
        # HSIC = (1/n^2) Tr(KHLH) - using biased estimator (standard)
        HSIC = torch.trace(K @ H @ L @ H) / (n ** 2)
        
        return HSIC


class NOCCOLoss(nn.Module):
    """Normalized Cross-Covariance Operator (NOCCO) Loss for Conditional Independence.
    
    Measures the conditional dependence between X and Y given Z using the Hilbert-Schmidt
    norm of the normalized conditional cross-covariance operator.
    
    Based on: "Kernel Measures of Conditional Dependence" (Fukumizu et al., NIPS 2007)
    
    Args:
        epsilon (float): Regularization parameter for matrix inversion stability. Default: 1e-4
        sigma_x (float, optional): Fixed bandwidth for X. If None, uses automatic selection.
        sigma_y (float, optional): Fixed bandwidth for Y. If None, uses automatic selection.
        sigma_z (float, optional): Fixed bandwidth for Z. If None, uses automatic selection.
        sigma_method (str): Method for automatic sigma selection. Options:
            - 'median': Median heuristic (default, recommended)
            - 'scott': Scott's rule
        sigma_scale (float): Additional scaling factor for computed sigma. Default: 1.0
    
    Example:
        >>> # Automatic sigma selection (recommended)
        >>> loss_fn = NOCCOLoss()
        >>> X, Y, Z = torch.randn(100, 5), torch.randn(100, 3), torch.randn(100, 2)
        >>> loss = loss_fn(X, Y, Z)
        >>> 
        >>> # Manual sigma specification
        >>> loss_fn = NOCCOLoss(sigma_x=1.0, sigma_y=1.0, sigma_z=0.5)
        >>> loss = loss_fn(X, Y, Z)
    """
    
    def __init__(
        self,
        epsilon: float = 1e-4,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_z: Optional[float] = None,
        sigma_method: Literal['median', 'scott'] = 'median',
        sigma_scale: float = 1.0
    ):
        """Initializes the NOCCO loss."""
        super(NOCCOLoss, self).__init__()
        self.epsilon = epsilon
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.sigma_method = sigma_method
        self.sigma_scale = sigma_scale

    def _median_heuristic(self, x: torch.Tensor) -> float:
        """Computes sigma using median heuristic."""
        n = x.size(0)
        
        if n > 1000:
            idx = torch.randperm(n, device=x.device)[:1000]
            x = x[idx]
            n = 1000
        
        dist_sq = torch.cdist(x, x, p=2) ** 2
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool().to(x.device)
        distances = dist_sq[mask]
        
        if distances.numel() == 0:
            return 1.0
        
        median_dist_sq = torch.median(distances)
        sigma = torch.sqrt(torch.clamp(median_dist_sq, min=1e-6))
        return sigma.item() * self.sigma_scale

    def _scotts_rule(self, x: torch.Tensor) -> float:
        """Computes sigma using Scott's rule."""
        n, d = x.shape
        std = x.std(dim=0).mean()
        sigma = n ** (-1.0 / (d + 4)) * std
        return max(sigma.item(), 1e-6) * self.sigma_scale

    def _compute_sigma(self, x: torch.Tensor, fixed_sigma: Optional[float]) -> float:
        """Computes or returns sigma for a given tensor."""
        if fixed_sigma is not None:
            return fixed_sigma
        
        if self.sigma_method == 'median':
            return self._median_heuristic(x)
        elif self.sigma_method == 'scott':
            return self._scotts_rule(x)
        else:
            raise ValueError(f"Unknown sigma_method: {self.sigma_method}")

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Computes Gaussian RBF kernel matrix."""
        dist_sq = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-0.5 * dist_sq / (sigma ** 2))

    def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
        """Centers the Gram matrix in feature space."""
        n = K.size(0)
        H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
        return H @ K @ H

    def _compute_normalized_cov(self, G_Y: torch.Tensor, G_X: torch.Tensor, n: int) -> torch.Tensor:
        """Computes normalized cross-covariance operator V_Y_X."""
        reg = n * self.epsilon * torch.eye(n, device=G_Y.device)
        
        try:
            L_Y = torch.linalg.cholesky(G_Y + reg)
            L_X = torch.linalg.cholesky(G_X + reg)
        except RuntimeError:
            eigenvalues_Y, eigenvectors_Y = torch.linalg.eigh(G_Y + reg)
            eigenvalues_X, eigenvectors_X = torch.linalg.eigh(G_X + reg)
            
            L_Y_inv = eigenvectors_Y @ torch.diag(1.0 / torch.sqrt(eigenvalues_Y))
            L_X_inv = eigenvectors_X @ torch.diag(1.0 / torch.sqrt(eigenvalues_X))
            
            return L_Y_inv.T @ G_Y @ G_X @ L_X_inv
        
        temp = torch.linalg.solve_triangular(L_Y, G_Y @ G_X, upper=False)
        V_Y_X = torch.linalg.solve_triangular(L_X.T, temp.T, upper=True).T
        
        return V_Y_X

    def forward(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Computes the NOCCO loss for conditional independence testing.
        
        Args:
            X: Input tensor of shape [batch_size, dim_x].
            Y: Input tensor of shape [batch_size, dim_y].
            Z: Conditioning tensor of shape [batch_size, dim_z].
        
        Returns:
            Scalar loss value (non-negative). Zero indicates conditional independence.
        """
        if X.size(0) != Y.size(0) or X.size(0) != Z.size(0):
            raise ValueError(
                f"Batch size mismatch: X={X.size(0)}, Y={Y.size(0)}, Z={Z.size(0)}"
            )
        
        n = X.size(0)
        
        # Compute or use fixed sigmas
        sigma_x = self._compute_sigma(X, self.sigma_x)
        sigma_y = self._compute_sigma(Y, self.sigma_y)
        sigma_z = self._compute_sigma(Z, self.sigma_z)

        # Compute base Gram matrices
        K_X = self._gaussian_kernel(X, X, sigma_x)
        K_Y = self._gaussian_kernel(Y, Y, sigma_y)
        K_Z = self._gaussian_kernel(Z, Z, sigma_z)

        # Construct extended variables using product kernels
        K_X_ddot = K_X * K_Z  # Ẍ = (X, Z)
        K_Y_ddot = K_Y * K_Z  # Ÿ = (Y, Z)

        # Center all Gram matrices
        G_X_ddot = self._center_gram(K_X_ddot)
        G_Y_ddot = self._center_gram(K_Y_ddot)
        G_Z = self._center_gram(K_Z)

        # Compute normalized cross-covariance operators
        V_Y_X = self._compute_normalized_cov(G_Y_ddot, G_X_ddot, n)
        V_Y_Z = self._compute_normalized_cov(G_Y_ddot, G_Z, n)
        V_Z_X = self._compute_normalized_cov(G_Z, G_X_ddot, n)

        # Compute conditional normalized cross-covariance operator
        V_Y_X_given_Z = V_Y_X - V_Y_Z @ V_Z_X

        # Compute Hilbert-Schmidt norm squared
        loss = torch.sum(V_Y_X_given_Z ** 2)
        
        return loss


class UnnormalizedHSCICLoss(nn.Module):
    """Unnormalized Hilbert-Schmidt Conditional Independence Criterion (HSCIC).
    
    Computes the unnormalized squared Hilbert-Schmidt norm of the conditional
    cross-covariance operator: ||sigma_Yẍ|Z||^2_HS.
    
    Based on: Fukumizu et al. (2004) and Sheng & Sriperumbudur (2023)
    
    Args:
        epsilon (float): Regularization parameter for matrix inversion. Default: 1e-4
        sigma_x (float, optional): Fixed bandwidth for X. If None, uses automatic selection.
        sigma_y (float, optional): Fixed bandwidth for Y. If None, uses automatic selection.
        sigma_z (float, optional): Fixed bandwidth for Z. If None, uses automatic selection.
        sigma_method (str): Method for automatic sigma selection. Options:
            - 'median': Median heuristic (default, recommended)
            - 'scott': Scott's rule
        sigma_scale (float): Additional scaling factor for computed sigma. Default: 1.0
    
    Example:
        >>> # Automatic sigma selection (recommended)
        >>> loss_fn = UnnormalizedHSCICLoss()
        >>> X, Y, Z = torch.randn(100, 5), torch.randn(100, 3), torch.randn(100, 2)
        >>> loss = loss_fn(X, Y, Z)
        >>> 
        >>> # Manual sigma specification
        >>> loss_fn = UnnormalizedHSCICLoss(sigma_x=1.0, sigma_y=1.0, sigma_z=0.5)
        >>> loss = loss_fn(X, Y, Z)
    """

    def __init__(
        self,
        epsilon: float = 1e-4,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
        sigma_z: Optional[float] = None,
        sigma_method: Literal['median', 'scott'] = 'median',
        sigma_scale: float = 1.0
    ):
        """Initializes the Unnormalized HSCIC loss."""
        super(UnnormalizedHSCICLoss, self).__init__()
        self.epsilon = epsilon
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.sigma_method = sigma_method
        self.sigma_scale = sigma_scale

    def _median_heuristic(self, x: torch.Tensor) -> float:
        """Computes sigma using median heuristic."""
        n = x.size(0)
        
        if n > 1000:
            idx = torch.randperm(n, device=x.device)[:1000]
            x = x[idx]
            n = 1000
        
        dist_sq = torch.cdist(x, x, p=2) ** 2
        mask = torch.triu(torch.ones(n, n), diagonal=1).bool().to(x.device)
        distances = dist_sq[mask]
        
        if distances.numel() == 0:
            return 1.0
        
        median_dist_sq = torch.median(distances)
        sigma = torch.sqrt(torch.clamp(median_dist_sq, min=1e-6))
        return sigma.item() * self.sigma_scale

    def _scotts_rule(self, x: torch.Tensor) -> float:
        """Computes sigma using Scott's rule."""
        n, d = x.shape
        std = x.std(dim=0).mean()
        sigma = n ** (-1.0 / (d + 4)) * std
        return max(sigma.item(), 1e-6) * self.sigma_scale

    def _compute_sigma(self, x: torch.Tensor, fixed_sigma: Optional[float]) -> float:
        """Computes or returns sigma for a given tensor."""
        if fixed_sigma is not None:
            return fixed_sigma
        
        if self.sigma_method == 'median':
            return self._median_heuristic(x)
        elif self.sigma_method == 'scott':
            return self._scotts_rule(x)
        else:
            raise ValueError(f"Unknown sigma_method: {self.sigma_method}")

    def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Computes Gaussian RBF kernel matrix."""
        dist_sq = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-0.5 * dist_sq / (sigma ** 2))

    def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
        """Centers the Gram matrix in the RKHS feature space."""
        n = K.size(0)
        eye = torch.eye(n, device=K.device)
        ones = torch.ones(n, n, device=K.device) / n
        H = eye - ones
        return H @ K @ H

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Computes the unnormalized HSCIC loss value.
        
        Args:
            x: Input tensor X of shape [batch_size, dim_x].
            y: Input tensor Y of shape [batch_size, dim_y].
            z: Conditioning tensor Z of shape [batch_size, dim_z].
        
        Returns:
            Scalar loss value (non-negative). Zero indicates X ⊥⊥ Y | Z.
        """
        if x.size(0) != y.size(0) or x.size(0) != z.size(0):
            raise ValueError(
                f"Batch size mismatch: x={x.size(0)}, y={y.size(0)}, z={z.size(0)}"
            )
        
        n = x.size(0)
        
        # Compute or use fixed sigmas
        sigma_x = self._compute_sigma(x, self.sigma_x)
        sigma_y = self._compute_sigma(y, self.sigma_y)
        sigma_z = self._compute_sigma(z, self.sigma_z)

        # Compute Gram matrices for raw variables
        k_x = self._gaussian_kernel(x, x, sigma_x)
        k_y = self._gaussian_kernel(y, y, sigma_y)
        k_z = self._gaussian_kernel(z, z, sigma_z)

        # Construct extended variable Ẍ = (X, Z)
        k_x_ddot = k_x * k_z

        # Center the Gram matrices
        g_x_ddot = self._center_gram(k_x_ddot)
        g_y = self._center_gram(k_y)
        g_z = self._center_gram(k_z)

        # Compute projection operator Q
        reg = n * self.epsilon * torch.eye(n, device=x.device)
        v = torch.linalg.solve(g_z + reg, g_z)
        q = torch.eye(n, device=x.device) - v

        # Compute unnormalized HSCIC statistic
        qgy = q @ g_y @ q
        loss = torch.sum(qgy * g_x_ddot)  # Efficient trace computation
        
        return loss


# # ============================================================================
# # Usage Examples
# # ============================================================================

# if __name__ == "__main__":
#     torch.manual_seed(42)
    
#     # Generate test data
#     n = 200
#     X = torch.randn(n, 5)
#     Y_indep = torch.randn(n, 3)
#     Y_dep = X[:, :3] + 0.3 * torch.randn(n, 3)
#     Z = torch.randn(n, 2)
    
#     print("=" * 70)
#     print("HSIC Loss (Independence Testing)")
#     print("=" * 70)
    
#     # Test HSIC with automatic sigma
#     hsic_auto = HSICLoss()
#     loss_indep = hsic_auto(X, Y_indep)
#     loss_dep = hsic_auto(X, Y_dep)
#     print(f"Automatic sigma (median):")
#     print(f"  Independent: {loss_indep.item():.6f}")
#     print(f"  Dependent:   {loss_dep.item():.6f}")
    
#     # Test HSIC with fixed sigma
#     hsic_fixed = HSICLoss(sigma_x=1.0, sigma_y=1.0)
#     loss_indep_fixed = hsic_fixed(X, Y_indep)
#     loss_dep_fixed = hsic_fixed(X, Y_dep)
#     print(f"\nFixed sigma (sigma_X=1.0, sigma_Y=1.0):")
#     print(f"  Independent: {loss_indep_fixed.item():.6f}")
#     print(f"  Dependent:   {loss_dep_fixed.item():.6f}")
    
#     # Test HSIC with Scott's rule
#     hsic_scott = HSICLoss(sigma_method='scott')
#     loss_indep_scott = hsic_scott(X, Y_indep)
#     loss_dep_scott = hsic_scott(X, Y_dep)
#     print(f"\nScott's rule:")
#     print(f"  Independent: {loss_indep_scott.item():.6f}")
#     print(f"  Dependent:   {loss_dep_scott.item():.6f}")
    
#     print("\n" + "=" * 70)
#     print("NOCCO Loss (Conditional Independence Testing)")
#     print("=" * 70)
    
#     # Test NOCCO with automatic sigma
#     nocco_auto = NOCCOLoss()
#     loss_nocco = nocco_auto(X, Y_dep, Z)
#     print(f"Automatic sigma (median): {loss_nocco.item():.6f}")
    
#     # Test NOCCO with fixed sigma
#     nocco_fixed = NOCCOLoss(sigma_x=1.0, sigma_y=1.0, sigma_z=0.5)
#     loss_nocco_fixed = nocco_fixed(X, Y_dep, Z)
#     print(f"Fixed sigma (sigma_X=1.0, sigma_Y=1.0, sigma_Z=0.5): {loss_nocco_fixed.item():.6f}")
    
#     print("\n" + "=" * 70)
#     print("Unnormalized HSCIC Loss (Conditional Independence Testing)")
#     print("=" * 70)
    
#     # Test HSCIC with automatic sigma
#     hscic_auto = UnnormalizedHSCICLoss()
#     loss_hscic = hscic_auto(X, Y_dep, Z)
#     print(f"Automatic sigma (median): {loss_hscic.item():.6f}")
    
#     # Test HSCIC with fixed sigma
#     hscic_fixed = UnnormalizedHSCICLoss(sigma_x=1.0, sigma_y=1.0, sigma_z=0.5)
#     loss_hscic_fixed = hscic_fixed(X, Y_dep, Z)
#     print(f"Fixed sigma (sigma_X=1.0, sigma_Y=1.0, sigma_Z=0.5): {loss_hscic_fixed.item():.6f}")
    
#     print("\n" + "=" * 70)
#     print("Summary")
#     print("=" * 70)
#     print("✓ All three losses support both automatic and manual sigma selection")
#     print("✓ Automatic methods: 'median' (default) and 'scott'")
#     print("✓ Manual: specify sigma_x, sigma_y (and sigma_z for conditional losses)")
#     print("✓ Recommended: Use automatic 'median' for most cases")


# def rbf_kernel(X, Y, sigma=1.0):
#     """
#     Compute the RBF (Gaussian) kernel matrix
    
#     Args:
#         X: input samples 2 (batch_size, dim)
#         Y: input samples 2 (batch_size, dim)
#         sigma: RBF kernel width
    
#     Returns:
#         K: RBF kernel matrix
#     """
#     pairwise_sq_dists = torch.cdist(X, Y, p=2) ** 2 
#     K = torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))
#     return K

# def HSICloss(X, Y, sigma=1.0):
#     """
#     Compute Hilbert-Schmidt Independence Criterion (HSIC) between two sets of samples
    
#     Args:
#         X: input samples 1 (batch_size, dim)
#         Y: input samples 2 (batch_size, dim)
#         sigma: RBF kernel width
    
#     Returns:
#         HSIC: Hilbert-Schmidt Independence Criterion (smaller HSIC means more independent)

#     """
#     n = X.shape[0]
#     K = rbf_kernel(X, X, sigma) #rbf_kernel(X, X, gamma=1 / (2 * sigma**2))
#     L = rbf_kernel(Y, Y, sigma) #rbf_kernel(Y, Y, gamma=1 / (2 * sigma**2))
#     H = torch.eye(n).to(X.device) - (1.0 / n) * torch.ones((n, n)).to(X.device)
#     # HSIC: Tr(KHLH) / (n-1)^2
#     # HSIC = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
#     HSIC = torch.trace(K @ H @ L @ H)  / ((n - 1) ** 2)
#     return HSIC

# # def pairwise_distances(x):
# #     #x should be two dimensional
# #     instances_norm = torch.sum(x**2,-1).reshape((-1,1))
# #     return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

# # def GaussianKernelMatrix(x, sigma=1):
# #     pairwise_distances_ = pairwise_distances(x)
# #     return torch.exp(-pairwise_distances_ /sigma)

# # def HSIC(x, y, s_x=1, s_y=1):
# #     m,_ = x.shape #batch size
# #     K = GaussianKernelMatrix(x,s_x)
# #     L = GaussianKernelMatrix(y,s_y)
# #     H = torch.eye(m) - 1.0/m * torch.ones((m,m))
# #     H = H.double().cuda()
# #     HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
# #     return HSIC 

# def center_kernel(K, Kz):
#     """Centers the kernel matrix K with respect to the conditioning variable Z.

#     Args:
#         K (numpy.ndarray): Kernel matrix of shape (n, n).
#         Kz (numpy.ndarray): Kernel matrix for Z of shape (n, n).

#     Returns:
#         numpy.ndarray: Centered kernel matrix of shape (n, n).
#     """
#     Kz_inv = torch.pinverse(Kz) #np.linalg.pinv(Kz)  # Compute the pseudo-inverse of Kz
#     return K - Kz @ Kz_inv @ K

# def conditional_HISCloss(X, Y, Z, sigma=1.0):
#     """Computes the Conditional Hilbert-Schmidt Independence Criterion (CHSIC).

#     Args:
#         X (numpy.ndarray): Sample matrix for variable X, shape (n, d_x).
#         Y (numpy.ndarray): Sample matrix for variable Y, shape (n, d_y).
#         Z (numpy.ndarray): Sample matrix for conditioning variable Z, shape (n, d_z).
#         sigma (float, optional): Bandwidth parameter for the RBF kernel. Defaults to 1.0.

#     Returns:
#         float: The conditional HSIC value, representing the dependency between X and Y given Z.
#     """
#     n = X.shape[0]

#     # Step 1: Remove the influence of Z by computing the residuals
#     # Fit a linear model (or another model) conditioned on Z for both X and Y.
#     X_residual = X - torch.matmul(Z, torch.pinverse(Z) @ X)
#     Y_residual = Y - torch.matmul(Z, torch.pinverse(Z) @ Y)

#     # Step 2: Compute RBF kernels for the residuals of X and Y
#     K = rbf_kernel(X_residual, X_residual, sigma)
#     L = rbf_kernel(Y_residual, Y_residual, sigma)

#     # Step 3: Centering matrix
#     H = torch.eye(n).to(X.device) - (1.0 / n) * torch.ones((n, n)).to(X.device)

#     # Step 4: Compute the conditional HSIC
#     HSIC_conditioned = torch.trace(K @ H @ L @ H) / ((n - 1) ** 2)
#     return HSIC_conditioned

#     # n = len(X)    
#     # # Compute kernel matrices
#     # Kx = rbf_kernel(X, X, sigma) #rbf_kernel(X, X, gamma=1 / (2 * sigma**2))
#     # Ky = rbf_kernel(Y, Y, sigma) #rbf_kernel(Y, Y, gamma=1 / (2 * sigma**2))
#     # Kz = rbf_kernel(Z, Z, sigma) #rbf_kernel(Z, Z, gamma=1 / (2 * sigma**2))
#     # # Center kernel matrices
#     # Kx_c = center_kernel(Kx, Kz)
#     # Ky_c = center_kernel(Ky, Kz)
#     # # Compute CHSIC
#     # return torch.trace(Kx_c @ Ky_c) / (n - 1) ** 2  # Normalization


# class NOCCOLoss(nn.Module):
#     """Normalized Cross-Covariance Operator (NOCCO) Loss for Conditional Independence.
    
#     Measures the conditional dependence between X and Y given Z using the Hilbert-Schmidt
#     norm of the normalized conditional cross-covariance operator.
    
#     Based on: "Kernel Measures of Conditional Dependence" (Fukumizu et al., NIPS 2007)
    
#     The NOCCO measures conditional independence through:
#         I_COND = ||V_Ÿ_Ẍ|Z||^2_HS
#     where V_Y_X|Z = V_Y_X - V_Y_Z V_Z_X is the normalized conditional cross-covariance
#     operator, with V_Y_X = sigma_Y_Y^(-1/2) sigma_Y_X sigma_X_X^(-1/2).
    
#     For the conditional case, we use extended variables:
#         Ẍ = (X, Z) and Ÿ = (Y, Z)
    
#     Args:
#         epsilon (float): Regularization parameter for matrix inversion stability.
#             Default: 1e-4
#         sigma_scale (float): Scaling factor for median heuristic bandwidth selection.
#             Set to None to use fixed sigma=1.0. Default: 1.0
    
#     Example:
#         >>> loss_fn = NOCCOLoss(epsilon=1e-4, sigma_scale=1.0)
#         >>> X = torch.randn(100, 5)  # 100 samples, 5 dims
#         >>> Y = torch.randn(100, 3)  # 100 samples, 3 dims
#         >>> Z = torch.randn(100, 2)  # 100 samples, 2 dims
#         >>> loss = loss_fn(X, Y, Z)
#         >>> print(loss.item())  # Lower values indicate more conditional independence
#     """
    
#     def __init__(self, epsilon: float = 1e-4, sigma_scale: float = 1.0):
#         """Initializes the NOCCO loss.
        
#         Args:
#             epsilon: Small positive constant for Tikhonov regularization.
#             sigma_scale: Multiplier for the median heuristic bandwidth.
#         """
#         super(NOCCOLoss, self).__init__()
#         self.epsilon = epsilon
#         self.sigma_scale = sigma_scale

#     def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """Computes Gaussian RBF kernel matrix with median heuristic for bandwidth.
        
#         K(x, y) = exp(-||x - y||^2 / (2sigma^2))
#         where sigma^2 is determined by the median of pairwise distances.
        
#         Args:
#             x: Tensor of shape [batch_size, dim_x].
#             y: Tensor of shape [batch_size, dim_y].
        
#         Returns:
#             Kernel matrix of shape [batch_size, batch_size].
#         """
#         n = x.size(0)
#         dist_sq = torch.cdist(x, y, p=2) ** 2
        
#         # Median heuristic for bandwidth selection
#         if self.sigma_scale is not None:
#             with torch.no_grad():
#                 # Extract upper triangular distances (excluding diagonal)
#                 mask = torch.triu(torch.ones(n, n), diagonal=1).bool().to(x.device)
#                 dists = dist_sq[mask]
#                 if dists.numel() > 0:
#                     median_dist = torch.median(dists)
#                     sigma_sq = median_dist * self.sigma_scale
#                 else:
#                     sigma_sq = torch.tensor(1.0, device=x.device)
#                 # Avoid division by zero
#                 sigma_sq = torch.clamp(sigma_sq, min=1e-6)
#         else:
#             sigma_sq = 1.0

#         K = torch.exp(-0.5 * dist_sq / sigma_sq)
#         return K

#     def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
#         """Centers the Gram matrix in feature space.
        
#         Computes G = H K H where H = I - (1/n)11^T is the centering matrix.
#         This removes the mean in the RKHS feature space.
        
#         Args:
#             K: Kernel matrix of shape [n, n].
        
#         Returns:
#             Centered Gram matrix of shape [n, n].
#         """
#         n = K.size(0)
#         H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
#         G = H @ K @ H
#         return G

#     def _compute_normalized_cov(
#         self, 
#         G_Y: torch.Tensor, 
#         G_X: torch.Tensor, 
#         n: int
#     ) -> torch.Tensor:
#         """Computes normalized cross-covariance operator V_Y_X.
        
#         V_Y_X = (G_Y + nεI)^(-1/2) G_Y G_X (G_X + nεI)^(-1/2)
        
#         This is the correlation operator that separates the marginal covariances
#         from the dependence structure, analogous to Pearson correlation.
        
#         Args:
#             G_Y: Centered Gram matrix for Y, shape [n, n].
#             G_X: Centered Gram matrix for X, shape [n, n].
#             n: Number of samples.
        
#         Returns:
#             Normalized cross-covariance operator of shape [n, n].
#         """
#         reg = n * self.epsilon * torch.eye(n, device=G_Y.device)
        
#         # Compute (G_Y + nεI)^(-1/2) using Cholesky decomposition for stability
#         # L_Y L_Y^T = G_Y + nεI, so (G_Y + nεI)^(-1/2) = L_Y^(-T)
#         try:
#             L_Y = torch.linalg.cholesky(G_Y + reg)
#             L_X = torch.linalg.cholesky(G_X + reg)
#         except RuntimeError:
#             # Fallback to eigendecomposition if Cholesky fails
#             eigenvalues_Y, eigenvectors_Y = torch.linalg.eigh(G_Y + reg)
#             eigenvalues_X, eigenvectors_X = torch.linalg.eigh(G_X + reg)
            
#             L_Y_inv = eigenvectors_Y @ torch.diag(1.0 / torch.sqrt(eigenvalues_Y))
#             L_X_inv = eigenvectors_X @ torch.diag(1.0 / torch.sqrt(eigenvalues_X))
            
#             V_Y_X = L_Y_inv.T @ G_Y @ G_X @ L_X_inv
#             return V_Y_X
        
#         # V_Y_X = L_Y^(-1) G_Y G_X L_X^(-T)
#         temp = torch.linalg.solve_triangular(L_Y, G_Y @ G_X, upper=False)
#         V_Y_X = torch.linalg.solve_triangular(L_X.T, temp.T, upper=True).T
        
#         return V_Y_X

#     def forward(
#         self, 
#         X: torch.Tensor, 
#         Y: torch.Tensor, 
#         Z: torch.Tensor
#     ) -> torch.Tensor:
#         """Computes the NOCCO loss for conditional independence testing.
        
#         The loss measures X ⊥⊥ Y | Z. Lower values indicate stronger conditional
#         independence. The loss is zero if and only if X and Y are conditionally
#         independent given Z (assuming characteristic kernels).
        
#         Args:
#             X: Input tensor of shape [batch_size, dim_x].
#             Y: Input tensor of shape [batch_size, dim_y].
#             Z: Conditioning tensor of shape [batch_size, dim_z].
        
#         Returns:
#             Scalar loss value (non-negative). Zero indicates conditional independence.
        
#         Raises:
#             ValueError: If batch sizes don't match.
#         """
#         if X.size(0) != Y.size(0) or X.size(0) != Z.size(0):
#             raise ValueError(
#                 f"Batch size mismatch: X={X.size(0)}, Y={Y.size(0)}, Z={Z.size(0)}"
#             )
        
#         n = X.size(0)

#         # 1. Compute base Gram matrices
#         K_X = self._gaussian_kernel(X, X)
#         K_Y = self._gaussian_kernel(Y, Y)
#         K_Z = self._gaussian_kernel(Z, Z)

#         # 2. Construct extended variables using product kernels
#         # For joint spaces, the characteristic kernel is the product kernel
#         # (Theorem 3 in NIPS 2007 paper)
#         K_X_ddot = K_X * K_Z  # Ẍ = (X, Z)
#         K_Y_ddot = K_Y * K_Z  # Ÿ = (Y, Z)

#         # 3. Center all Gram matrices
#         G_X_ddot = self._center_gram(K_X_ddot)
#         G_Y_ddot = self._center_gram(K_Y_ddot)
#         G_Z = self._center_gram(K_Z)

#         # 4. Compute normalized cross-covariance operators
#         V_Y_X = self._compute_normalized_cov(G_Y_ddot, G_X_ddot, n)
#         V_Y_Z = self._compute_normalized_cov(G_Y_ddot, G_Z, n)
#         V_Z_X = self._compute_normalized_cov(G_Z, G_X_ddot, n)

#         # 5. Compute conditional normalized cross-covariance operator
#         # V_Y_X|Z = V_Y_X - V_Y_Z V_Z_X
#         V_Y_X_given_Z = V_Y_X - V_Y_Z @ V_Z_X

#         # 6. Compute Hilbert-Schmidt norm squared: ||V_Y_X|Z||^2_HS = Tr(V^T V)
#         # For symmetric V, this is Tr(V^2) = sum of squared singular values
#         loss = torch.sum(V_Y_X_given_Z ** 2)
        
#         return loss


# class UnnormalizedHSCICLoss(nn.Module):
#     """Unnormalized Hilbert-Schmidt Conditional Independence Criterion (HSCIC).
    
#     Computes the unnormalized squared Hilbert-Schmidt norm of the conditional
#     cross-covariance operator: ||sigma_Yẍ|Z||^2_HS.
    
#     Unlike NOCCO, this metric is unnormalized (scale-sensitive) regarding X and Y,
#     but computationally more efficient as it requires only one matrix inversion for Z.
#     It strictly characterizes conditional independence using extended variables.
    
#     Theoretical Basis:
#         - Based on Fukumizu et al. (2004) and Sheng & Sriperumbudur (2023).
#         - Uses projection operator Q = I - (G_Z + nεI)^(-1) G_Z to remove the
#           effect of Z from the conditional covariance.
#         - Loss = Tr(Q G_Y Q G_Ẍ) where Ẍ = (X, Z).
    
#     Key Property:
#         For characteristic kernels, Loss = 0 ⟺ X ⊥⊥ Y | Z
    
#     Args:
#         epsilon (float): Regularization parameter for matrix inversion.
#             Default: 1e-4
#         sigma_scale (float): Scaling factor for median heuristic bandwidth.
#             Set to None to use fixed sigma=1.0. Default: 1.0
    
#     Example:
#         >>> loss_fn = UnnormalizedHSCICLoss(epsilon=1e-4)
#         >>> X = torch.randn(100, 5)
#         >>> Y = torch.randn(100, 3)
#         >>> Z = torch.randn(100, 2)
#         >>> loss = loss_fn(X, Y, Z)
#         >>> print(f"HSCIC: {loss.item():.6f}")
#     """

#     def __init__(self, epsilon: float = 1e-4, sigma_scale: float = 1.0):
#         """Initializes the Unnormalized HSCIC loss.
        
#         Args:
#             epsilon: Small positive constant for Tikhonov regularization to ensure
#                 numerical stability during matrix inversion.
#             sigma_scale: Multiplier for the Gaussian kernel bandwidth determined
#                 via median heuristic. None uses sigma=1.0.
#         """
#         super(UnnormalizedHSCICLoss, self).__init__()
#         self.epsilon = epsilon
#         self.sigma_scale = sigma_scale

#     def _gaussian_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """Computes Gaussian RBF kernel matrix with automatic bandwidth selection.
        
#         Uses the median heuristic: sigma^2 = median(||x_i - x_j||^2) * sigma_scale.
#         This adaptive bandwidth selection often works well in practice.
        
#         Args:
#             x: Tensor of shape [batch_size, dim_x].
#             y: Tensor of shape [batch_size, dim_y].
        
#         Returns:
#             Kernel matrix K(x, y) of shape [batch_size, batch_size].
#         """
#         n = x.size(0)
#         dist_sq = torch.cdist(x, y, p=2) ** 2

#         # Median heuristic for bandwidth selection
#         if self.sigma_scale is not None:
#             with torch.no_grad():
#                 mask = torch.triu(torch.ones(n, n), diagonal=1).bool().to(x.device)
#                 dists = dist_sq[mask]
#                 if dists.numel() > 0:
#                     median_dist = torch.median(dists)
#                     sigma_sq = median_dist * self.sigma_scale
#                 else:
#                     sigma_sq = torch.tensor(1.0, device=x.device)
#                 sigma_sq = torch.clamp(sigma_sq, min=1e-6)
#         else:
#             sigma_sq = 1.0

#         return torch.exp(-0.5 * dist_sq / sigma_sq)

#     def _center_gram(self, K: torch.Tensor) -> torch.Tensor:
#         """Centers the Gram matrix in the RKHS feature space.
        
#         Computes G = H K H where H = I - (1/n)11^T is the centering matrix.
#         This operation removes the mean in the implicit feature space.
        
#         Args:
#             K: Kernel matrix of shape [n, n].
        
#         Returns:
#             Centered Gram matrix G of shape [n, n].
#         """
#         n = K.size(0)
#         eye = torch.eye(n, device=K.device)
#         ones = torch.ones(n, n, device=K.device) / n
#         H = eye - ones
#         return H @ K @ H

#     def forward(
#         self, 
#         x: torch.Tensor, 
#         y: torch.Tensor, 
#         z: torch.Tensor
#     ) -> torch.Tensor:
#         """Computes the unnormalized HSCIC loss value.
        
#         Algorithm:
#             1. Compute kernel matrices K_X, K_Y, K_Z
#             2. Construct extended variable kernel K_Ẍ = K_X * K_Z (product kernel)
#             3. Center all Gram matrices
#             4. Compute projection operator Q = I - (G_Z + nεI)^(-1) G_Z
#             5. Return Tr(Q G_Y Q G_Ẍ)
        
#         The projection operator Q projects onto the orthogonal complement of
#         the range of Z in the RKHS, effectively removing Z's influence.
        
#         Args:
#             x: Input tensor X of shape [batch_size, dim_x].
#             y: Input tensor Y of shape [batch_size, dim_y].
#             z: Conditioning tensor Z of shape [batch_size, dim_z].
        
#         Returns:
#             Scalar loss value (non-negative). Zero indicates X ⊥⊥ Y | Z.
#             Minimizing this loss enforces conditional independence.
        
#         Raises:
#             ValueError: If batch sizes don't match.
#         """
#         if x.size(0) != y.size(0) or x.size(0) != z.size(0):
#             raise ValueError(
#                 f"Batch size mismatch: x={x.size(0)}, y={y.size(0)}, z={z.size(0)}"
#             )
        
#         n = x.size(0)

#         # 1. Compute Gram matrices for raw variables
#         k_x = self._gaussian_kernel(x, x)
#         k_y = self._gaussian_kernel(y, y)
#         k_z = self._gaussian_kernel(z, z)

#         # 2. Construct extended variable Ẍ = (X, Z)
#         # Theoretical requirement: To guarantee X ⊥⊥ Y | Z ⟺ Loss = 0,
#         # we must use a characteristic kernel for (X, Z).
#         # For product spaces, this is the element-wise product: k_Ẍ = k_X * k_Z
#         # (See Sheng & Sriperumbudur 2023, Theorem 1)
#         k_x_ddot = k_x * k_z

#         # 3. Center the Gram matrices
#         g_x_ddot = self._center_gram(k_x_ddot)
#         g_y = self._center_gram(k_y)
#         g_z = self._center_gram(k_z)

#         # 4. Compute projection operator Q
#         # Q projects onto the orthogonal complement of Z's range in RKHS
#         # Ideally: Q = I - P_Z where P_Z projects onto Z
#         # Empirically: Q = I - (G_Z + nεI)^(-1) G_Z
#         # Note: Regularization is scaled by n to match eigenvalue scale
#         reg = n * self.epsilon * torch.eye(n, device=x.device)

#         # Solve (G_Z + reg) V = G_Z ⟹ V = (G_Z + reg)^(-1) G_Z
#         # Using solve is more numerically stable than explicit inversion
#         # Note: G_Z is symmetric, so (G_Z + reg)^(-1) G_Z = G_Z (G_Z + reg)^(-1)
#         v = torch.linalg.solve(g_z + reg, g_z)

#         # Q = I - V
#         q = torch.eye(n, device=x.device) - v

#         # 5. Compute unnormalized HSCIC statistic
#         # Loss = ||sigma_Yẍ|Z||^2_HS
#         # Empirically estimated as: Tr(Q G_Y Q G_Ẍ)
#         # This measures correlation between Y and Ẍ after removing Z's effect
#         # (See Fukumizu et al. 2004, Proposition 5 and Sheng & Sriperumbudur 2023)
        
#         # Optimization: Compute trace efficiently
#         # Tr(Q G_Y Q G_Ẍ) = Tr(Q^2 G_Y G_Ẍ) since Q is symmetric
#         # Further optimization: use element-wise multiplication for trace
#         # Tr(AB) = sum(A * B^T) when both are symmetric
#         qgy = q @ g_y @ q
#         loss = torch.sum(qgy * g_x_ddot)  # Efficient trace computation
        
#         return loss


# # Utility function for testing conditional independence
# def test_conditional_independence(
#     X: torch.Tensor,
#     Y: torch.Tensor,
#     Z: torch.Tensor,
#     method: str = 'hscic',
#     epsilon: float = 1e-4,
#     sigma_scale: float = 1.0,
#     threshold: float = 0.01
# ) -> dict:
#     """Tests conditional independence X ⊥⊥ Y | Z using kernel methods.
    
#     Args:
#         X: Input tensor of shape [n, dim_x].
#         Y: Input tensor of shape [n, dim_y].
#         Z: Conditioning tensor of shape [n, dim_z].
#         method: Either 'hscic' (faster) or 'nocco' (normalized). Default: 'hscic'.
#         epsilon: Regularization parameter.
#         sigma_scale: Kernel bandwidth scaling.
#         threshold: Threshold for considering independence (heuristic).
    
#     Returns:
#         Dictionary with keys:
#             - 'statistic': The computed test statistic
#             - 'independent': Boolean indicating if X ⊥⊥ Y | Z (heuristic)
#             - 'method': Method used
    
#     Example:
#         >>> X = torch.randn(100, 5)
#         >>> Y = X + torch.randn(100, 5) * 0.1  # Dependent
#         >>> Z = torch.randn(100, 2)
#         >>> result = test_conditional_independence(X, Y, Z)
#         >>> print(f"Independent: {result['independent']}")
#     """
#     if method.lower() == 'hscic':
#         loss_fn = UnnormalizedHSCICLoss(epsilon=epsilon, sigma_scale=sigma_scale)
#     elif method.lower() == 'nocco':
#         loss_fn = NOCCOLoss(epsilon=epsilon, sigma_scale=sigma_scale)
#     else:
#         raise ValueError(f"Unknown method: {method}. Use 'hscic' or 'nocco'.")
    
#     with torch.no_grad():
#         statistic = loss_fn(X, Y, Z).item()
    
#     return {
#         'statistic': statistic,
#         'independent': statistic < threshold,
#         'method': method.upper()
#     }
