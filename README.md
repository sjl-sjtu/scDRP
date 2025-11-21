This is a package to analyze single-cell perturbation data. 
It learns the latent factors that generates the observed gene profiles via sparsity-regularized disentangled VAE, 
which disentangles the latent space into perturbation-dependent and perturbation-invariant subspaces. 
Leveraging the disentangled latent factors, 
we can estimate individual treatment effects (ITE) and generate counterfactual samples via soft-conditional optimal transport on the disentangled latent space. 
This is based on the idea that the effect of perturbation on those perturbation-dependent latent factors should be rank-preserved when conditioning on confounders.

![Fig1](https://github.com/user-attachments/assets/002e1f66-3bfb-42bb-982d-8bd8a5840bf7)
