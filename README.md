This is a package to analyze single-cell perturbation data. 
It learns the latent factors that generates the observed gene profiles via sparsity-regularized disentangled VAE, 
which disentangles the latent space into perturbation-dependent and perturbation-invariant subspaces. 
Leveraging the disentangled latent factors, 
we can estimate individual treatment effects (ITE) and generate counterfactual samples via soft-conditional optimal transport on the disentangled latent space. 
This is based on the idea that the effect of perturbation on those perturbation-dependent latent factors should be rank-preserved when conditioning on confounders.

![Fig1](https://github.com/user-attachments/assets/002e1f66-3bfb-42bb-982d-8bd8a5840bf7)

## Tutorial
To install the package
```
git clone https://github.com/sjl-sjtu/scDRP.git
cd scMRDR
pip install -e .
```
A simple example
```
from scDRP.module import *

# train model
adata = sc.read_h5ad("example.h5ad")
model = Perturb(adata, perturbation_key="perturbation", 
                celltype_key="celltype",distribution="Normal_positive") 
model.setup(latent_dependent = 50, latent_independent = 50, hidden_layers = [512, 512], 
            beta = 2, sparse_coef = 0, l0_latent=0.01, lambda_hsic=1)
model.train(epoch_num = 200, batch_size = 64, lr = 1e-3, tensorboard=False, accumulation_steps=1,
            valid_prop=0.1, early_stopping=True, adaptlr=False)
model.inference(n_samples=10, batch_size = 64, update=False, returns=False)

# get latent embeddings
zd, zu, mu_d, mu_u = model.get_latent()

# estimate effect
ITE = model.effect_estimate(control='control',treatment='treatment', strategy="ot", alpha=0.1, beta=1, projection_strategy="full", method="emd")

# predict counterfactual samples
samples_counterfactual = model.counterfactual_samples(control='control',treatment='treatment', strategy="ot", alpha=0.1, beta=1, projection_strategy="full", method="emd")
```

## Reference
Jianle Sun, Petar Stojanov, Kun Zhang. Single-cell disentangled representations for perturbation modeling and treatment effect estimation. bioRxiv 2025.11.21.689783; doi: https://doi.org/10.1101/2025.11.21.689783

