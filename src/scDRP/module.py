import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
import torch.utils.data as Data
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from .model import PerturbNet
from .train import train_model,inference_model
from torch.utils.tensorboard import SummaryWriter
import scanpy as sc
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from .data import CombinedDataset,LatentDataset,LabelDataset
import ot
import umap
import matplotlib.pyplot as plt
from .utils import * 
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.multioutput import MultiOutputRegressor
from scipy.interpolate import UnivariateSpline, interp1d, make_interp_spline


import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class DosageModel:
    def __init__(self, models):
        '''
        initialize the DosageModel with given models for mu and sigma.

        Args:
            models: a dictionary containing 'mu' and 'sigma' models.
        '''
        self.models = models
        self.zd_dim = len(models['mu'])
        self.is_gpr = isinstance(self.models['mu'][0], GaussianProcessRegressor)

    def predict(self, v_new_log):
        """
        Predict the mu and sigma for new dosage values.

        Args:
            v_new_log: a numpy array of log-transformed dosage values.

        Returns:
            mu_pred: a numpy array of predicted means.
            sigma_pred: a numpy array of predicted standard deviations.
        """
        v_new_log = v_new_log.flatten()
        n_samples = len(v_new_log)
        
        mu_pred = np.zeros((n_samples, self.zd_dim))
        sigma_pred = np.zeros((n_samples, self.zd_dim))

        v_pred_gpr = v_new_log.reshape(-1, 1) if self.is_gpr else None

        for i in range(self.zd_dim):
            # --- predict MU ---
            if self.is_gpr:
                mu_pred[:, i] = self.models['mu'][i].predict(v_pred_gpr)
            else:
                mu_pred[:, i] = self.models['mu'][i](v_new_log)

            # --- predict SIGMA ---
            if self.is_gpr:
                log_sigma = self.models['sigma'][i].predict(v_pred_gpr)
            else:
                log_sigma = self.models['sigma'][i](v_new_log)
            
            sigma_pred[:, i] = np.exp(log_sigma)
        
        return mu_pred, sigma_pred


class Perturb:
    def __init__(self, adata, layer=None, perturbation_key="perturbation", celltype_key=None, 
                 batch_key=None, dose_key=None, distribution="ZINB"): # count_data=False, positive_output=True):
        '''
        Initialize the Perturb object.

        Args:
            adata: an AnnData object    
            layer: a string representing the layer name of count matrix in adata
            perturbation_key: a string representing the key of perturbation in adata.obs
            celltype_key: a string representing the key of cell type in adata.obs
            batch_key: a string representing the key of batch in adata.obs
            dose_key: a string representing the key of dose in adata.obs
            distribution: a string representing the distribution of the data. Options (default: "ZINB"):
                - "ZINB": Zero-Inflated Negative Binomial distribution
                - "NB": Negative Binomial distribution
                - "Normal": Gaussian distribution
                - "Normal_positive": Gaussian distribution with positive output
        '''
        super(Perturb, self).__init__()
        self.layer = layer
        self.perturbation_key = perturbation_key
        self.celltype_key = celltype_key
        self.batch_key = batch_key
        self.dose_key = dose_key
        self.distribution = distribution
        self.count_data = self.distribution in ["ZINB","NB"]

        self.adata = adata
        if layer is None:
            self.data = to_dense_array(self.adata.X)
        else:
            self.data = to_dense_array(self.adata.layers[layer])
        # label_encoder = LabelEncoder()
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        
        self.perturbation_encoder = LabelEncoder()
        self.celltype_encoder = LabelEncoder()
        self.covariate_encoder = LabelEncoder()

        self.perturbation_label = adata.obs[perturbation_key].to_numpy()
        self.perturbation = self.perturbation_encoder.fit_transform(self.perturbation_label)
        self.perturbation = self.onehot_encoder.fit_transform(self.perturbation.reshape(-1, 1))
        if celltype_key is None:
            self.celltype = None
        else:
            self.celltype_label = adata.obs[celltype_key].to_numpy()
            self.celltype = self.celltype_encoder.fit_transform(self.celltype_label)
            self.celltype = self.onehot_encoder.fit_transform(self.celltype.reshape(-1, 1))
        if batch_key is None:
            self.covariates = None
        else:
            self.covariates_label = adata.obs[batch_key].to_numpy()
            self.covariates = self.covariate_encoder.fit_transform(self.covariates_label)
            self.covariates = self.onehot_encoder.fit_transform(self.covariates.reshape(-1, 1))
        if dose_key is not None:
            self.dose = adata.obs[dose_key].to_numpy().reshape(-1,1).astype(float)
            self.perturbation = np.concatenate([self.perturbation,self.dose],axis=-1)
            
        self.perturbation_num = self.perturbation.shape[1]
        if self.celltype is not None:
            self.celltype_num = self.celltype.shape[1]
        else:
            self.celltype_num = 0
        
        if self.covariates is not None:
            self.covariates_dim = self.covariates.shape[1]
        else:
            self.covariates_dim = 0
        
        # self.count_data = count_data
        # self.positive_output = positive_output
    
    def setup(self, hidden_layers = [128,128], latent_dependent = 50, latent_independent = 50, 
              beta = 1, sparse_coef = 0, l0_latent = 0.001, lambda_hsic = 0.2,
              library_size_strategy="observed", device=None):
        '''
        Setup the model for training.
        
        Args:
            hidden_layers: a list of integers representing the number of neurons in each hidden layer
            latent_dependent: an integer representing the dimensions of z_D
            latent_independent: an integer representing the dimensions of z_I
            beta: a float number representing the weight of the KL divergence term
            sparse_coef: a float number representing the weight of the sparsity regularization on the Jacobian matrix
            l0_latent: a float number representing the weight of the L0 regularization on the dimensions of latent variables
            lambda_hsic: a float number representing the weight of the HSIC regularization
            library_size_strategy: a string representing the library size normalization strategy. Options (default: "observed"):
                - "batch_sample": sample from batch empirical distribution
                - "observed": use the observed library size
                - "original": set the library size as 1
            device: the device to run the model on. Default is None, which uses 'cuda' if available, else 'cpu'.
        '''
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print("using "+self.device)
        self.input_dim = self.data.shape[1]
        self.hidden_layers = hidden_layers
        self.latent_dependent = latent_dependent
        self.latent_independent = latent_independent
        self.beta = beta
        self.sparse_coef = sparse_coef
        self.l0_latent = l0_latent
        self.library_size_strategy = library_size_strategy
        self.lambda_hsic = lambda_hsic
        self.model = PerturbNet(self.device, input_dim=self.input_dim, covariate_dim = self.covariates_dim, 
                    perturbation_num = self.perturbation_num, celltype_num = self.celltype_num,
                    layer_dims = self.hidden_layers, latent_dep_dim = self.latent_dependent, 
                    latent_ind_dim = self.latent_independent, 
                    dropout_rate = 0.5, lambda_sparse = self.sparse_coef, l0_latent = self.l0_latent,
                    beta = self.beta, distribution = self.distribution, # count_data=self.count_data, positive_output=self.positive_output,
                    library_size_strategy=self.library_size_strategy, 
                    lambda_hsic = self.lambda_hsic, eps=1e-10)
        self.train_dataset = CombinedDataset(self.data,self.perturbation,self.celltype,self.covariates)
    
    def train(self, epoch_num = 200, batch_size = 64, lr = 1e-6, accumulation_steps = 1, 
              adaptlr = True, valid_prop = 0, early_stopping = False, patience = 10,
              tensorboard = False, savepath = "./"):
        '''
        Train the model.

        Args:
            epoch_num: an integer representing the number of epochs
            batch_size: an integer representing the batch size
            lr: a float number representing the learning rate
            accumulation_steps: an integer representing the number of steps for gradient accumulation
            adaptlr: a boolean representing whether to use adaptive learning rate
            valid_prop: a float number representing the proportion of the dataset to use for validation
            early_stopping: a boolean representing whether to use early stopping
            patience: an integer representing the number of epochs to wait for improvement before stopping
            tensorboard: a boolean representing whether to use tensorboard
            savepath: a string representing the path to save the tensorboard logs
        '''
        if tensorboard:
            print("Using tensorboard!")
            writer = SummaryWriter(savepath)
        else:
            writer = None
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        
        # self.num_batch = len(self.train_dataset)//self.batch_size
        if valid_prop > 0:
            train_indices, valid_indices = train_test_split(
                np.arange(len(self.train_dataset)),
                test_size=valid_prop,
                stratify=self.perturbation.argmax(-1),
                random_state=46
            )
            train_dataset = Data.Subset(self.train_dataset, train_indices)
            valid_dataset = Data.Subset(self.train_dataset, valid_indices)
        else:
            train_dataset, valid_dataset = self.train_dataset, self.train_dataset
        self.num_batch = len(train_dataset)//self.batch_size

        print("Training start!")
        # train_model(self.device, writer, self.train_dataset, 
        #             self.model, self.epoch_num, self.batch_size, 
        #             self.num_batch, self.lr, self.accumulation_steps, 
        #             adaptlr = adaptlr, count_data = self.count_data)
        train_model(self.device, writer, train_dataset, valid_dataset, 
                    self.model, self.epoch_num, self.batch_size, 
                    self.num_batch, self.lr, self.accumulation_steps, 
                    adaptlr = adaptlr, count_data = self.count_data, 
                    early_stopping = early_stopping, patience = patience)
        if tensorboard:
            writer.close()
        print("Training finished!")
    
    def create_dataset(self, adata):
        '''
        Create a dataset from the AnnData object.

        Args:
            adata: an AnnData object.
        
        Returns:
            a CombinedDataset object.
        '''
        layer = self.layer
        perturbation_key = self.perturbation_key
        celltype_key = self.celltype_key
        batch_key = self.batch_key
        dose_key = self.dose_key
        
        self.adata = adata
        if layer is None:
            self.data = to_dense_array(adata.X)
        else:
            self.data = to_dense_array(adata.layers[layer])
        self.perturbation_label = adata.obs[perturbation_key].to_numpy()
        self.perturbation = self.perturbation_encoder.fit_transform(self.perturbation_label)
        self.perturbation = self.onehot_encoder.fit_transform(self.perturbation.reshape(-1, 1))
        if celltype_key is None:
            self.celltype = None
        else:
            self.celltype_label = adata.obs[celltype_key].to_numpy()
            self.celltype = self.celltype_encoder.fit_transform(self.celltype_label)
            self.celltype = self.onehot_encoder.fit_transform(self.celltype.reshape(-1, 1))
        if batch_key is None:
            self.covariates = None
        else:
            self.covariates_label = adata.obs[batch_key].to_numpy()
            self.covariates = self.covariate_encoder.fit_transform(self.covariates_label)
            self.covariates = self.onehot_encoder.fit_transform(self.covariates.reshape(-1, 1))
        if dose_key is not None:
            self.dose = adata.obs[dose_key].to_numpy().reshape(-1,1).astype(float)
            self.perturbation = np.concatenate([self.perturbation,self.dose],axis=-1)
        dataset = CombinedDataset(self.data,self.perturbation,self.celltype,self.covariates)
        return dataset
        
    
    def inference(self, n_samples=1, dataset=None, batch_size=None, update=False, returns=False):
        '''
        Perform inference.
        
        Args:
            n_samples: an integer representing the number of samples repeated for inference process
            dataset: a CombinedDataset object. By default we use the training dataset
            batch_size: an integer representing the batch size
            update: a boolean representing whether to update the adata
            returns: a boolean representing whether to return the results
            
        Returns: 
            a tuple of numpy arrays representing the latent variables (z_d,z_u,mu_d,mu_u,rho,dispersion,dropout_rate,library_size)
        '''
        if dataset is None:
            dataset = self.train_dataset
        else:
            dataset = self.create_dataset(dataset)
        if batch_size is None:
            batch_size = self.batch_size
        if self.count_data:
            if n_samples > 1:
                zd,zu,mu_d,mu_u,logvar_d,logvar_u, rho,dispersion,pi,library_size = \
                    zip(*[inference_model(self.device, dataset, self.model, batch_size, self.count_data) for _ in range(n_samples)])
                self.zd = np.mean(np.stack(zd, axis=0), axis=0) 
                self.zu = np.mean(np.stack(zu, axis=0), axis=0) 
                self.mu_d = np.mean(np.stack(mu_d, axis=0), axis=0) 
                self.mu_u = np.mean(np.stack(mu_u, axis=0), axis=0) 
                self.logvar_d = np.mean(np.stack(logvar_d, axis=0), axis=0) 
                self.logvar_u = np.mean(np.stack(logvar_u, axis=0), axis=0) 
                self.rho = np.mean(np.stack(rho, axis=0), axis=0) 
                self.dispersion = np.mean(np.stack(dispersion, axis=0), axis=0) 
                self.pi = np.mean(np.stack(pi, axis=0), axis=0) 
                self.library_size = np.mean(np.stack(library_size, axis=0), axis=0) 
            else:
                self.zd,self.zu,self.mu_d,self.mu_u,self.logvar_d, self.logvar_u, self.rho,self.dispersion,self.pi,self.library_size = \
                    inference_model(self.device, dataset, self.model, batch_size, self.count_data)
            if update:
                # self.adata.obsm['latent_dependent'] = self.zd
                # self.adata.obsm['latent_independent'] = self.zu
                self.adata.obsm['latent_dependent'] = self.mu_d
                self.adata.obsm['latent_independent'] = self.mu_u
                # self.adata.obsm['logvar_latent_dependent'] = self.logvar_d
                # self.adata.obsm['logvar_latent_independent'] = self.logvar_u
                self.adata.layers['estimated_mean_expression'] = self.rho
                # self.adata.layers['estimated_dropout_rate'] = self.pi
                # self.adata.var['estimated_dispersion_factor'] = self.dispersion
                # self.adata.obs['estimated_library_size'] = self.library_size
                print('All results recorded in adata.')
            if returns:
                return self.zd,self.zu,self.mu_d,self.mu_u,self.logvar_d,self.logvar_u,self.rho,self.dispersion,self.pi,self.library_size
        else:
            if n_samples > 1:
                zd,zu,mu_d,mu_u,logvar_d,logvar_u,rho = \
                    zip(*[inference_model(self.device, dataset, self.model, batch_size, self.count_data) for _ in range(n_samples)])
                self.zd = np.mean(np.stack(zd, axis=0), axis=0) 
                self.zu = np.mean(np.stack(zu, axis=0), axis=0) 
                self.mu_d = np.mean(np.stack(mu_d, axis=0), axis=0) 
                self.mu_u = np.mean(np.stack(mu_u, axis=0), axis=0) 
                self.logvar_d = np.mean(np.stack(logvar_d, axis=0), axis=0) 
                self.logvar_u = np.mean(np.stack(logvar_u, axis=0), axis=0) 
                self.rho = np.mean(np.stack(rho, axis=0), axis=0) 
            else:
                self.zd,self.zu,self.mu_d,self.mu_u,self.logvar_d, self.logvar_u, self.rho = \
                    inference_model(self.device, dataset, self.model, batch_size, self.count_data)
            if update:
                # self.adata.obsm['latent_dependent'] = self.zd
                # self.adata.obsm['latent_independent'] = self.zu
                self.adata.obsm['latent_dependent'] = self.mu_d
                self.adata.obsm['latent_independent'] = self.mu_u
                # self.adata.obsm['logvar_latent_dependent'] = self.logvar_d
                # self.adata.obsm['logvar_latent_independent'] = self.logvar_u
                self.adata.layers['estimated_mean_expression'] = self.rho
                print('All results recorded in adata.')
            if returns:
                return self.zd,self.zu,self.mu_d,self.mu_u,self.logvar_d,self.logvar_u,self.rho

    
    def save(self, path):
        '''
        Save the trained model.

        Args:
            path: a string representing the path to the model checkpoint.
        '''
        torch.save({
            "model": self.model.state_dict()
        }, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path, device=None):
        '''
        Load the trained model.

        Args:
            path: a string representing the path to the model checkpoint.
            device: the device to load the model onto. Default is None, which uses self.device.
        '''
        if device is None:
            device = self.device
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        self.model.load_state_dict(checkpoint["model"])

    def get_latent(self):
        '''
        Return the disentangled latent embeddings.
        '''
        return self.zd,self.zu,self.mu_d,self.mu_u

    def sample_posterior(self, n_samples=1, mu=None, theta=None, pi=None):
        '''
        Sample from the posterior distribution.
        
        Args:
            n_samples: an integer representing the number of samples (Note: if mu is assigned a n*p matrix, then n_sample*n samples will generated in total!)
            mu: a numpy array representing the mean of the negative binomial distribution.
            theta: a numpy array representing the dispersion of the negative binomial distribution
            pi: a numpy array representing the dropout rate
            
        Returns: 
            a numpy array representing the samples
        '''
        if mu is None:
            mu = self.rho * self.library_size
        if theta is None:
            theta = self.dispersion
        if pi is None:
            pi = self.pi
        is_zero = np.random.rand(n_samples, *pi.shape) < pi

        non_zero_samples = np.random.negative_binomial(
            n=theta, p=theta/(theta + mu), size=(n_samples, *mu.shape)
        )
        samples = np.where(is_zero, 0, non_zero_samples)
        samples = samples.squeeze()
        return samples

    def _sample_from_parameter(self,n_samples,inference=False):
        '''
        Generate samples from inferred ZINB parameters.

        Args:
            n_samples: an integer representing the number of samples
            inference: a boolean representing whether to reconduct the inference process each time or simply use inferred parameters stored in the object
        
        Returns:
            a numpy array representing the samples
        
        '''
        if not self.count_data:
            print("Can't do with normalized data")
            return
        if inference:
            _,_,_,_,_,_,rho,dispersion,pi,library_size = \
                zip(*[self.inference(n_samples=1,update=False,returns=True) for _ in range(n_samples)])
            sample = [self.sample_posterior(n_samples=1,mu=rho[i]*library_size[i],theta=dispersion[i],pi=pi[i]) for i in range(n_samples)]
            sample = np.stack(sample,axis=0).squeeze()
        else:
            rho,dispersion,pi,library_size = self.rho,self.dispersion,self.pi,self.library_size
            sample = self.sample_posterior(n_samples,mu=rho*library_size,theta=dispersion,pi=pi)
            sample = sample.squeeze()
        return sample
    
    def get_parameter_from_latent(self,latent,covariates=None,batch_size=32):
        '''
        Generate generative model (ZINB) parameters from latent embeddings
        
        Args:
            latent: torch Tensor containing latent factors, (batch_size, latent_dimensions)
            covariates: torch Tensor containing one-hot encoded covariates, (batch_size, covariate_dimensions). Default: None
        
        Returns:
            a tuple, ZINB paramters rho, dispersion, and pi in generative models
        '''
        latent_dataset = LatentDataset(latent,covariates)
        latent_data = DataLoader(latent_dataset,batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
        if self.count_data:
            rho_list, dispersion_list, pi_list = [],[],[]
            for _,(z,c) in enumerate(latent_data):
                z,c = z.to(self.device),c.to(self.device)
                self.model.eval()
                rho, dispersion, pi = self.model.decoder(z,c)
                rho_list.append(rho.detach().cpu().numpy())
                dispersion_list.append(dispersion.detach().cpu().numpy())
                pi_list.append(pi.detach().cpu().numpy())
            rho = np.concatenate(rho_list, axis=0)
            dispersion = np.mean(dispersion_list, axis=0) # dispersion is gene-based
            pi = np.concatenate(pi_list, axis=0)
            return rho, dispersion, pi
        else:
            rho_list = []
            for _,(z,c) in enumerate(latent_data):
                z,c = z.to(self.device),c.to(self.device)
                self.model.eval()
                rho = self.model.decoder(z,c)
                rho_list.append(rho.detach().cpu().numpy())
            rho = np.concatenate(rho_list, axis=0)
            return rho
                    
    def _generate_from_latent(self,latent,covariates=None,library_size=None,n_samples=1):
        '''
        Generate samples from latent embeddings

        Args:
            latent: torch Tensor containing latent factors, (sample_size, latent_dimensions)
            covariates: torch Tensor containing one-hot encoded covariates, (sample_size, covariate_dimensions). Default: None
            library_size: torch Tensor containing library sizes for new generate samples, (sample_size,1). Inferred library size of original adata will be used if None
            n_samples: an integer representing the number of samples (Note: if latent is assigned a n*p matrix, then n_samples*n samples will generated in total!)
        
        Returns:
            a numpy array representing the generated samples
        '''
        if self.count_data:
            if library_size is None:
                library_size = self.library_size
            rho, dispersion, pi = self.get_parameter_from_latent(latent,covariates)
            sample = self.sample_posterior(n_samples,mu=rho*library_size,theta=dispersion,pi=pi)
        else:
            sample = self.get_parameter_from_latent(latent,covariates)
        return sample
    
    def _generate_latent(self,perturbations,celltype,batch_size=32):
        '''
        Generate latent embeddings from perturbations and cell types.

        Args:
            perturbations: a numpy array representing one-hot encoded perturbations
            celltype: a numpy array representing one-hot encoded cell types
            batch_size: an integer representing the batch size for DataLoader

        Returns: 
            a numpy array representing the generated latent embeddings
        '''
        label_dataset = LabelDataset(perturbations,celltype)
        label_data = DataLoader(label_dataset,batch_size,shuffle=False,drop_last=False,num_workers=4,pin_memory=True)
        zd_list = []
        for _,(u,t) in enumerate(label_data):
            u,t = u.to(self.device),t.to(self.device)
            self.model.eval()
            if self.celltype_num>0:
                treatment_zd = self.model.prior_zd(torch.cat([u,t],dim=-1))
            else:
                treatment_zd = self.model.prior_zd(u)
            mu_treatment_zd, logvar_treatment_zd = treatment_zd.chunk(2, dim=-1)
            # treatment_zd = self.model.reparameterize(mu_treatment_zd, logvar_treatment_zd).detach().cpu().numpy()
            # zd_list.append(treatment_zd)
            zd_list.append(mu_treatment_zd.detach().cpu().numpy())
        zd_list = np.concatenate(zd_list, axis=0)
        return zd_list
        # pass
    
    def get_adata(self):
        '''
        Get the adata with latent variables and estimated generative parameters.
        
        Returns: 
            an AnnData object with latent embeddings and estimated ZINB generative parameters.
        '''
        return self.adata

    def latent_adaption(self,control_zd, treatment_zd, control_zu, treatment_zu,
                            control_celltype, treatment_celltype,strategy="ot", 
                            alpha = 1, beta = 1, projection_strategy = "full",
                              value = "raw",
                              method="emd", reg=0.01, reg_m=1.0):
        '''
        Generate counterfactual latent embeddings for a pair of control and treatment. 

        Args:
            control_zd: a numpy array representing the dependent latent embeddings of control group
            treatment_zd: a numpy array representing the dependent latent embeddings of treatment group
            control_zu: a numpy array representing the independent latent embeddings of control group
            treatment_zu: a numpy array representing the independent latent embeddings of treatment group
            control_celltype: a numpy array representing the one-hot encoded cell types of control group
            treatment_celltype: a numpy array representing the one-hot encoded cell types of treatment group
            strategy: a string representing the strategy to generate counterfactual latent embeddings. Options:
                - "ot": use optimal transport to estimate the latent shift
                - "average": use average effect to estimate the latent shift
            alpha: a float number representing the weight for zu in optimal transport. Default is 1.
            beta: a float number representing the weight for zd in optimal transport. Default is 1.
            projection_strategy: a string representing the projection strategy in optimal transport. Options:
                - "full": use full latent embeddings for optimal transport
                - "zd_only": transport only on dependent latent embeddings with independent latent embeddings fixed
            value: a string representing the value type in optimal transport. Options:
                - "raw": use raw latent embeddings for optimal transport
                - "quantile": use quantile transformed latent embeddings for optimal transport
            method: a string representing the optimal transport method. Options:
                - "emd": Exact Optimal Transport
                - "sinkhorn": Sinkhorn Regularized OT
                - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
                Defaults to "emd".
            reg: a float number representing the entropic regularization parameter for Sinkhorn. Default is 0.01. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
            reg_m: a float number representing the marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".
        
        Returns:
            counterfactual_z: a numpy array representing the counterfactual latent embeddings
            counterfactual_zd: a numpy array representing the counterfactual dependent latent embeddings
            counterfactual_zu: a numpy array representing the counterfactual independent latent embeddings
        '''
        if strategy=="ot":
            counterfactual_z, W = conditional_OT_latent(control_zd, treatment_zd, control_zu, treatment_zu,
                            control_celltype, treatment_celltype, alpha=alpha, beta=beta, 
                            projection_strategy=projection_strategy, value=value,
                            method=method, reg=reg, reg_m=reg_m)
        elif strategy=="average":
            counterfactual_zd = add_effect(control_zd, treatment_zd, control_celltype, treatment_celltype)
            counterfactual_z = np.concatenate([counterfactual_zd,control_zu],axis=-1)
        else:
            raise ValueError("Unknown strategy!")
        
        if np.any(np.isnan(counterfactual_z)): 
            raise ValueError("NaN in counterfactual_z!")
        
        counterfactual_zd = counterfactual_z[:,:self.latent_dependent]
        counterfactual_zu = counterfactual_z[:,self.latent_dependent:]
        # print(counterfactual_zd.shape, counterfactual_zu.shape, counterfactual_z.shape)
        return counterfactual_z, counterfactual_zd, counterfactual_zu

    def get_counterfactual_latent(self, control, treatment, dose=None,
                              strategy="ot", alpha = 1, beta = 1, projection_strategy = "full",
                              value = "raw",
                              method="emd", reg=0.01, reg_m=1.0):
        '''
        Generate counterfactual latent embeddings for a pair of control and treatment.  

        Args:
            control: a string representing the name of control group (should be a value in adata.obs[perturbation_key]).
            treatment: a string representing the name of treatment group (should be a value in adata.obs[perturbation_key]).
            dose: a float representing the dose level (should be a value in adata.obs[dose_key]). If None, all doses will be considered.
            strategy: a string representing the strategy to generate counterfactual latent embeddings. Options:
                - "ot": use optimal transport to estimate the latent shift
                - "average": use average effect to estimate the latent shift
            alpha: a float number representing the weight for zu in optimal transport. Default is 1.
            beta: a float number representing the weight for zd in optimal transport. Default is 1.
            projection_strategy: a string representing the projection strategy in optimal transport. Options:
                - "full": use full latent embeddings for optimal transport
                - "zd_only": transport only on dependent latent embeddings with independent latent embeddings fixed
            value: a string representing the value type in optimal transport. Options:
                - "raw": use raw latent embeddings for optimal transport
                - "quantile": use quantile transformed latent embeddings for optimal transport
            method: a string representing the optimal transport method. Options:
                - "emd": Exact Optimal Transport
                - "sinkhorn": Sinkhorn Regularized OT
                - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
                Defaults to "emd".
            reg: a float number representing the entropic regularization parameter for Sinkhorn. Default is 0.01. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
            reg_m: a float number representing the marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".
        
        Returns: 
            a tuple of numpy arrays representing the counterfactual latent embeddings (z,zd,zu)
        '''
        control_id = np.where(self.perturbation_label == control)[0]
        if dose is not None:
            treatment_id = np.where((self.perturbation_label == treatment)&(self.dose.flatten() == dose))[0]
        else:
            treatment_id = np.where(self.perturbation_label == treatment)[0]
        if len(control_id) == 0 or len(treatment_id) == 0:
            return None
        
        control_zd = self.mu_d[control_id]
        treatment_zd = self.mu_d[treatment_id]
        control_zu = self.mu_u[control_id]
        treatment_zu = self.mu_u[treatment_id]

        control_rho = self.rho[control_id,:]
        treatment_rho = self.rho[treatment_id,:]
        
        if self.celltype is not None:
            control_celltype = self.celltype_label[control_id]
            treatment_celltype = self.celltype_label[treatment_id]
        else:
            control_celltype, treatment_celltype = None, None

        counterfactual_z, counterfactual_zd, counterfactual_zu = self.latent_adaption(
            control_zd, treatment_zd, control_zu, treatment_zu,
            control_celltype, treatment_celltype,strategy=strategy, 
            alpha = alpha, beta = beta, projection_strategy = projection_strategy,
            value = value,
            method=method, reg=reg, reg_m=reg_m)
        return counterfactual_z, counterfactual_zd, counterfactual_zu
    
    def effect_estimate(self, control, treatment, dose=None,
                        strategy="ot", alpha = 1, beta = 1, projection_strategy = "full",
                        value = "raw",
                        method="emd", reg=0.01, reg_m=1.0):
        '''
        Estimate the Individual Treatment Effect (ITE) for a pair of control and treatment.
        
        Args:
            control: a string representing the name of control group (should be a value in adata.obs[perturbation_key]).
            treatment: a string representing the name of treatment group (should be a value in adata.obs[perturbation_key]).
            method (str, optional): Optimal transport method. Options:
                - "emd": Exact Optimal Transport
                - "sinkhorn": Sinkhorn Regularized OT
                - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
                Defaults to "emd".
            reg (float, optional): Entropic regularization parameter for Sinkhorn. Default is 0.1. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
            reg_m (float, optional): Marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".
        
        Returns: 
            a numpy array representing the ITE
        '''
        control_id = np.where(self.perturbation_label == control)[0]
        control_rho = self.rho[control_id]
        counterfactual_z,_,_ = self.get_counterfactual_latent(control, treatment, dose=dose, 
                                                      strategy=strategy, alpha=alpha, beta=beta, projection_strategy=projection_strategy,
                                                      value=value,method=method, reg=reg, reg_m=reg_m)
        if self.count_data:
            rho_counterfactual,_,_ = self.get_parameter_from_latent(counterfactual_z, self.covariates)
            # ite_pred = rho_counterfactual - control_rho
        else:
            rho_counterfactual = self.get_parameter_from_latent(counterfactual_z, self.covariates)
        
        # self.adata.uns['matching_matrix'] = W.copy()
        ITE = rho_counterfactual - control_rho
        return ITE

    def counterfactual_samples(self, control, treatment, dose=None,
                               strategy="ot", alpha = 1, beta = 1, projection_strategy = "full",
                               value = "raw",
                               method="sinkhorn", reg=0.01, reg_m=1.0):
        '''
        Estimate the Individual Treatment Effect (ITE) for a pair of control and treatment.
        
        Args:
            control: a string representing the name of control group (should be a value in adata.obs[perturbation_key]).
            treatment: a string representing the name of treatment group (should be a value in adata.obs[perturbation_key]).
            dose: a float representing the dose level (should be a value in adata.obs[dose_key]). If None, all doses will be considered.
            strategy: a string representing the counterfactual generation strategy. Options:
                - "ot": Optimal Transport
                - "average": Average Effect Addition
            alpha: a float representing the weight for zu embeddings in OT calculation. Default is 1.
            beta: a float representing the weight for zd embeddings in OT calculation. Default is 1.
            projection_strategy: a string representing the projection strategy for OT calculation. Options:
                - "full": use full embeddings for OT calculation
                - "zd_only": project zd embeddings while keep zu embeddings unchanged
            value: a string representing the value type for OT cost calculation. Options:
                - "raw": use raw embeddings for OT cost calculation
                - "quantile": use quantile normalized embeddings for OT cost calculation
            method (str, optional): Optimal transport method. Options:
                - "emd": Exact Optimal Transport
                - "sinkhorn": Sinkhorn Regularized OT
                - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
                Defaults to "emd".
            reg (float, optional): Entropic regularization parameter for Sinkhorn. Default is 0.1. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
            reg_m (float, optional): Marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".
        
        Returns: 
            a numpy array representing the ITE
        '''
        control_id = np.where(self.perturbation_label == control)[0]
        if dose is not None:
            treatment_id = np.where((self.perturbation_label == treatment)&(self.dose.flatten() == dose))[0]
        else:
            treatment_id = np.where(self.perturbation_label == treatment)[0]
        if len(control_id) == 0 or len(treatment_id) == 0:
            return None
        
        counterfactual_z, _, _ = self.get_counterfactual_latent(control, treatment, dose=dose,
                        strategy=strategy, alpha=alpha, beta=beta, projection_strategy=projection_strategy,
                        value=value,
                        method=method, reg=reg, reg_m=reg_m)
        
        if np.any(np.isnan(counterfactual_z)):
            print("NaN in counterfactual_z!")
            raise ValueError("NaN in counterfactual_z!")
        if self.covariates is not None:
            covariates = self.covariates[control_id]
        else:
            covariates = None
        if self.count_data:
            rho_counterfactual,_,_ = self.get_parameter_from_latent(counterfactual_z, covariates)
            dispersion = self.dispersion
            pi = self.pi[control_id]
            library_size = self.library_size[control_id,:]
            sample = self.sample_posterior(n_samples=1, mu=rho_counterfactual*library_size, theta=dispersion, pi=pi)
            sample = sample.squeeze()
        else:
            rho_counterfactual = self.get_parameter_from_latent(counterfactual_z, covariates)
            sample = rho_counterfactual
        return sample

    def _representative_latent(self, mu, logvar):
        '''
        Generate representative latent embeddings by sampling from the learned distribution.

        Args:
            mu: a numpy array representing the mean of the latent distribution
            logvar: a numpy array representing the log variance of the latent distribution

        Returns:
            a numpy array representing the sampled latent embeddings
        '''
        std = np.exp(0.5 * logvar)
        eps = np.random.normal(0, 1, size=std.shape)
        return mu + eps * std
    
    def get_counterfactual_adata(self, control, treatment, dose=None, covariates=None, strategy="ot", 
                                 alpha = 1, beta = 1, projection_strategy = "full",value = "raw",
                                 method="sinkhorn", reg=0.01, reg_m=1.0):
        '''
        Generate counterfactual AnnData for a pair of control and treatment.

        Args:
            control: a string representing the name of control group (should be a value in adata.obs[perturbation_key]).
            treatment: a string representing the name of treatment group (should be a value in adata.obs[perturbation_key]).
            dose: a float representing the dose level (should be a value in adata.obs[dose_key]). If None, all doses will be considered.
            covariates: a numpy array representing the one-hot encoded covariates for control group. If None, inferred covariates will be used.
            strategy: a string representing the counterfactual generation strategy. Options:
                - "ot": Optimal Transport
                - "average": Average Effect Addition
            alpha: a float representing the weight for zu embeddings in OT calculation. Default is 1.
            beta: a float representing the weight for zd embeddings in OT calculation. Default is 1.
            projection_strategy: a string representing the projection strategy for OT calculation. Options:
                - "full": use full embeddings for OT calculation
                - "zd_only": project zd embeddings while keep zu embeddings unchanged
            value: a string representing the value type for OT cost calculation. Options:
                - "raw": use raw embeddings for OT cost calculation
                - "quantile": use quantile normalized embeddings for OT cost calculation
            method (str, optional): Optimal transport method. Options:
                - "emd": Exact Optimal Transport
                - "sinkhorn": Sinkhorn Regularized OT
                - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
                Defaults to "emd".
            reg (float, optional): Entropic regularization parameter for Sinkhorn. Default is 0.1. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
            reg_m (float, optional): Marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".

        Returns: 
            an AnnData object representing the counterfactual samples
        '''
        adata_cf = self.adata[self.perturbation_label==control].copy()
        sample = self.counterfactual_samples(control, treatment, dose=dose, # covariates=covariates,
                                            strategy=strategy, alpha=alpha, beta=beta, 
                                            projection_strategy=projection_strategy,value=value,
                                            method=method, reg=reg, reg_m=reg_m)
        adata_cf.X = sample.copy()
        adata_cf.obs[self.perturbation_key] = treatment+"_counterfactual"
        if dose is not None:
            adata_cf.obs[self.dose_key] = dose
        else:
            adata_cf.obs[self.dose_key] = np.nan
        return adata_cf

    
    def joint_effect_estimate(self, control, treatment1, treatment2, dose1=None, dose2=None, 
                        strategy="ot", alpha = 1, beta = 1, projection_strategy = "full",
                        value = "raw",
                        method="sinkhorn", reg=0.01, reg_m=1.0):
        '''
        Estimate the joint effect for a pair of treatments compared to a control.

        Args:
            control: a string representing the name of control group (should be a value in adata.obs[perturbation_key]).
            treatment1: a string representing the name of first treatment group (should be a value in adata.obs[perturbation_key]).
            treatment2: a string representing the name of second treatment group (should be a value in adata.obs[perturbation_key]).
            dose1: a float representing the dose level for first treatment (should be a value in adata.obs[dose_key]). If None, all doses will be considered.
            dose2: a float representing the dose level for second treatment (should be a value in adata.obs[dose_key]). If None, all doses will be considered.
            strategy: a string representing the counterfactual generation strategy. Options:
                - "ot": Optimal Transport
                - "average": Average Effect Addition
            alpha: a float representing the weight for zu embeddings in OT calculation. Default is 1.
            beta: a float representing the weight for zd embeddings in OT calculation. Default is 1.
            projection_strategy: a string representing the projection strategy for OT calculation. Options:
                - "full": use full embeddings for OT calculation
                - "zd_only": project zd embeddings while keep zu embeddings unchanged
            value: a string representing the value type for OT cost calculation. Options:
                - "raw": use raw embeddings for OT cost calculation
                - "quantile": use quantile normalized embeddings for OT cost calculation
            method (str, optional): Optimal transport method. Options:
                - "emd": Exact Optimal Transport
                - "sinkhorn": Sinkhorn Regularized OT
                - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
                Defaults to "emd".
            reg (float, optional): Entropic regularization parameter for Sinkhorn. Default is 0.1. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
            reg_m (float, optional): Marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".

        Returns: 
            a numpy array representing the joint effect
        '''
        ite1 = self.effect_estimate(control, treatment1, dose=dose1, 
                        strategy=strategy, alpha=alpha, beta=beta, 
                        projection_strategy=projection_strategy,
                        value=value,
                        method=method, reg=reg, reg_m=reg_m)
        ite2 = self.effect_estimate(control, treatment2, dose=dose2,
                        strategy=strategy, alpha=alpha, beta=beta, 
                        projection_strategy=projection_strategy,
                        value=value,
                        method=method, reg=reg, reg_m=reg_m)
        if ite1 is None or ite2 is None:
            return None
        def synergistic_effect(effect_a,effect_b,celltype=None):
            if celltype is None:
                correlation, _ = pearsonr(effect_a, effect_b)
                joint_effects = np.sqrt(effect_a**2 + effect_b**2 + 2 * effect_a * effect_b * correlation)
            else:
                joint_effects = np.zeros_like(effect_a)
                unique_celltypes = np.unique(celltype)
                for ct in unique_celltypes:
                    indices = np.where(celltype == ct)[0]
                    effect_a1 = effect_a[indices]
                    effect_b1 = effect_b[indices]
                    correlation, _ = pearsonr(effect_a1, effect_b1)
                    joint_effects[indices] = np.sqrt(effect_a1**2 + effect_b1**2 + 2 * effect_a1 * effect_b1 * correlation)
            return joint_effects
        if self.celltype_num > 0:
            control_celltype = self.celltype_label[np.where(self.perturbation_label == control)[0]]
            joint_effects = np.array(list(map(
                lambda i: synergistic_effect(ite1[:, i], ite2[:, i], control_celltype),
                range(ite1.shape[1])
            ))).T 
            # joint_effects = np.apply_along_axis(synergistic_effect, axis=0, arr=np.array([ite1, ite2]), celltype=control_celltype)
        else:
            joint_effects = np.array(list(map(
                lambda i: synergistic_effect(ite1[:, i], ite2[:, i]),
                range(ite1.shape[1])
            ))).T 
            # joint_effects = np.apply_along_axis(synergistic_effect, axis=0, arr=np.array([ite1, ite2]))
        return joint_effects
    
    def _fit_dose_to_dist(self, v, zd, method='gpr'):
        """
        Fit models to map dosage values to latent distributions.

        Args:
            v: a numpy array representing the dosage values.
            zd: a numpy array representing the latent embeddings.
            method: a string representing the method to fit dosage to latent distribution. Options are 'linear', 'spline', or 'gpr'. Defaults to 'gpr'.
        
        Returns:
            a DosageModel object containing the fitted models.
        """
        unique_doses, inverse_indices = np.unique(v, return_inverse=True)
        mean_zd_per_dose = np.array([np.mean(zd[inverse_indices == i], axis=0) for i in range(len(unique_doses))])
        std_zd_per_dose = np.array([np.std(zd[inverse_indices == i], axis=0) for i in range(len(unique_doses))])

        if method in ['linear', 'spline']:
            models = self._fit_interpolator(unique_doses, mean_zd_per_dose, std_zd_per_dose, method)
        elif method == 'gpr':
            models = self._fit_gpr(unique_doses, mean_zd_per_dose, std_zd_per_dose)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'linear', 'spline', or 'gpr'.")

        return DosageModel(models)

    def _fit_interpolator(self, doses, means, stds, method):
        """
        Fit interpolators for the given doses, means, and standard deviations.

        Args:
            doses: a numpy array representing the unique dosage values.
            means: a numpy array representing the mean latent embeddings.
            stds: a numpy array representing the standard deviation of latent embeddings.
            method: a string representing the interpolation method. Options are 'linear' or 'spline'.
        
        Returns:
            a dictionary containing the fitted mu and sigma models.
        """
        zd_dim = means.shape[1]
        mu_models, sigma_models = [], []
        
        # Determine the degree of the spline based on the method and number of unique doses
        # 1: linear, 3: cubic
        k = 1 if method == 'linear' else (3 if len(doses) >= 4 else 1)
        if method == 'spline' and k == 1:
            print("Warning: Not enough unique points for cubic spline, falling back to linear (k=1).")
            
        for i in range(zd_dim):
            # --- fit mu ---
            # 1. Create B-spline object using make_interp_spline
            spline_mu_obj = make_interp_spline(doses, means[:, i], k=k)
            # 2. Wrap with lambda to enable extrapolation by default
            mu_model = lambda x: spline_mu_obj(x, extrapolate=True)
            mu_models.append(mu_model)

            # --- fit sigma ---
            log_sigma = np.log(stds[:, i] + 1e-8)
            spline_sigma_obj = make_interp_spline(doses, log_sigma, k=k)
            sigma_model = lambda x: spline_sigma_obj(x, extrapolate=True)
            sigma_models.append(sigma_model)
            
        return {'mu': mu_models, 'sigma': sigma_models}

    def _fit_gpr(self, doses, means, stds):
        """
        Fit Gaussian Process Regressors for the given doses, means, and standard deviations.

        Args:
            doses: a numpy array representing the unique dosage values.
            means: a numpy array representing the mean latent embeddings.
            stds: a numpy array representing the standard deviation of latent embeddings.
        
        Returns:
            a dictionary containing the fitted mu and sigma models.
        """
        zd_dim = means.shape[1]
        mu_models, sigma_models = [], []
        doses_reshaped = doses.reshape(-1, 1)
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
        
        for i in range(zd_dim):
            gp_mu = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, random_state=0)
            gp_mu.fit(doses_reshaped, means[:, i])
            mu_models.append(gp_mu)
            
            log_sigma = np.log(stds[:, i] + 1e-8)
            gp_sigma = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, random_state=0)
            gp_sigma.fit(doses_reshaped, log_sigma)
            sigma_models.append(gp_sigma)

        return {'mu': mu_models, 'sigma': sigma_models}


    def dosage_extrapolate(self, control, treatment, dose_value=None, method="gpr"):
        """
        Extrapolate latent embeddings for control group to new dosage levels based on treatment group data.
        
        Args:
            control: a string representing the name of control group (should be a value in adata.obs[perturbation_key]).
            treatment: a string representing the name of treatment group (should be a value in adata.obs[perturbation_key]).
            dose_value: a scalar or array-like representing the dosage value(s) for extrapolation.
            method: a string representing the method to fit dosage to latent distribution. Options are 'linear', 'spline', or 'gpr'. Defaults to 'gpr'.
        
        Returns: 
            a numpy array or a list of numpy arrays representing the counterfactual mean expressions at the specified dosage level(s).
        """
        # --- 1. prepare data ---
        control_id = np.where(self.perturbation_label == control)[0]
        treatment_id = np.where(self.perturbation_label == treatment)[0]
        if len(control_id) == 0 or len(treatment_id) == 0:
            print(f"Warning: Control ('{control}') or Treatment ('{treatment}') not found.")
            return None

        if (dose_value is None) or (self.dose is None):
            print("No dosage information provided.")
            return None

        # --- 2. fit models ---
        models_by_celltype = {}
        global_dist_model = None

        if self.celltype_num > 0:
            control_celltype_idx = np.argmax(self.celltype[control_id], axis=1)
            treatment_celltype_idx = np.argmax(self.celltype[treatment_id], axis=1)
            
            for ct in np.unique(control_celltype_idx):
                treatment_ct_mask = (treatment_celltype_idx == ct)
                if not np.any(treatment_ct_mask):
                    models_by_celltype[ct] = None
                    continue
                
                v_ct = self.dose[treatment_id][treatment_ct_mask].flatten()
                zd_ct = self.zd[treatment_id][treatment_ct_mask]


                models_by_celltype[ct] = self._fit_dose_to_dist(v_ct, zd_ct, method=method)
                # models_by_celltype[ct] = self._fit_dose_to_dist(v_ct, zd_ct)
        else:
            print("No cell type information, using global dosage model.")
            v = self.dose[treatment_id].flatten()
            zd = self.zd[treatment_id]
            global_dist_model = self._fit_dose_to_dist(v, zd, method=method)

        # --- 3. predict ---
        is_scalar = np.ndim(dose_value) == 0
        dose_values = [dose_value] if is_scalar else list(dose_value)
        
        results = []
        control_zd = self.zd[control_id]
        control_zu = self.zu[control_id]
        
        if self.covariates_dim > 0:
            control_covariates = self.covariates[control_id]
        else:
            control_covariates = None

        for d_val in dose_values:
            current_dose_input = np.tile(d_val, control_zd.shape[0]).reshape(-1, 1)
            zd_new = np.zeros_like(control_zd)

            if self.celltype_num > 0:
                for ct in np.unique(control_celltype_idx):
                    control_ct_mask = (control_celltype_idx == ct)
                    dist_model_ct = models_by_celltype[ct]

                    if dist_model_ct is None:
                        print(f"Warning: Cell type {ct} not found in treatment. Using control latent states.")
                        zd_new[control_ct_mask] = control_zd[control_ct_mask]
                        continue
                    
                    dose_value_ct = current_dose_input[control_ct_mask] #np.tile(d_val, np.sum(control_ct_mask))
                    mu_new_ct, sigma_new_ct = dist_model_ct.predict(dose_value_ct.flatten())

                    control_zd_ct = control_zd[control_ct_mask]
                    control_mu_ct = np.mean(control_zd_ct, axis=0, keepdims=True)
                    control_sigma_ct = np.std(control_zd_ct, axis=0, keepdims=True)
                    control_sigma_ct[control_sigma_ct == 0] = 1.0
                    standardized_eps_control_ct = (control_zd_ct - control_mu_ct) / control_sigma_ct
                    eps_new_ct = standardized_eps_control_ct * sigma_new_ct
                    zd_new[control_ct_mask] = mu_new_ct + eps_new_ct
            else:

                mu_new, sigma_new = global_dist_model.predict(current_dose_input.flatten())
                
                control_mu = np.mean(control_zd, axis=0, keepdims=True)
                control_sigma = np.std(control_zd, axis=0, keepdims=True)
                control_sigma[control_sigma == 0] = 1.0
                standardized_eps_control = (control_zd - control_mu) / control_sigma
                eps_new = standardized_eps_control * sigma_new
                zd_new = mu_new + eps_new

            # --- 4. generate counterfactual ---
            zu_new = control_zu.copy()
            counterfactual_z = np.concatenate([zd_new, zu_new], axis=-1)
            # print("z",counterfactual_z)

            if self.count_data:
                rho_counterfactual, _, _ = self.get_parameter_from_latent(
                    latent=counterfactual_z, covariates=control_covariates
                )
            else:
                rho_counterfactual = self.get_parameter_from_latent(
                    latent=counterfactual_z, covariates=control_covariates
                )
            
            results.append(rho_counterfactual)

        # --- 5. return ---
        return results[0] if is_scalar else results

