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
from .utils import condition_quantile,groupwise_OT,groupwise_OT_rank,groupwise_OT_latent,to_dense_array,add_effect
import pickle
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

class Perturb:
    def __init__(self, adata, layer=None, perturbation_key="perturbation", celltype_key=None, 
                 batch_key=None, dose_key=None, count_data=False):
        '''
        Initialize the Perturb object.
        
        Args:
            adata: an AnnData object    
            layer: a string representing the layer name of count matrix in adata
            perturbation_key: a string representing the key of perturbation in adata.obs
            celltype_key: a string representing the key of cell type in adata.obs
            batch_key: a string representing the key of batch in adata.obs
        '''
        super(Perturb, self).__init__()
        self.adata = adata
        if layer is None:
            self.data = to_dense_array(self.adata.X)
        else:
            self.data = to_dense_array(self.adata.layers[layer])
        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse_output=False)
        self.perturbation_label = adata.obs[perturbation_key].to_numpy()
        self.perturbation = label_encoder.fit_transform(self.perturbation_label)
        self.perturbation = onehot_encoder.fit_transform(self.perturbation.reshape(-1, 1))
        if celltype_key is None:
            self.celltype = None
        else:
            self.celltype_label = adata.obs[celltype_key].to_numpy()
            self.celltype = label_encoder.fit_transform(self.celltype_label)
            self.celltype = onehot_encoder.fit_transform(self.celltype.reshape(-1, 1))
        if batch_key is None:
            self.covariates = None
        else:
            self.covariates_label = adata.obs[batch_key].to_numpy()
            self.covariates = label_encoder.fit_transform(self.covariates_label)
            self.covariates = onehot_encoder.fit_transform(self.covariates.reshape(-1, 1))
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
        
        self.count_data = count_data
    
    def setup(self, hidden_layers = [128,128], latent_dependent = 20, latent_independent = 20, 
              beta = 5, sparse_coef = 0, l1_latent = 0, library_size_strategy="observed", device=None):
        '''
        Setup the model for training.
        
        Args:
            hidden_layers: a list of integers representing the number of neurons in each hidden layer
            latent_dependent: an integer representing the dimensions of z_D
            latent_independent: an integer representing the dimensions of z_I
            beta: a float number representing the weight of the KL divergence term
            sparse_coef: a float number representing the weight of the sparsity term
            
        '''
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        print("using "+self.device)
        self.input_dim = self.data.shape[1]
        self.hidden_layers = hidden_layers
        self.latent_dependent = latent_dependent
        self.latent_independent = latent_independent
        self.beta = beta
        self.sparse_coef = sparse_coef
        self.l1_latent = l1_latent
        self.library_size_strategy = library_size_strategy
        self.model = PerturbNet(self.device, input_dim=self.input_dim, covariate_dim = self.covariates_dim, 
                    perturbation_num = self.perturbation_num, celltype_num = self.celltype_num,
                    layer_dims = self.hidden_layers, latent_dep_dim = self.latent_dependent, 
                    latent_ind_dim = self.latent_independent, 
                    dropout_rate = 0.5, lambda_sparse = self.sparse_coef, l1_latent = self.l1_latent,
                    beta = self.beta, count_data=self.count_data, library_size_strategy=self.library_size_strategy,
                    eps=1e-10)
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
    
    def inference(self, n_samples=1, dataset=None, batch_size=None, update=True, returns=True):
        '''
        Perform inference.
        
        Args:
            n_samples: an integer representing the number of samples repeated for inference process
            dataset: a CombinedDataset object. By default we use the training dataset
            batch_size: an integer representing the batch size
            update: a boolean representing whether to update the adata
            returns: a boolean representing whether to return the results
            
        Returns: 
            a tuple of numpy arrays representing the latent variables (z_d,z_i,mu_d,mu_i,rho,dispersion,dropout_rate,library_size)
        '''
        if dataset is None:
            dataset = self.train_dataset
        if batch_size is None:
            batch_size = self.batch_size
        if self.count_data:
            if n_samples > 1:
                zd,zi,mu_d,mu_i,logvar_d,logvar_i, rho,dispersion,pi,library_size = \
                    zip(*[inference_model(self.device, dataset, self.model, batch_size, self.count_data) for _ in range(n_samples)])
                self.zd = np.mean(np.stack(zd, axis=0), axis=0) 
                self.zi = np.mean(np.stack(zi, axis=0), axis=0) 
                self.mu_d = np.mean(np.stack(mu_d, axis=0), axis=0) 
                self.mu_i = np.mean(np.stack(mu_i, axis=0), axis=0) 
                self.logvar_d = np.mean(np.stack(logvar_d, axis=0), axis=0) 
                self.logvar_i = np.mean(np.stack(logvar_i, axis=0), axis=0) 
                self.rho = np.mean(np.stack(rho, axis=0), axis=0) 
                self.dispersion = np.mean(np.stack(dispersion, axis=0), axis=0) 
                self.pi = np.mean(np.stack(pi, axis=0), axis=0) 
                self.library_size = np.mean(np.stack(library_size, axis=0), axis=0) 
            else:
                self.zd,self.zi,self.mu_d,self.mu_i,self.logvar_d, self.logvar_i, self.rho,self.dispersion,self.pi,self.library_size = \
                    inference_model(self.device, dataset, self.model, batch_size, self.count_data)
            if update:
                self.adata.obsm['latent_dependent'] = self.zd
                self.adata.obsm['latent_independent'] = self.zi
                self.adata.obsm['mu_latent_dependent'] = self.mu_d
                self.adata.obsm['mu_latent_independent'] = self.mu_i
                self.adata.obsm['logvar_latent_dependent'] = self.logvar_d
                self.adata.obsm['logvar_latent_independent'] = self.logvar_i
                self.adata.layers['estimated_mean_expression'] = self.rho
                self.adata.layers['estimated_dropout_rate'] = self.pi
                self.adata.var['estimated_dispersion_factor'] = self.dispersion
                self.adata.obs['estimated_library_size'] = self.library_size
                print('All results recorded in adata.')
            if returns:
                return self.zd,self.zi,self.mu_d,self.mu_i,self.logvar_d,self.logvar_i,self.rho,self.dispersion,self.pi,self.library_size
        else:
            if n_samples > 1:
                zd,zi,mu_d,mu_i,logvar_d,logvar_i,rho = \
                    zip(*[inference_model(self.device, dataset, self.model, batch_size, self.count_data) for _ in range(n_samples)])
                self.zd = np.mean(np.stack(zd, axis=0), axis=0) 
                self.zi = np.mean(np.stack(zi, axis=0), axis=0) 
                self.mu_d = np.mean(np.stack(mu_d, axis=0), axis=0) 
                self.mu_i = np.mean(np.stack(mu_i, axis=0), axis=0) 
                self.logvar_d = np.mean(np.stack(logvar_d, axis=0), axis=0) 
                self.logvar_i = np.mean(np.stack(logvar_i, axis=0), axis=0) 
                self.rho = np.mean(np.stack(rho, axis=0), axis=0) 
            else:
                self.zd,self.zi,self.mu_d,self.mu_i,self.logvar_d, self.logvar_i, self.rho = \
                    inference_model(self.device, dataset, self.model, batch_size, self.count_data)
            if update:
                self.adata.obsm['latent_dependent'] = self.zd
                self.adata.obsm['latent_independent'] = self.zi
                self.adata.obsm['mu_latent_dependent'] = self.mu_d
                self.adata.obsm['mu_latent_independent'] = self.mu_i
                self.adata.obsm['logvar_latent_dependent'] = self.logvar_d
                self.adata.obsm['logvar_latent_independent'] = self.logvar_i
                self.adata.layers['estimated_mean_expression'] = self.rho
                print('All results recorded in adata.')
            if returns:
                return self.zd,self.zi,self.mu_d,self.mu_i,self.logvar_d,self.logvar_i,self.rho
    
    # def save(self, path):
    #     state = {name: module.state_dict() for name, module in self.__dict__.items() if isinstance(module, nn.Module)}
    #     torch.save(state, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    # def load(self, path):
    #     state = torch.load(path, weights_only=False)
    #     for name, module_state in state.items():
    #         getattr(self, name).load_state_dict(module_state)
    
    def save(self, path):
        '''
        Save the trained model.
        '''
        torch.save({
            "model": self.model.state_dict()
        }, path, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path, device=None):
        '''
        Load the trained model.
        '''
        if device is None:
            device = self.device
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        self.model.load_state_dict(checkpoint["model"])

    def get_latent(self):
        '''
        Return the disentangled latent embeddings.
        '''
        return self.zd,self.zi,self.mu_d,self.mu_i

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
        # print("pi:",np.mean(pi))
        # print("proportions:",np.mean(is_zero))
        non_zero_samples = np.random.negative_binomial(
            n=theta, p=1/(1+mu), size=(n_samples, *mu.shape)
        )
        # print("mu:",np.mean(mu))
        # print("rho:",np.mean(non_zero_samples))
        samples = np.where(is_zero, 0, non_zero_samples)
        samples = samples.squeeze()
        # print("samples:",np.mean(samples))
        return samples
    
    def sample_from_parameter(self,n_samples,inference=False):
        '''
        Generate samples from inferered ZINB parameters.
        
        Args:
            n_samples: an integer representing the number of samples
            inference: a boolean representing whether to reconducted the inference process each time or simply use inferred parameters stored in the object
        
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
    
    # def effect_estimate(self, control, treatment, method="sinkhorn", reg=0.1, reg_m=1.0):
    #     '''
    #     Estimate the Individual Treatment Effect (ITE) for a pair of control and treatment.
        
    #     Args:
    #         control: a string representing the name of control group (should be a value in adata.obs[perturbation_key]).
    #         treatment: a string representing the name of treatment group (should be a value in adata.obs[perturbation_key]).
    #         method (str, optional): Optimal transport method. Options:
            
    #             - "emd": Exact Optimal Transport
    #             - "sinkhorn": Sinkhorn Regularized OT
    #             - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
    #             Defaults to "sinkhorn".
    #         reg (float, optional): Entropic regularization parameter for Sinkhorn. Default is 0.1. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
    #         reg_m (float, optional): Marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".
        
    #     Returns: 
    #         a numpy array representing the ITE
    #     '''
    #     control_id = np.where(self.perturbation_label == control)[0]
    #     treatment_id = np.where(self.perturbation_label == treatment)[0]
    #     # control_zd = self.zd[control_id]
    #     # treatment_zd = self.zd[treatment_id]
    #     # control_zi = self.zi[control_id]
    #     # treatment_zi = self.zi[treatment_id]
    #     control_zd = self.mu_d[control_id]
    #     treatment_zd = self.mu_d[treatment_id]
    #     # control_logvard = self.logvar_d[control_id]
    #     control_zi = self.mu_i[control_id]
    #     treatment_zi = self.mu_i[treatment_id]
    #     control_rho = self.rho[control_id]
    #     treatment_rho = self.rho[treatment_id]
        
    #     if self.celltype is not None:
    #         control_celltype = self.celltype_label[control_id]
    #         treatment_celltype = self.celltype_label[treatment_id]
    #     else:
    #         control_celltype, treatment_celltype = None, None
        
    #     W = groupwise_OT(control_zd, treatment_zd,
    #                      control_celltype, treatment_celltype, 
    #                      method=method, reg=reg, reg_m=reg_m)
    #     # W = groupwise_OT_latent(control_zd, control_zi, treatment_zd, treatment_zi, 
    #     #                 control_celltype, treatment_celltype, 
    #     #                 method=method, reg=reg, reg_m=reg_m)
    #     counterfactual_zd = W @ treatment_zd
    #     # counterfactual_zi = W @ treatment_zi
    #     # ite_pred = W @ treatment_rho - control_rho
    #     # self.adata.uns['counterfactual_mu_d'] = counterfactual_zd.copy()
    #     # counterfactual_zd = counterfactual_zd + np.random.normal(0, 1, size=counterfactual_zd.shape) # * np.exp(0.5 * control_logvard)
    #     self.adata.uns['counterfactual_zd'] = counterfactual_zd.copy()
    #     self.adata.uns['matching_matrix'] = W.copy()
        
    #     counterfactual_z = np.concatenate([counterfactual_zd,control_zi],axis=-1)
    #     # counterfactual_z = np.concatenate([counterfactual_zd,counterfactual_zi],axis=-1)
    #     rho_counterfactual,dispersion_counterfactual,pi_counterfactual = self.get_parameter_from_latent(counterfactual_z, self.covariates)
    #     ite_pred = rho_counterfactual - control_rho
        
    #     return ite_pred,rho_counterfactual,dispersion_counterfactual,pi_counterfactual
    
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
            # latent = torch.tensor(latent).float().to(self.device)
            # covariates = torch.tensor(covariates).float().to(self.device)
            # rho, dispersion, pi = self.model.decoder(latent,covariates)
            return rho, dispersion, pi
        else:
            rho_list = []
            for _,(z,c) in enumerate(latent_data):
                z,c = z.to(self.device),c.to(self.device)
                rho = self.model.decoder(z,c)
                rho_list.append(rho.detach().cpu().numpy())
            rho = np.concatenate(rho_list, axis=0)
            return rho
                
    
    def generate_from_latent(self,latent,covariates=None,library_size=None,n_samples=1):
        '''
        Generate samples from latent embeddings
        
        Args:
            latent: torch Tensor containing latent factors, (sample_size, latent_dimensions)
            covariates: torch Tensor containing one-hot encoded covariates, (sample_size, covariate_dimensions). Default: None
            library_size: torch Tensor containing library sizes for new generate samples, (sample_size,1). Inferred library size of original adata will be used if None
            n_sample: an integer representing the number of samples (Note: if latent is assigned a n*p matrix, then n_sample*n samples will generated in total!)
        
        '''
        if self.count_data:
            if library_size is None:
                library_size = self.library_size
            rho, dispersion, pi = self.get_parameter_from_latent(latent,covariates)
            sample = self.sample_posterior(n_samples,mu=rho*library_size,theta=dispersion,pi=pi)
        else:
            sample = self.get_parameter_from_latent(latent,covariates)
        return sample
    
    def generate_latent(self,perturbations,celltype,batch_size=32):
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
            treatment_zd = self.model.reparameterize(mu_treatment_zd, logvar_treatment_zd).detach().cpu()
            zd_list.append(treatment_zd)
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
    
    # def save(self, output_path):
    #     torch.save(self, output_path)
        
    
    def effect_estimate(self, control, treatment, dose=None, #values=None,
                        strategy="minimal_change", method="sinkhorn", reg=0.1, reg_m=1.0):
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
        if dose is not None:
            treatment_id = np.where((self.perturbation_label == treatment)&(self.dose.flatten() == dose))[0]
        else:
            treatment_id = np.where(self.perturbation_label == treatment)[0]
        control_zd = self.zd[control_id]
        treatment_zd = self.zd[treatment_id]
        control_zi = self.zi[control_id]
        treatment_zi = self.zi[treatment_id]

        # if values is None:
        #     if self.count_data:
        #         values = "inferred"
        #     else:
        #         values = "observed"
        # if (values=="inferred") or (self.count_data):
        #     control_rho = self.rho[control_id,:]
        #     treatment_rho = self.rho[treatment_id,:]
        # elif (not self.count_data) and (values=="observed"):
        #     control_rho = to_dense_array(self.adata.X[control_id,:])
        #     treatment_rho = to_dense_array(self.adata.X[treatment_id,:])

        control_rho = self.rho[control_id,:]
        treatment_rho = self.rho[treatment_id,:]
        
        if self.celltype is not None:
            control_celltype = self.celltype_label[control_id]
            treatment_celltype = self.celltype_label[treatment_id]
        else:
            control_celltype, treatment_celltype = None, None
        

        # W = groupwise_OT_latent(control_zd, control_zi, treatment_zd, treatment_zi, 
        #                 control_celltype, treatment_celltype, 
        #                 method=method, reg=reg, reg_m=reg_m)
        # # W = groupwise_OT(control_zd, treatment_zd,
        # #                  control_celltype, treatment_celltype, 
        # #                  method=method, reg=reg, reg_m=reg_m)
        # # W = W / W.sum(axis=1, keepdims=True)
        # row_sums = W.sum(axis=1, keepdims=True)
        # W = np.divide(W, row_sums, where=row_sums != 0)
        # # ite_pred = W @ treatment_rho - control_rho
        # # delta = W @ treatment_zd - control_zd
        # # counterfactual_zd = control_zd + delta
        # counterfactual_zd = W @ treatment_zd
        if strategy=="minimal_change":
            W = groupwise_OT(control_zd, treatment_zd,
                             control_celltype, treatment_celltype, 
                             method=method, reg=reg, reg_m=reg_m)
            row_sums = W.sum(axis=1, keepdims=True)
            W = np.divide(W, row_sums, where=row_sums != 0)
            counterfactual_zd = W @ treatment_zd
        elif strategy=="minimal_change_rank":
            W = groupwise_OT_rank(control_zd, treatment_zd,
                             control_celltype, treatment_celltype, 
                             method=method, reg=reg, reg_m=reg_m)
            row_sums = W.sum(axis=1, keepdims=True)
            W = np.divide(W, row_sums, where=row_sums != 0)
            counterfactual_zd = W @ treatment_zd    
        elif strategy=="matching":
            W = groupwise_OT(control_zi, treatment_zi,
                             control_celltype, treatment_celltype, 
                             method=method, reg=reg, reg_m=reg_m)
            row_sums = W.sum(axis=1, keepdims=True)
            W = np.divide(W, row_sums, where=row_sums != 0)      
            counterfactual_zd = W @ treatment_zd  
        elif strategy=="average":
            counterfactual_zd = add_effect(control_zd, treatment_zd, control_celltype, treatment_celltype)
        
        counterfactual_z = np.concatenate([counterfactual_zd,control_zi],axis=-1)
        if self.count_data:
            rho_counterfactual,dispersion_counterfactual,pi_counterfactual = self.get_parameter_from_latent(counterfactual_z, self.covariates)
            ite_pred = rho_counterfactual - control_rho
        else:
            rho_counterfactual = self.get_parameter_from_latent(counterfactual_z, self.covariates)
        
        # self.adata.uns['matching_matrix'] = W.copy()
        ITE = rho_counterfactual - control_rho
        return ITE

    def counterfactual_samples(self, control, treatment, dose=None, covariates=None, strategy="minimal_change", method="sinkhorn", reg=0.1, reg_m=1.0):
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
        if dose is not None:
            treatment_id = np.where((self.perturbation_label == treatment)&(self.dose.flatten() == dose))[0]
        else:
            treatment_id = np.where(self.perturbation_label == treatment)[0]
        control_zd = self.zd[control_id]
        treatment_zd = self.zd[treatment_id]
        control_zi = self.zi[control_id]
        treatment_zi = self.zi[treatment_id]
        control_rho = self.rho[control_id]
        treatment_rho = self.rho[treatment_id]
        
        if self.celltype is not None:
            control_celltype = self.celltype_label[control_id]
            treatment_celltype = self.celltype_label[treatment_id]
        else:
            control_celltype, treatment_celltype = None, None
        
        if strategy=="minimal_change":
            W = groupwise_OT(control_zd, treatment_zd,
                             control_celltype, treatment_celltype, 
                             method=method, reg=reg, reg_m=reg_m)
            row_sums = W.sum(axis=1, keepdims=True)
            W = np.divide(W, row_sums, where=row_sums != 0)
            counterfactual_zd = W @ treatment_zd
        elif strategy=="minimal_change_rank":
            W = groupwise_OT_rank(control_zd, treatment_zd,
                             control_celltype, treatment_celltype, 
                             method=method, reg=reg, reg_m=reg_m)
            row_sums = W.sum(axis=1, keepdims=True)
            W = np.divide(W, row_sums, where=row_sums != 0)
            counterfactual_zd = W @ treatment_zd    
        elif strategy=="matching":
            W = groupwise_OT(control_zi, treatment_zi,
                             control_celltype, treatment_celltype, 
                             method=method, reg=reg, reg_m=reg_m)
            row_sums = W.sum(axis=1, keepdims=True)
            W = np.divide(W, row_sums, where=row_sums != 0)      
            counterfactual_zd = W @ treatment_zd  
        elif strategy=="average":
            counterfactual_zd = add_effect(control_zd, treatment_zd, control_celltype, treatment_celltype)
        
        counterfactual_z = np.concatenate([counterfactual_zd,control_zi],axis=-1)
        library_size = self.library_size[control_id,:]
        sample = self.generate_from_latent(counterfactual_z,covariates,library_size=library_size)
        return sample
        # if self.count_data:
        #     rho_counterfactual,dispersion_counterfactual,pi_counterfactual = self.get_parameter_from_latent(counterfactual_z, self.covariates)
        #     # ite_pred = rho_counterfactual - control_rho
        # else:
        #     rho_counterfactual = self.get_parameter_from_latent(counterfactual_z, self.covariates)
        
        # # self.adata.uns['matching_matrix'] = W.copy()
        # ITE = rho_counterfactual - control_rho
        # return ITE
    
    def dose_prediction(self, control, treatment, dose_value=None):
        control_id = np.where(self.perturbation_label == control)[0]
        treatment_id = np.where(self.perturbation_label == treatment)[0]
        # control_zd = self.zd[control_id]
        control_zi = self.zi[control_id]
        # control_rho = self.rho[control_id]
        if self.celltype_num > 0:
            control_cell = self.celltype[control_id]
        else:
            control_cell = None
        if dose_value is None:
            treatment_perturb = np.tile(self.perturbation[treatment_id][0,:], (len(control_id),1))
            treatment_zd = self.generate_latent(treatment_perturb,control_cell)
        else:
            treatment_perturb = np.tile(self.perturbation[treatment_id][0,:-1], (len(control_id),1))
            dose_value = np.tile(dose_value, control_cell.shape[0]).reshape(-1,1)
            treatment_zd = self.generate_latent(np.concatenate([treatment_perturb,dose_value],axis=-1),control_cell)

        # treatment_zi = self.model.prior_zi(torch.tensor(control_cell))
        # mu_treatment_zi, logvar_treatment_zi = treatment_zi.chunk(2, dim=-1)
        # treatment_zi = self.model.reparameterize(mu_treatment_zi, logvar_treatment_zi).detach().numpy()

        if self.celltype is not None:
            control_celltype = self.celltype_label[control_id]
            treatment_celltype = control_celltype
        else:
            control_celltype, treatment_celltype = None, None
        
        if self.covariates_dim > 0:
            control_covariates = self.covariates[control_id]
            # rho_counterfactual = self.model.decoder(torch.cat([treatment_zd,torch.tensor(control_covariates)],dim=-1))
        else:
            control_covariates = None
            # rho_counterfactual = self.model.decoder(treatment_zd, torch.zeros(treatment_zd.shape[0]))
        
        if self.count_data:
            rho_counterfactual,_,_ = self.get_parameter_from_latent(latent=np.concatenate([treatment_zd,control_zi],axis=-1),
                                                            covariates=control_covariates)
        else:
            rho_counterfactual = self.get_parameter_from_latent(latent=np.concatenate([treatment_zd,control_zi],axis=-1),
                                                                covariates=control_covariates)
        return rho_counterfactual
        # if strategy=="minimal_change":
        #     W = groupwise_OT(control_zd, treatment_zd,
        #                      control_celltype, treatment_celltype, 
        #                      method=method, reg=reg, reg_m=reg_m)
        #     row_sums = W.sum(axis=1, keepdims=True)
        #     W = np.divide(W, row_sums, where=row_sums != 0)
        #     counterfactual_zd = W @ treatment_zd
        # elif strategy=="minimal_change_rank":
        #     W = groupwise_OT_rank(control_zd, treatment_zd,
        #                      control_celltype, treatment_celltype, 
        #                      method=method, reg=reg, reg_m=reg_m)
        #     row_sums = W.sum(axis=1, keepdims=True)
        #     W = np.divide(W, row_sums, where=row_sums != 0)
        #     counterfactual_zd = W @ treatment_zd    
        # elif strategy=="matching":
        #     W = groupwise_OT(control_zi, treatment_zi,
        #                      control_celltype, treatment_celltype, 
        #                      method=method, reg=reg, reg_m=reg_m)
        #     row_sums = W.sum(axis=1, keepdims=True)
        #     W = np.divide(W, row_sums, where=row_sums != 0)      
        #     counterfactual_zd = W @ treatment_zd  
        # elif strategy=="average":
        #     counterfactual_zd = add_effect(control_zd, treatment_zd, control_celltype, treatment_celltype)
        
        # counterfactual_z = np.concatenate([counterfactual_zd,control_zi],axis=-1)
        # if self.count_data:
        #     rho_counterfactual,dispersion_counterfactual,pi_counterfactual = self.get_parameter_from_latent(counterfactual_z, self.covariates)
        #     ite_pred = rho_counterfactual - control_rho
        # else:
        #     rho_counterfactual = self.get_parameter_from_latent(counterfactual_z, self.covariates)

    
    def joint_effect_estimate(self, control, treatment1, treatment2, dose1=None, dose2=None, #values=None,
                        strategy="minimal_change", method="sinkhorn", reg=0.1, reg_m=1.0):
        ite1 = self.effect_estimate(control, treatment1, dose=dose1, 
                        strategy=strategy, method=method, reg=reg, reg_m=reg_m)
        ite2 = self.effect_estimate(control, treatment2, dose=dose2,
                        strategy=strategy, method=method, reg=reg, reg_m=reg_m)
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
    
    # def plot_umap(self,z,label,legend_title=None,output_path=None,title=None,color_map='tab20'):
    #     '''
    #     Plot the UMAP visualization of the latent variables.
        
    #     Args:
    #         z: a numpy array representing the latent variables
    #         label: a numpy array representing the labels
    #         legend_title: a string representing the title of the legend
    #         output_path: a string representing the path (file name) to save the plot
    #         title: a string representing the title of the plot
    #         color_map: a string representing the color map
        
    #     Returns:
    #         The plot will show in window if `output_path` is None. Otherwise the plot will be saved at the specified path.
    #     '''
    #     umap_model = umap.UMAP(n_components=2)
    #     z_2d = umap_model.fit_transform(z)
    #     unique_labels = np.unique(label)
    #     colors = plt.cm.get_cmap(color_map, len(unique_labels))
    #     plt.figure(figsize=(10, 7))
    #     for i, lab in enumerate(unique_labels):
    #         indices = label == lab
    #         plt.scatter(z_2d[indices, 0], z_2d[indices, 1], 
    #                     label=lab, c=[colors(i)], s=0.1, alpha=0.8)
    #     plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.xlabel("UMAP Dimension 1")
    #     plt.ylabel("UMAP Dimension 2")
    #     if title is not None:
    #         plt.title(title)
    #     if output_path is not None:
    #         plt.savefig(output_path)
    #     else:
    #         plt.show()
    #     plt.clf()
