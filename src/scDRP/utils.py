import numpy as np 
import pandas as pd 
import ot
from scipy.sparse import lil_matrix
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler

def condition_quantile(X,y=None):
    '''
    Calculate the quantile of each column in X conditioned on y.
    
    Args:
        X: a numpy array
        y: a numpy array
    
    Returns: 
        a numpy array representing the quantile of each column in X conditioned on y
    '''
    df = pd.DataFrame(X)
    if y is not None:
        quantiles_df = df.groupby(y).rank(pct=True,method='average')
    else:
        quantiles_df = df.rank(pct=True,method='average')
    return quantiles_df.to_numpy()

# def groupwise_OT_rank(control_z, treatment_z, control_label, treatment_label, method="sinkhorn", reg=0.1, reg_m=1.0):
#     """Computes a global optimal transport matching matrix (quantile matching of zd only), ensuring that matches occur only within groups.

#     Args:
#         control_z (np.ndarray): Source matrix of shape (N, D).
#         treatment_z (np.ndarray): Target matrix of shape (M, D).
#         control_label (np.ndarray): Group labels of shape (N,), indicating which group each row in X belongs to.
#         treatment_label (np.ndarray): Group labels of shape (M,), indicating which group each row in Y belongs to.
#         method (str, optional): Optimal transport method. Options:
        
#             - "emd": Exact Optimal Transport
#             - "sinkhorn": Sinkhorn Regularized OT
#             - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
#             Defaults to "emd".
#         reg (float, optional): Entropic regularization parameter for Sinkhorn. Default is 0.1. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
#         reg_m (float, optional): Marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".

#     Returns:
#         np.ndarray: A global matching matrix (N, M), where matches are restricted within the same group.
#     """
#     N, M = control_z.shape[0], treatment_z.shape[0]
#     coupling_matrix = lil_matrix((N, M))  # Sparse matrix for efficiency
    
#     if control_label is None:
#         control_label = np.ones(N)
#         treatment_label = np.ones(M)
    
#     # if not np.array_equal(np.unique(control_label), np.unique(treatment_label)):
#     #     raise ValueError(f"Unique cell types do not match!")

#     unique_groups = np.unique(control_label)  # Get unique group labels

#     for group in unique_groups:
#         # Get indices of X and Y that belong to the same group
#         control_z_idx = np.where(control_label == group)[0]
#         treatment_z_idx = np.where(treatment_label == group)[0]  # Assuming Y has the same groups as X

#         if len(control_z_idx) == 0 or len(treatment_z_idx) == 0:
#             continue  # Skip if no matching group in Y

#         control_group, treatment_group = control_z[control_z_idx], treatment_z[treatment_z_idx]

#         # Uniform weight distributions
#         a = ot.unif(control_group.shape[0])
#         b = ot.unif(treatment_group.shape[0])

#         # Compute cost matrix (Euclidean distance)
#         rank_control = condition_quantile(control_group)
#         rank_treatment = condition_quantile(treatment_group)
#         C = ot.dist(rank_control,rank_treatment,metric='euclidean')
#         # C_normalized = C / np.median(C)
#         C_normalized = C 

#         # Compute optimal transport plan
#         if method == "emd":
#             G = ot.emd(a, b, C_normalized)
#         elif method == "sinkhorn":
#             G = ot.bregman.sinkhorn_log(a, b, C_normalized, reg=reg)
#         elif method == "unbalanced_sinkhorn":
#             G = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C_normalized, reg=reg, reg_m=reg_m)
#         else:
#             raise ValueError("No this method!")

#         # Assign the computed transport plan to the corresponding block in the global matrix
#         coupling_matrix[np.ix_(control_z_idx, treatment_z_idx)] = G
#         if np.any(np.isnan(G)):
#             raise ValueError("NaN in the transport plan!")

#     return coupling_matrix.toarray()  # Convert sparse matrix to dense format before returning


# def groupwise_OT(control_z, treatment_z, control_label, treatment_label, method="sinkhorn", reg=0.1, reg_m=1.0, eps=1e-8):
#     """Computes a global optimal transport matching matrix (quantile matching of zd only), ensuring that matches occur only within groups.

#     Args:
#         control_z (np.ndarray): Source matrix of shape (N, D).
#         treatment_z (np.ndarray): Target matrix of shape (M, D).
#         control_label (np.ndarray): Group labels of shape (N,), indicating which group each row in X belongs to.
#         treatment_label (np.ndarray): Group labels of shape (M,), indicating which group each row in Y belongs to.
#         method (str, optional): Optimal transport method. Options:
        
#             - "emd": Exact Optimal Transport
#             - "sinkhorn": Sinkhorn Regularized OT
#             - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
#             Defaults to "emd".
#         reg (float, optional): Entropic regularization parameter for Sinkhorn. Default is 0.1. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
#         reg_m (float, optional): Marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".

#     Returns:
#         np.ndarray: A global matching matrix (N, M), where matches are restricted within the same group.
#     """
#     N, M = control_z.shape[0], treatment_z.shape[0]
#     coupling_matrix = lil_matrix((N, M))  # Sparse matrix for efficiency
#     counterfactual_z = np.zeros_like(control_z)
    
#     if control_label is None:
#         control_label = np.ones(N)
#         treatment_label = np.ones(M)
    
#     # if not np.array_equal(np.unique(control_label), np.unique(treatment_label)):
#     #     raise ValueError(f"Unique cell types do not match!")

#     unique_groups = np.unique(control_label)  # Get unique group labels

#     for group in unique_groups:
#         # Get indices of X and Y that belong to the same group
#         control_z_idx = np.where(control_label == group)[0]
#         treatment_z_idx = np.where(treatment_label == group)[0]  # Assuming Y has the same groups as X

#         if len(control_z_idx) == 0 or len(treatment_z_idx) == 0:
#             continue  # Skip if no matching group in Y

#         control_group, treatment_group = control_z[control_z_idx], treatment_z[treatment_z_idx]

#         # Uniform weight distributions
#         a = ot.unif(control_group.shape[0])
#         b = ot.unif(treatment_group.shape[0])

#         # Compute cost matrix (Euclidean distance)
#         C = ot.dist(control_group,treatment_group,metric='euclidean') ** 2
#         C_normalized = C / (np.max(C) + eps)
        
#         # Compute optimal transport plan
#         if method == "emd":
#             G = ot.emd(a, b, C_normalized)
#         elif method == "sinkhorn":
#             G = ot.bregman.sinkhorn_log(a, b, C_normalized, reg=reg)
#         elif method == "unbalanced_sinkhorn":
#             G = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C_normalized, reg=reg, reg_m=reg_m)
#         else:
#             raise ValueError("No this method!")
        
#         # row_sums = G.sum(axis=1, keepdims=True)
#         # G = np.divide(G, row_sums, where=row_sums != 0)
#         G = G / (G.sum(axis=1, keepdims=True) + eps)
#         if np.any(np.isnan(G)):
#             raise ValueError("NaN in the transport plan!")
#         counterfactuals = G @ treatment_group
#         counterfactual_z[control_z_idx] = counterfactuals

#         # import umap
#         # import matplotlib.pyplot as plt
#         # umap_model = umap.UMAP(random_state=42)
#         # Z = np.vstack([control_group, counterfactuals, treatment_group])
#         # Z_umap = umap_model.fit_transform(Z)
#         # labels = (["X"] * control_group.shape[0]) + (["X_tilde"] * counterfactuals.shape[0]) + (["Y"] * treatment_group.shape[0])
#         # plt.figure(figsize=(8, 6))
#         # for lab, color in zip(["X", "X_tilde", "Y"], ["red", "blue", "green"]):
#         #     idx = [i for i, l in enumerate(labels) if l == lab]
#         #     plt.scatter(Z_umap[idx, 0], Z_umap[idx, 1], s=10, c=color, label=lab, alpha=0.7)
#         # plt.legend()
#         # plt.title("UMAP visualization of OT alignment")
#         # plt.savefig(f"/home/jianles/scPerturb/new_simulation_codes/figures/umap_ot_alignment_{group}.png", dpi=300)
#         # plt.close()

#         # Assign the computed transport plan to the corresponding block in the global matrix
#         coupling_matrix[np.ix_(control_z_idx, treatment_z_idx)] = G
        
#     return counterfactual_z,coupling_matrix.toarray()  # Convert sparse matrix to dense format before returning


def conditional_OT_latent(control_zd, treatment_zd, control_zu, treatment_zu, control_label, treatment_label, 
                        alpha=1, beta=1, projection_strategy="full", value="raw",
                        method="emd", reg=0.1, reg_m=1.0, eps=1e-8):
    """Computes a global optimal transport matching matrix (quantile matching of zd and value matching of zip), ensuring that matches occur only within groups.

    Args:
        control_zd (np.ndarray): Source matrix of shape (N, D_d).
        control_zu (np.ndarray): Source matrix of shape (N, D_i).
        treatment_zd (np.ndarray): Target matrix of shape (M, D_d).
        treatment_zu (np.ndarray): Target matrix of shape (M, D_i).
        control_label (np.ndarray): Group labels of shape (N,), indicating which group each row in X belongs to.
        treatment_label (np.ndarray): Group labels of shape (M,), indicating which group each row in Y belongs to.
        alpha (float, optional): Weight for the distance in zd space. Default is 1.
        beta (float, optional): Weight for the distance in zu space. Default is 1
        projection_strategy (str, optional): Strategy for constructing counterfactual latent representations. Options:
            - "full": Use both zd and zu for counterfactual construction.
            - "zd_only": Use only zd for counterfactual construction, keeping zu unchanged.
            Defaults to "full".
        method (str, optional): Optimal transport method. Options:
            - "emd": Exact Optimal Transport
            - "sinkhorn": Sinkhorn Regularized OT
            - "unbalanced_sinkhorn": Unbalanced Sinkhorn Regularized OT
            Defaults to "emd".
        reg (float, optional): Entropic regularization parameter for Sinkhorn. Default is 0.1. Only useful when specifying method as "sinkhorn" or "unbalanced_sinkhorn".
        reg_m (float, optional): Marginal relaxation parameter (higher allows more mass deviation). Default is 1.0. Only useful when specifying method as "unbalanced_sinkhorn".

    Returns:
        np.ndarray: A global matching matrix (N, M), where matches are restricted within the same group.
    """
    N, M = control_zd.shape[0], treatment_zd.shape[0]
    coupling_matrix = lil_matrix((N, M))  # Sparse matrix for efficiency
    
    if projection_strategy == "full":
        counterfactual_z = np.zeros((N, control_zd.shape[1] + control_zu.shape[1]))
    elif projection_strategy == "zd_only":
        counterfactual_zd = np.zeros_like(control_zd)
    else:
        raise ValueError(f"Unknown projection strategy: {projection_strategy}")
    
    if control_label is None:
        control_label = np.ones(N)
        treatment_label = np.ones(M)

    unique_groups = np.unique(control_label)  # Get unique group labels

    for group in unique_groups:
        # Get indices of X and Y that belong to the same group
        control_z_idx = np.where(control_label == group)[0]
        treatment_z_idx = np.where(treatment_label == group)[0]  # Assuming Y has the same groups as X

        if len(control_z_idx) == 0 or len(treatment_z_idx) == 0:
            continue  # Skip if no matching group in Y

        control_group_d, treatment_group_d, control_group_u, treatment_group_u = \
            control_zd[control_z_idx], treatment_zd[treatment_z_idx], \
                control_zu[control_z_idx], treatment_zu[treatment_z_idx]

        # Uniform weight distributions
        a = ot.unif(control_group_d.shape[0]) #ot.unif(control_group_d.shape[0])
        b = ot.unif(treatment_group_d.shape[0]) #ot.unif(treatment_group_d.shape[0])

        # Compute cost matrix (Euclidean distance)
        # rank_control = condition_quantile(control_group_d)
        # rank_treatment = condition_quantile(treatment_group_d)
        # scaler = StandardScaler()
        # C = ot.dist(rank_control,rank_treatment,metric='euclidean') + \
        #     ot.dist(scaler.fit_transform(control_group_u),scaler.fit_transform(treatment_group_u),metric='euclidean')
        # C = ot.dist(scaler.fit_transform(control_group_u),scaler.fit_transform(treatment_group_u),metric='euclidean')
        # C = ot.dist(control_group_u,treatment_group_u,metric='euclidean')
        # C = ot.dist(control_group_d,treatment_group_d,metric='euclidean')
        if value == "raw":
            C = alpha * ot.dist(control_group_d, treatment_group_d, metric='sqeuclidean') + \
                beta * ot.dist(control_group_u,treatment_group_u, metric='sqeuclidean')
        elif value == "quantile":
            rank_control = condition_quantile(control_group_d)
            rank_treatment = condition_quantile(treatment_group_d)
            scale = np.concatenate([rank_control, control_group_u], axis=1).max()
            control_group_u = control_group_u / scale
            treatment_group_u = treatment_group_u / scale
            C = alpha * ot.dist(rank_control,rank_treatment,metric='sqeuclidean') + \
                beta * ot.dist(control_group_u,treatment_group_u, metric='sqeuclidean')
        else:
            raise ValueError(f"Unknown value type: {value}")
        C = C / np.max(C)

        # Compute optimal transport plan
        if method == "emd":
            G = ot.emd(a, b, C)
        elif method == "sinkhorn":
            G = ot.bregman.sinkhorn_log(a, b, C, reg=reg)
        elif method == "unbalanced_sinkhorn":
            G = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C, reg=reg, reg_m=reg_m)
        else:
            raise ValueError("No this OT method!")
        G = G / (G.sum(axis=1, keepdims=True) + eps)

        # Assign the computed transport plan to the corresponding block in the global matrix
        coupling_matrix[np.ix_(control_z_idx, treatment_z_idx)] = G

        if projection_strategy == "full":
            counterfactuals_z = G @ np.hstack([treatment_group_d, treatment_group_u])
            counterfactual_z[control_z_idx] = counterfactuals_z
        elif projection_strategy == "zd_only":
            counterfactuals_zd = G @ treatment_group_d
            counterfactual_zd[control_z_idx] = counterfactuals_zd
    
    if projection_strategy == "full":
        return counterfactual_z, coupling_matrix.toarray()  # Convert sparse matrix to dense format before returning
    elif projection_strategy == "zd_only":
        counterfactual_z = np.hstack([counterfactual_zd, control_zu])
        return counterfactual_z, coupling_matrix.toarray()  # Convert sparse matrix to dense
    # return coupling_matrix.toarray()  # Convert sparse matrix to dense format before returning

def add_effect(control_zd, treatment_zd, control_label, treatment_label):
    """
    Computes counterfactual latent representations by adding the average treatment effect within each group.
    Args:
        control_zd (np.ndarray): Source matrix of shape (N, D_d).
        treatment_zd (np.ndarray): Target matrix of shape (M, D_d).
        control_label (np.ndarray): Group labels of shape (N,), indicating which group each row in X belongs to.
        treatment_label (np.ndarray): Group labels of shape (M,), indicating which group each row in Y belongs to.
    """
    N, M, D = control_zd.shape[0], treatment_zd.shape[0], control_zd.shape[1]
    counterfactual_zd = np.zeros((N,D))
    if control_label is None:
        control_label = np.ones(N)
        treatment_label = np.ones(M)
    unique_groups = np.unique(control_label) 
    for group in unique_groups:
        # Get indices of X and Y that belong to the same group
        control_z_idx = np.where(control_label == group)[0]
        treatment_z_idx = np.where(treatment_label == group)[0]  # Assuming Y has the same groups as X

        if len(control_z_idx) == 0 or len(treatment_z_idx) == 0:
            continue  # Skip if no matching group in Y

        control_group_d, treatment_group_d = control_zd[control_z_idx], treatment_zd[treatment_z_idx]
        delta = np.mean(treatment_group_d, axis=0) - np.mean(control_group_d, axis=0)
        counterfactual_zd[control_z_idx,:] = control_group_d + delta
    return counterfactual_zd


def to_dense_array(x):
    '''
    Transform a potential sparse array to numpy array
    Args:
        x: input array, can be sparse or numpy array
    Returns:
        a numpy array
    '''
    if issparse(x):
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x.copy()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
