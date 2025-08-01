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

def groupwise_OT_rank(control_z, treatment_z, c_x, c_y, method="sinkhorn", reg=0.1, reg_m=1.0):
    """Computes a global optimal transport matching matrix (quantile matching of zd only), ensuring that matches occur only within groups.

    Args:
        control_z (np.ndarray): Source matrix of shape (N, D).
        treatment_z (np.ndarray): Target matrix of shape (M, D).
        c_x (np.ndarray): Group labels of shape (N,), indicating which group each row in X belongs to.
        c_y (np.ndarray): Group labels of shape (M,), indicating which group each row in Y belongs to.
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
    N, M = control_z.shape[0], treatment_z.shape[0]
    matching_matrix = lil_matrix((N, M))  # Sparse matrix for efficiency
    
    if c_x is None:
        c_x = np.ones(N)
        c_y = np.ones(M)
    
    if not np.array_equal(np.unique(c_x), np.unique(c_y)):
        raise ValueError(f"Unique cell types do not match!")

    unique_groups = np.unique(c_x)  # Get unique group labels

    for group in unique_groups:
        # Get indices of X and Y that belong to the same group
        control_zidx = np.where(c_x == group)[0]
        treatment_zidx = np.where(c_y == group)[0]  # Assuming Y has the same groups as X

        if len(control_zidx) == 0 or len(treatment_zidx) == 0:
            continue  # Skip if no matching group in Y

        control_group, treatment_group = control_z[control_zidx], treatment_z[treatment_zidx]

        # Uniform weight distributions
        a = ot.unif(control_group.shape[0])
        b = ot.unif(treatment_group.shape[0])

        # Compute cost matrix (Euclidean distance)
        rank_control = condition_quantile(control_group)
        rank_treatment = condition_quantile(treatment_group)
        C = ot.dist(rank_control,rank_treatment,metric='euclidean')
        # C_normalized = C / np.median(C)
        C_normalized = C 

        # Compute optimal transport plan
        if method == "emd":
            G = ot.emd(a, b, C_normalized)
        elif method == "sinkhorn":
            G = ot.sinkhorn(a, b, C_normalized, reg=reg)
        elif method == "unbalanced_sinkhorn":
            G = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C_normalized, reg=reg, reg_m=reg_m)
        else:
            raise ValueError("No this method!")

        # Assign the computed transport plan to the corresponding block in the global matrix
        matching_matrix[np.ix_(control_zidx, treatment_zidx)] = G

    return matching_matrix.toarray()  # Convert sparse matrix to dense format before returning


def groupwise_OT(control_z, treatment_z, c_x, c_y, method="sinkhorn", reg=0.1, reg_m=1.0):
    """Computes a global optimal transport matching matrix (quantile matching of zd only), ensuring that matches occur only within groups.

    Args:
        control_z (np.ndarray): Source matrix of shape (N, D).
        treatment_z (np.ndarray): Target matrix of shape (M, D).
        c_x (np.ndarray): Group labels of shape (N,), indicating which group each row in X belongs to.
        c_y (np.ndarray): Group labels of shape (M,), indicating which group each row in Y belongs to.
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
    N, M = control_z.shape[0], treatment_z.shape[0]
    matching_matrix = lil_matrix((N, M))  # Sparse matrix for efficiency
    
    if c_x is None:
        c_x = np.ones(N)
        c_y = np.ones(M)
    
    if not np.array_equal(np.unique(c_x), np.unique(c_y)):
        raise ValueError(f"Unique cell types do not match!")

    unique_groups = np.unique(c_x)  # Get unique group labels

    for group in unique_groups:
        # Get indices of X and Y that belong to the same group
        control_zidx = np.where(c_x == group)[0]
        treatment_zidx = np.where(c_y == group)[0]  # Assuming Y has the same groups as X

        if len(control_zidx) == 0 or len(treatment_zidx) == 0:
            continue  # Skip if no matching group in Y

        control_group, treatment_group = control_z[control_zidx], treatment_z[treatment_zidx]

        # Uniform weight distributions
        a = ot.unif(control_group.shape[0])
        b = ot.unif(treatment_group.shape[0])

        # Compute cost matrix (Euclidean distance)
        C = ot.dist(control_group,treatment_group,metric='euclidean')
        C_normalized = C / np.median(C)
        
        # Compute optimal transport plan
        if method == "emd":
            G = ot.emd(a, b, C_normalized)
        elif method == "sinkhorn":
            G = ot.sinkhorn(a, b, C_normalized, reg=reg)
        elif method == "unbalanced_sinkhorn":
            G = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C_normalized, reg=reg, reg_m=reg_m)
        else:
            raise ValueError("No this method!")

        # Assign the computed transport plan to the corresponding block in the global matrix
        matching_matrix[np.ix_(control_zidx, treatment_zidx)] = G

    return matching_matrix.toarray()  # Convert sparse matrix to dense format before returning


def groupwise_OT_latent(control_zd, control_zi, treatment_zd, treatment_zi, c_x, c_y, method="sinkhorn", reg=0.1, reg_m=1.0):
    """Computes a global optimal transport matching matrix (quantile matching of zd and value matching of zip), ensuring that matches occur only within groups.

    Args:
        control_zd (np.ndarray): Source matrix of shape (N, D_d).
        control_zi (np.ndarray): Source matrix of shape (N, D_i).
        treatment_zd (np.ndarray): Target matrix of shape (M, D_d).
        treatment_zi (np.ndarray): Target matrix of shape (M, D_i).
        c_x (np.ndarray): Group labels of shape (N,), indicating which group each row in X belongs to.
        c_y (np.ndarray): Group labels of shape (M,), indicating which group each row in Y belongs to.
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
    matching_matrix = lil_matrix((N, M))  # Sparse matrix for efficiency
    
    if c_x is None:
        c_x = np.ones(N)
        c_y = np.ones(M)
    
    if not np.array_equal(np.unique(c_x), np.unique(c_y)):
        raise ValueError(f"Unique cell types do not match!")

    unique_groups = np.unique(c_x)  # Get unique group labels

    for group in unique_groups:
        # Get indices of X and Y that belong to the same group
        control_zidx = np.where(c_x == group)[0]
        treatment_zidx = np.where(c_y == group)[0]  # Assuming Y has the same groups as X

        if len(control_zidx) == 0 or len(treatment_zidx) == 0:
            continue  # Skip if no matching group in Y

        control_group_d, treatment_group_d, control_group_i, treatment_group_i = control_zd[control_zidx], treatment_zd[treatment_zidx], control_zi[control_zidx], treatment_zi[treatment_zidx]

        # Uniform weight distributions
        a = ot.unif(control_group_d.shape[0]) #ot.unif(control_group_d.shape[0])
        b = ot.unif(treatment_group_d.shape[0]) #ot.unif(treatment_group_d.shape[0])

        # Compute cost matrix (Euclidean distance)
        rank_control = condition_quantile(control_group_d)
        rank_treatment = condition_quantile(treatment_group_d)
        # scaler = StandardScaler()
        # C = ot.dist(rank_control,rank_treatment,metric='euclidean') + \
        #     ot.dist(scaler.fit_transform(control_group_i),scaler.fit_transform(treatment_group_i),metric='euclidean')
        # C = ot.dist(scaler.fit_transform(control_group_i),scaler.fit_transform(treatment_group_i),metric='euclidean')
        # C = ot.dist(control_group_i,treatment_group_i,metric='euclidean')
        # C = ot.dist(control_group_d,treatment_group_d,metric='euclidean')
        C = ot.dist(rank_control,rank_treatment,metric='euclidean')
        C_normalized = C / np.median(C)

        # Compute optimal transport plan
        if method == "emd":
            G = ot.emd(a, b, C)
        elif method == "sinkhorn":
            G = ot.sinkhorn(a, b, C, reg=reg)
        elif method == "unbalanced_sinkhorn":
            G = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, C, reg=reg, reg_m=reg_m)
        else:
            raise ValueError("No this method!")
        # G = G / G.sum(axis=1, keepdims=True)

        # Assign the computed transport plan to the corresponding block in the global matrix
        matching_matrix[np.ix_(control_zidx, treatment_zidx)] = G

    return matching_matrix.toarray()  # Convert sparse matrix to dense format before returning

def add_effect(control_zd, treatment_zd, c_x, c_y):
    N, M, D = control_zd.shape[0], treatment_zd.shape[0], control_zd.shape[1]
    counterfactual_zd = np.zeros((N,D))
    if c_x is None:
        c_x = np.ones(N)
        c_y = np.ones(M)
    unique_groups = np.unique(c_x) 
    for group in unique_groups:
        # Get indices of X and Y that belong to the same group
        control_zidx = np.where(c_x == group)[0]
        treatment_zidx = np.where(c_y == group)[0]  # Assuming Y has the same groups as X

        if len(control_zidx) == 0 or len(treatment_zidx) == 0:
            continue  # Skip if no matching group in Y

        control_group_d, treatment_group_d = control_zd[control_zidx], treatment_zd[treatment_zidx]
        delta = np.mean(treatment_group_d, axis=0) - np.mean(control_group_d, axis=0)
        counterfactual_zd[control_zidx,:] = control_group_d + delta
    return counterfactual_zd


def to_dense_array(x):
    '''
    Transform a potential sparse array to numpy array
    '''
    if issparse(x):
        return x.toarray()
    elif isinstance(x, np.ndarray):
        return x.copy()
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

# from scipy.interpolate import interp1d

# def groupwise_quantile_matching(control_zd, treatment_zd, c_x, c_y):
#     """Computes a global quantile matching transformation, ensuring matches occur only within groups.

#     Args:
#         control_zd (np.ndarray): Source matrix of shape (N, D_d).
#         treatment_zd (np.ndarray): Target matrix of shape (M, D_d).
#         c_x (np.ndarray): Group labels of shape (N,), indicating which group each row in X belongs to.
#         c_y (np.ndarray): Group labels of shape (M,), indicating which group each row in Y belongs to.

#     Returns:
#         np.ndarray: Adjusted treatment_zd' with the same conditional distribution as control_zd.
#     """
#     N, M = control_zd.shape[0], treatment_zd.shape[0]
#     adjusted_treatment = np.zeros_like(treatment_zd)

#     if c_x is None:
#         c_x = np.ones(N)
#         c_y = np.ones(M)
    
#     if not np.array_equal(np.unique(c_x), np.unique(c_y)):
#         raise ValueError(f"Unique group labels do not match between control and treatment sets!")

#     unique_groups = np.unique(c_x)  # Get unique group labels

#     for group in unique_groups:
#         # Get indices of X and Y that belong to the same group
#         control_zidx = np.where(c_x == group)[0]
#         treatment_zidx = np.where(c_y == group)[0]  # Assuming Y has the same groups as X

#         if len(control_zidx) == 0 or len(treatment_zidx) == 0:
#             continue  # Skip if no matching group in Y

#         control_group_d = control_zd[control_zidx]
#         treatment_group_d = treatment_zd[treatment_zidx]

#         # Compute quantile ranks
#         control_quantiles = condition_quantile(control_group_d)
#         treatment_quantiles = condition_quantile(treatment_group_d)

#         # Apply quantile matching per feature
#         for d in range(control_zd.shape[1]):
#             # Build interpolation function based on control distribution
#             sorted_control = np.sort(control_group_d[:, d])
#             sorted_control_quantiles = np.sort(control_quantiles[:, d])  # Ensure monotonicity
#             interpolator = interp1d(sorted_control_quantiles, sorted_control, bounds_error=False, fill_value="extrapolate")

#             # Transform treatment data to match control distribution
#             adjusted_treatment[treatment_zidx, d] = interpolator(treatment_quantiles[:, d])

#     return adjusted_treatment