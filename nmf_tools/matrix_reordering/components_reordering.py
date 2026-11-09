import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import scipy.cluster.hierarchy as sch
from .matrix_reordering import MatrixReordering
from .hcluster import hierarchical_clustering


def order_components(W, *, by='primary', normalize=False, **kwargs):
    """
    Sorts elements in each column according to the specified method.

    Parameters
    ----------
    W : np.ndarray (n_components, n_records)
        Matrix to reorder.

    by : str
        Method to use for reordering.
        Can be one of:
            primary - sort by primary component
            template - sort by template matching
            mean_loading - sort by mean loading
            cluster - sort by cluster
            None - no reordering

    Returns
    -------
    component_order : MatrixReordering object
    """
    if normalize:
        W = W / W.sum(axis=0, keepdims=True)

    if isinstance(by, MatrixReordering):
        return by
    elif by == 'primary':
        return order_components_primary(W)
    elif by == 'template':
        return order_components_by_template(W, **kwargs)
    elif by == 'mean_loading':
        return order_components_by_mean_loading(W, **kwargs)
    elif by == 'cluster':
        return order_components_by_cluster(W, **kwargs)
    elif by == 'component_data':
        if 'component_data' not in kwargs:
            raise ValueError("component_data must be provided when by='component_data'")
        return order_components_by_component_data(W, **kwargs)
    elif by is None:
        return MatrixReordering()
    else:
        raise ValueError(f"Unknown ordering method: {by}")


def order_components_primary(W):
    '''
    Reorder the components of W by primary record.
    '''
    component_order = np.argsort(W, axis=0)[::-1, :].T
    return MatrixReordering(row_order=component_order)


def order_components_by_template(W, W_template):
    '''
    Reorder the components of W to match the components of W_template.

    Parameters
    ----------
    W : np.ndarray (n_components, n_records)
        Matrix to reorder.
    W_template : np.ndarray (n_components_ref, n_records)
        Matrix to match.

    Returns
    -------
    MatrixReordering
        ``row_order``: order of the components of W, matched one-to-one to the
        components of W_template by cosine similarity.
    '''
    cosine_similarity = 1 - cdist(W_template, W, 'cosine')
    _, order = linear_sum_assignment(-cosine_similarity)
    return MatrixReordering(row_order=order)


def order_components_by_mean_loading(W):
    """
    Order components by their mean loading across records, highest first.

    Parameters
    ----------
    W : np.ndarray (n_components, n_records)
        Matrix to reorder.

    Returns
    -------
    MatrixReordering
        ``row_order``: order of the components, shape (n_components,).
    """
    component_orders = np.argsort(W.mean(axis=1))[::-1].T
    return MatrixReordering(row_order=component_orders)


def order_components_by_cluster(W, cluster_labels=None, linkage_matrix=None,
                                cluster_threshold=0.7, criterion='distance',
                                component_orders_by_cluster=None, **kwargs):
    """
    Sorts elements in each column according to the cluster they belong to.
    If the cluster_labels are not provided, they will be computed using hierarchical clustering.

    Parameters
    ----------
    W : np.ndarray (n_components, n_records)
        Matrix to reorder.
    cluster_labels : np.ndarray (n_records,)
        Cluster labels for each record.
    component_orders_by_cluster : dict
        Dictionary with the order of components for each cluster.
        If None, the order will be determined by the mean loading in each cluster.
    linkage_matrix : np.ndarray (n_records-1, 4)
        Linkage matrix for hierarchical clustering.
    cluster_threshold : float
        Threshold for clustering.
    criterion : str
        Criterion for clustering.
    
    Returns
    -------
    component_orders : np.ndarray (n_components, n_records)
        New order of components for each record.
    """
    if cluster_labels is None:
        if linkage_matrix is None:
            linkage_matrix = hierarchical_clustering(W, **kwargs)
        cluster_labels = sch.fcluster(linkage_matrix, t=cluster_threshold, criterion=criterion)
    
    component_orders = np.zeros(W.shape, dtype=int)
    for i in np.unique(cluster_labels):
        idx = cluster_labels == i
        if component_orders_by_cluster is None:
            component_priority = np.argsort((W[:, idx] / W[:, idx].sum(axis=0)).mean(axis=1))[::-1]
        else:
            component_priority = component_orders_by_cluster[i]
        component_orders[:, idx] = component_priority[:, None]
    
    return MatrixReordering(row_order=component_orders.T)


def order_components_by_component_data(W, component_data, order_column='index'):
    """
    Sorts elements in each column according to the order specified in component_data.

    Parameters
    ----------
    W : np.ndarray (n_components, n_records)
        Matrix to reorder.
    component_data : pd.DataFrame
        DataFrame with component information.
    order_column : str
        Column in component_data to use for ordering.

    Returns
    -------
    component_orders : np.ndarray (n_components, n_records)
        New order of components for each record.
    """
    if order_column not in component_data.columns:
        raise ValueError(f"Column {order_column} not found in component_data")
    
    order = component_data[order_column].values
    if len(order) != W.shape[0]:
        raise ValueError("Length of order does not match number of components in W")
    
    return MatrixReordering(row_order=order)