import numpy as np
from numpy.random import shuffle
import torch
import scipy.sparse as sp
import scipy


def normalize_graph(graph, add_self_loops=True):
    """
    args:
      graph: adjacency matrix; [batch_size, num_nodes, num_nodes]
    """
    num_nodes = graph.size(1)
    if add_self_loops:
        I = torch.eye(num_nodes).unsqueeze(0)
        I.expand(graph.size(0),I.size(1),I.size(2))
        graph += I
    degree = graph.sum(-1) #shape:[batch_size, num_nodes]
    degree = 1./torch.sqrt(degree) #shape:[batch_size,num_nodes]
    degree[degree==torch.inf]=0 #convert infs to 0s
    degree = torch.diag_embed(degree) #shape:[batch_size,num_nodes, num_nodes]
    return degree@graph@degree


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot




def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)  






def load_data(path="./data/cora/", dataset="cora",
             label_rate = 0.02):
    """Load citation network dataset"""
    print("Loading {} dataset...".format(dataset))
    #load indices, features, and labels; shape: [num_nodes, 1+num_features+1]
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:,1:-1], dtype=np.float32)
    features = normalize(features)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    idx = np.array(idx_features_labels[:,0], dtype=np.int32)
    idx_map = {j:i for i,j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), 
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj+adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj)
    
    indices = np.arange(features.shape[0])
    shuffle(indices)
    train_up = int(indices.shape[0]*0.6) 
    valid_up = int(indices.shape[0]*0.8)
    train_indices = indices[:train_up]
    valid_indices = indices[train_up: valid_up]
    test_indices = indices[valid_up:]
    features_train = features[train_indices]
    features_valid = features[valid_indices]
    features_test = features[test_indices]
    labels_train = labels[train_indices]
    labels_valid = labels[valid_indices]
    labels_test = labels[test_indices]
    labels_relation_train = np.matmul(labels_train, labels_train.T)
    labels_relation_valid = np.matmul(labels_valid, labels_valid.T)
    labels_test = np.argmax(labels_test, axis=-1)
    #set 0 to -1 for training
    labels_relation_train[np.where(labels_relation_train==0)]=-1
    labels_relation_valid[np.where(labels_relation_valid==0)]=-1
    #create mask
    mask_train = np.random.choice([1,0], size=labels_relation_train.shape, p=[label_rate, 1-label_rate])
    mask_valid = np.random.choice([1,0], size=labels_relation_valid.shape, p=[label_rate, 1-label_rate])
    #mask relation labels
    labels_train_masked = mask_train*labels_relation_train
    labels_valid_masked = mask_valid*labels_relation_valid
    #split matrices for training, validation and test
    adj_train = adj[train_indices,:][:,train_indices]
    adj_valid = adj[valid_indices,:][:,valid_indices]
    adj_test = adj[test_indices,:][:,test_indices]
    
    features_train = torch.from_numpy(np.array(features_train.todense()))
    features_valid = torch.from_numpy(np.array(features_valid.todense()))
    features_test = torch.from_numpy(np.array(features_test.todense()))
    
    labels_train_masked = torch.from_numpy(labels_train_masked)
    labels_valid_masked = torch.from_numpy(labels_valid_masked)
    adj_train = sparse_mx_to_torch_sparse_tensor(adj_train)
    adj_valid = sparse_mx_to_torch_sparse_tensor(adj_valid)
    adj_test = sparse_mx_to_torch_sparse_tensor(adj_test)
    
    return (features_train, features_valid, features_test, adj_train, adj_valid, adj_test,
           labels_train_masked, labels_valid_masked, labels_test)



def load_npz(
    filename
):
  """
  Copied from https://github.com/fatcatZF/google-research/blob/master/graph_embedding/dmon/train.py
  Loads an attributed graph with sparse features from a specified Numpy file.
  Args:
    filename: A valid file name of a numpy file containing the input data.
  Returns:
    A tuple (graph, features, labels, label_indices) with the sparse adjacency
    matrix of a graph, sparse feature matrix, dense label array, and dense label
    index array (indices of nodes that have the labels in the label array).
  """
  with np.load(open(filename, 'rb'), allow_pickle=True) as loader:
    loader = dict(loader)
    adjacency = scipy.sparse.csr_matrix(
        (loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
        shape=loader['adj_shape'])

    features = scipy.sparse.csr_matrix(
        (loader['feature_data'], loader['feature_indices'],
         loader['feature_indptr']),
        shape=loader['feature_shape'])

    label_indices = loader['label_indices']
    labels = loader['labels']
  assert adjacency.shape[0] == features.shape[
      0], 'Adjacency and feature size must be equal!'
  assert labels.shape[0] == label_indices.shape[
      0], 'Labels and label_indices size must be equal!'
  return adjacency, features, labels, label_indices















