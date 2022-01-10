import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from utils import *



class GCNLayer_Dmon(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        """
        args:
            n_in: input features
            n_out: number of output dimensions
        """
        super(GCNLayer_Dmon, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.weight = Parameter(torch.FloatTensor(n_in, n_out))
        self.skip_weight = Parameter(torch.FloatTensor(n_in, n_out))
        if bias:
            self.bias = Parameter(torch.FloatTensor(n_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.skip_weight.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, A, X=None):
        """
        args:
            X: node features; shape: [batch_size, num_nodes, n_in]
             if X is None, an identity matrix will be used
            A: normalized adjacency matrix (not including self-loops)
               shape: [batch_size, num_nodes, num_nodes]
        """
        num_nodes = A.size(1)
        if X is None:
            X = torch.eye(num_nodes).unsqueeze(0)
            X = X.expand(A.size(0), X.size(1), X.size(2)) #expand to batch size
        
        agg = torch.matmul(torch.matmul(A, X), self.weight) #aggregation of neighbours
        skip = torch.matmul(X, self.skip_weight)
        if self.bias is not None:
          return skip+agg+self.bias
        else:
            return skip+agg
            
        
        



class GCNLayer_Kipf(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        """
        args:
            n_in: input features
            n_out: number of output dimensions
        """
        super(GCNLayer_Kipf, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.weight = Parameter(torch.FloatTensor(n_in, n_out))
        if bias:
            self.bias = Parameter(torch.FloatTensor(n_out))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, A, X=None):
        """
        args:
            X: node features; shape: [batch_size, num_nodes, n_in]
             if X is None, an identity matrix will be used
            A: normalized adjacency matrix 
               shape: [batch_size, num_nodes, num_nodes]           
        """
        num_nodes = A.size(1)
        if X is None:
            X = torch.eye(num_nodes).unsqueeze(0)
            X = X.expand(A.size(0), X.size(1), X.size(2)) #expand to batch size
        
        outputs = torch.matmul(A, X)
        if self.bias is not None:
            outputs = torch.matmul(outputs, self.weight)+self.bias
        else:
            outputs = torch.matmul(outputs, self.weight)
        return outputs
    
    
    



class DMoN(nn.Module):
    """Implementation of Deep Modularity Network (DMoN) Layer.
    Deep Modularity Network (DMoN) Layer implementation
    DMoN optimizes modularity clustering objective in a
    fully unsupervised mode
    args:
        n_clusters: Number of clusters in the model
        collapse_regularization: Collapse regularization weight
        dropout_rate: Dropout rate. The dropout in applied to the
            intermediate representations before softmax
        do_unpooling: Parameter controlling whether to perform 
            unpooling of the features with respect to their soft clusters.
            If true, shape of the input is preserved.
    """
    def __init__(self, n_in ,n_clusters, collapse_regularization=0.1,
                 dropout_rate =0, do_unpooling = False):
        super(DMoN, self).__init__()
        self.n_in = n_in
        self.n_clusters = n_clusters
        self.collapse_regularization = collapse_regularization
        self.dropout_rate = dropout_rate
        self.do_unpooling = do_unpooling
        self.fc = nn.Linear(n_in, n_clusters)
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, A, X=None):
        """
        Performs DMoN clustering according to input features and input graph
        args:
            X: node features, shape:[batch_size, num_nodes, n_in]
            A: adjacency matrix, shape:[batch_size, num_nodes, num_nodes]
        """
        if X is None:
            X = torch.eye(num_nodes).unsqueeze(0)
            X = X.expand(A.size(0), X.size(1), X.size(2)) #expand to batch size
        
        batch_size = A.size(0)
        num_nodes = A.size(1) #get number of nodes
        assignments = torch.softmax(self.dropout(self.fc(X)), dim=-1) #shape:[batch_size,num_nodes,n_clusters]
        cluster_sizes = assignments.sum(1) #shape: [batch_size, num_clusters]
        assignments_pooling = assignments / cluster_sizes #shape:[batch_size,num_nodes,n_clusters]
        
        degrees = A.sum(-1) #shape: [batch_size, num_nodes]
        degrees = degrees.unsqueeze(-1) #shape:[batch_size, num_nodes, 1]
        edge_weights = degrees.sum(-1).sum(-1) #shape: [batch_size]
        
        #graph_pooled = torch.matmul(A, assignments).transpose(-1,-2) #[batch_size, n_clusters, num_nodes]
        graph_pooled = torch.matmul(assignments.transpose(-1,-2),A)
        graph_pooled = torch.matmul(graph_pooled, assignments) #[batch_size, n_clusters, n_clusters]
        
        #Compute the rank-1 normalizer matrix S^T*d*d^T*S
        normalizer_left = torch.matmul(assignments.transpose(-1,-2), degrees)
        #shape: [batch_size, n_cluster, 1]
        normalizer_right = torch.matmul(degrees.transpose(-1,-2), assignments)
        #shape: [batch_size, 1, n_cluster]
        
        normalizer = torch.matmul(normalizer_left, normalizer_right)/2/edge_weights
        #shape:[batch_size, n_cluster, n_cluster]
        
        spectral_loss = -torch.diagonal(graph_pooled-normalizer, dim1=-2, dim2=-1).sum()/2/edge_weights/batch_size
        
        collapse_loss = (torch.norm(cluster_sizes)/num_nodes*torch.sqrt(torch.FloatTensor([self.n_clusters]))-1)/batch_size
        
        
        return assignments, spectral_loss, collapse_loss
    
    




class GCN_DMoN(nn.Module):
    def __init__(self, n_in, n_hid, n_clusters, gcn_type="dmon",
                 activation="relu", collapse_regularization=0.1,
                 dropout_rate=0):
        super(GCN_DMoN, self).__init__()
        if gcn_type.lower() == "dmon":
            self.gcn = GCNLayer_Dmon(n_in, n_hid)
        else:
            self.gcn = GCNLayer_Kipf(n_in, n_hid)
            
        self.dmon = DMoN(n_hid, n_clusters, collapse_regularization, dropout_rate)
        if activation.lower() == "relu":
            self.activation = F.relu
        else: self.activation = F.selu
        
    def forward(self, A, X):
        """
        args:
            A: adjacency matrix; shape:[batch_size, num_nodes]
            X:node features; shape:[batch_size, num_nodes, n_in]
        """
        if isinstance(self.gcn, GCNLayer_Dmon):
            A_normalized = normalize_graph(A, add_self_loops=False)
        else:
            A_normalized = normalize_graph(A, add_self_loops=True)
            
        hidden = self.activation(self.gcn(A_normalized, X))
        #shape: [batch_size, num_nodes, n_hid]
        assignments, spectral_loss, collapse_loss = self.dmon(A, hidden)
        
        return assignments, spectral_loss, collapse_loss
        
        
        
            
        
            
        
        
    




























            
        
    
        
