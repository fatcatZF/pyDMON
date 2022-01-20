from __future__ import division
from __future__ import print_function

import time
import datetime
import os
import pickle
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from models_dmon import *

from sklearn.metrics import normalized_mutual_info_score as nmi

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--out', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--n-clusters', type=int, default=10,
                    help='Number of output units.')
parser.add_argument('--gcn-type', type=str, default="dmon",
                    help="type of GCN")
parser.add_argument('--gcn-activation', type=str, default="selu",
                    help="activation function for gcn")
parser.add_argument("--collapse-regularization", type=float, default=1.,
                    help="collapse regularization.")
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs/dmon',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    

#Save model and meta-data
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    model_file = os.path.join(save_folder, "model.pt")
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))

else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")
    


adj, features, labels, label_indices = load_npz("data/cora.npz")
adj_tensor = torch.tensor(adj.todense()).unsqueeze(0).float()
features_tensor = torch.tensor(features.todense()).unsqueeze(0).float()

model = GCN_DMoN(features_tensor.size(-1), args.hidden, args.out ,args.n_clusters, 
                      args.gcn_type, args.gcn_activation, args.collapse_regularization,
                      args.dropout)


if args.load_folder:
    model_file = os.path.join(args.load_folder, 'model.pt')
    model.load_state_dict(torch.load(model_file))
    


if args.cuda:
    model = model.to("cuda")
    features_tensor = features_tensor.to("cuda")
    #features_valid.cuda()
    #features_test.cuda()
    adj_tensor = adj_tensor.to("cuda")
    #adj_valid.cuda()
    #adj_test.cuda()



optimizer = optim.Adam(list(model.parameters()),
                       lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)




"""
if args.cuda:
    model = model.to("cuda")
    features_tensor = features_tensor.to("cuda")
    #features_valid.cuda()
    #features_test.cuda()
    adj_tensor = adj_tensor.to("cuda")
    #adj_valid.cuda()
    #adj_test.cuda()
"""    



def train(epoch, best_val_loss):
    t = time.time()
    loss_train = []
    sp_loss_train = []
    co_loss_train = []
    
    model.train()
    optimizer.zero_grad()
    
    #print(next(model.parameters()).device)
    #print(adj_tensor.device)
    #print(features_tensor.device)    

    assignments, spectral_loss, collapse_loss = model(adj_tensor, features_tensor)
    loss = spectral_loss+collapse_loss
    loss.backward()
    optimizer.step()
    loss_train.append(loss.item())
    sp_loss_train.append(spectral_loss.item())
    co_loss_train.append(collapse_loss.item())
    
    
    loss_val = []
    sp_loss_val = []
    co_loss_val = []
    
    model.eval()
    with torch.no_grad():
        assignments, spectral_loss, collapse_loss = model(adj_tensor, features_tensor)
        loss = spectral_loss+collapse_loss
        loss_val.append(loss.item())
        sp_loss_val.append(spectral_loss.item())
        co_loss_val.append(collapse_loss.item())
        
    print("Epoch: {:04d}".format(epoch+1),
          "loss_train: {:.10f}".format(loss_train[0]),
          "sp_loss_train: {:.10f}".format(sp_loss_train[0]),
          "co_loss_train: {:.10f}".format(co_loss_train[0]),
          "loss_val: {:.10f}".format(loss_val[0]),
          "sp_loss_val: {:.10f}".format(sp_loss_val[0]),
          "co_loss_val: {:.10f}".format(co_loss_val[0]))
    
    
    if args.save_folder and np.mean(loss_val) < best_val_loss:
        torch.save(model.state_dict(), model_file)
        print('Best model so far, saving...')
        print("Epoch: {:04d}".format(epoch+1),
              "loss_train: {:.10f}".format(loss_train[0]),
              "sp_loss_train: {:.10f}".format(sp_loss_train[0]),
              "co_loss_train: {:.10f}".format(co_loss_train[0]),
              "loss_val: {:.10f}".format(loss_val[0]),
              "sp_loss_val: {:.10f}".format(sp_loss_val[0]),
              "co_loss_val: {:.10f}".format(co_loss_val[0]),
              file=log)
        log.flush()
        
    return np.mean(loss_val)




def test():
    
    print("Test model")
    
    model_file = os.path.join(save_folder, 'model.pt')
    model.load_state_dict(torch.load(model_file))
    
    loss_test = []
    sp_loss_test = []
    co_loss_test = []
    
    model.eval()
    with torch.no_grad():
        assignments, spectral_loss, collapse_loss = model(adj_tensor, features_tensor)
        loss = spectral_loss+collapse_loss
        loss_test.append(loss.item())
        sp_loss_test.append(spectral_loss.item())
        co_loss_test.append(collapse_loss.item())
        
        clusters = assignments.cpu().argmax(-1).squeeze().numpy()
        nmi_value = nmi(clusters[label_indices], labels)
        
    print("Epoch: {:04d}".format(epoch+1),
          "loss_test: {:.10f}".format(loss_test[0]),
          "sp_loss_test: {:.10f}".format(sp_loss_test[0]),
          "co_loss_test: {:.10f}".format(co_loss_test[0]),
          "nmi_value: {:.10f}".format(nmi_value)
          )
    





#Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch+1))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
    
test()
    
    
        
        
    




    

    










