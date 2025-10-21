import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from imports import FC_Dataset
from models import Server, Client, mlp_fn
import argparse
from sklearn.metrics import f1_score, roc_auc_score


torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Create a parser object
parser = argparse.ArgumentParser(description="Argument parser.")

# Add arguments
parser.add_argument('--test_id', type=int, default=5, help="Data parcellation used for testing")
parser.add_argument('--num_round', type=int, default=100, help="Number of update rounds")
parser.add_argument('--num_epoch', type=int, default=1, help="Number of epochs")
parser.add_argument('--dim', type=int, default=1024, help="Number of embedding dimension")
parser.add_argument('--use_img', type=bool, default=False, help="if using the image input")
parser.add_argument('--conform_type', type=str, default="quantile", help="output type")
parser.add_argument('--conv_type', type=str, default="GAT", help="graph convolution layer")
parser.add_argument('--weighted_mse', type=bool, default=False, help="use mse loss weighted by confidence")

# Parse the arguments
args = parser.parse_args()
print(args)

# parameters
batch_size=8
num_epoch=args.num_epoch
conform_type=args.conform_type
conv_type=args.conv_type
num_roi=116
num_edge=num_roi*(num_roi-1)//2

# subject parcel for cross-validation
sites_list = torch.load("../preprocess_code/ABIDE_sites")
sites = ['NYU', 'UM', 'USM', 'UCLA'] 
num_sites = len(sites)

temp_ids = np.arange(5)
train_parcel = np.delete(temp_ids, args.test_id-1)
test_parcel = [args.test_id-1]

def collect_subjects_from_folds(folds_subjects,sites,folds):
    out = []
    for site in sites:
        per_folds = folds_subjects[site]
        for f in folds:
            out.extend(per_folds[f])
    return out

# define client model and sites data
site_datasets=[]
for s in sites:
    train_sub = collect_subjects_from_folds(sites_list, [s], train_parcel)    
    print("Site "+ s +":", len(train_sub))
    train_dataset = FC_Dataset('/gpfs/gibbs/pi/duncan/jw2695/code/preprocess_code/ABIDE_fc/', train_sub)
    site_datasets.append(train_dataset)

# parameters
test_sub = collect_subjects_from_folds(sites_list, sites, test_parcel)
print("Global testing: ", len(test_sub))

# test loader
test_dataset = FC_Dataset('/gpfs/gibbs/pi/duncan/jw2695/code/preprocess_code/ABIDE_fc/', test_sub)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

clients = [Client(mlp_fn(in_dim=num_edge), site_datasets[i]) for i in range(num_sites)]
server = Server(mlp_fn(in_dim=num_edge))

def update(round):
    global_state = server.broadcast()                 # (SHARED) global → clients

    payloads = []
    for c in clients:
        c.load_global(global_state)                   # receive global weights
        payloads.append(c.local_train(args.num_epoch))  # train locally, return weights + n
        c.evaluate()

    # server.aggregate_trimmed_mean(payloads)                        # (SHARED) aggregate to new global
    # server.aggregate_par(payloads, gamma=100, c=2) 
    server.aggregate_barycenter(payloads, gamma=1, c=2) 
    # quick eval on global model (optional)
    acc = evaluate(server.global_model.to(device), test_loader, device)
    print(f"Global test acc: {acc:.3f}")

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in loader:
            # data = data.to(self.device)
            # n_seen += data.x.size(0)//self.num_roi
            x, y = data
            x = x.to(device)
            y = y.long().view(-1).to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += 1
    return correct / total

def metric_evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    y_true_all = []
    y_pred_all = []
    y_score_all = []  # probs/logits for AUC
    with torch.no_grad():
        for data in loader:
            x, y = data
            x = x.to(device)
            y = y.long().view(-1).to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            y_score = logits[:,1]  # shape [B]

            correct += (preds == y).sum().item()
            total += y.numel()

            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())
            y_score_all.append(y_score.cpu().numpy())

    acc = correct / max(1, total)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    y_score = np.concatenate(y_score_all)
    # F1 (binary => 'binary', else 'macro'); avoid divide-by-zero warnings
    n_classes = np.unique(y_true).size
    f1 = f1_score(
        y_true, y_pred,
        average='binary',
        zero_division=0
    )

    auc = roc_auc_score(y_true, y_score)

    return acc, f1, auc

test_site_loader=[]
for s in sites:
    persite_test_sub = collect_subjects_from_folds(sites_list, [s], test_parcel)    
    print("Site "+ s +":", len(persite_test_sub))
    persite_test_dataset = FC_Dataset('/gpfs/gibbs/pi/duncan/jw2695/code/preprocess_code/ABIDE_fc/', persite_test_sub)
    test_site_loader.append(DataLoader(persite_test_dataset,batch_size=1,shuffle=False))

def per_site_evaluate():
    for i in range(len(sites)):
        acc, f1, auc = metric_evaluate(server.global_model.to(device), test_site_loader[i], device)
        print(f"Test site {sites[i]}, ACC {acc:.3g} F1 {f1:.3g} AUC {auc:.3g}")    
    acc, f1, auc = metric_evaluate(server.global_model.to(device), test_loader, device)
    print(f"Global test ACC {acc:.3g} F1 {f1:.3g} AUC {auc:.3g}")

for i in range(1, args.num_round+1):
    print("Round: ", i)
    update(i)
    if i%20 == 0:
        torch.save(server.global_model.state_dict(), "logs/server_round"+str(i)+"_div"+str(args.test_id))
        for j in range(4):
            torch.save(clients[j].model.state_dict(), "logs/client"+str(j)+"_round"+str(i)+"_div"+str(args.test_id))

per_site_evaluate()
