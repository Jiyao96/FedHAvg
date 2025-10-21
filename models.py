import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
# CGL dependency
from torch_geometric.nn import TopKPooling, ChebConv
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_sparse import spspmm
from torch import Tensor
from typing import Optional, Tuple
import copy
from collections import OrderedDict
from torch_geometric.loader import DataLoader


geom_eps = 1e-12
clamp_margin = 1e-5

#---------- Poincaré ball maps at the origin ----------
def _exp0(v, c):
    """exp_0^c(v) with curvature c>0, applied elementwise to vector v (radial formula)."""
    if v.numel() == 0:
        return v
    sqrt_c = c ** 0.5
    # Frobenius norm over the whole tensor
    r = torch.norm(v.reshape(-1), p=2)
    #print("exp0 norm:", r)
    if r < geom_eps:
        return v.clone()
    coef = torch.tanh(sqrt_c * r) / (sqrt_c * r)
    # print("exp0 coef:", coef)
    return v * coef

def _log0(x, c):
    """log_0^c(x) inverse of exp at origin, stays inside the open ball."""
    if x.numel() == 0:
        return x
    sqrt_c = c ** 0.5
    r = torch.norm(x.reshape(-1), p=2)
    # print("log0 norm:", r)
    if r < geom_eps:
        return x.clone()    
    # clamp to strictly inside the ball
    max_r = (1.0 / sqrt_c) - clamp_margin
    if r >= max_r:
        print("log0 radius clampled")
        x = x * (max_r / (r + geom_eps))
        r = max_r
    coef = (1.0 / sqrt_c) * torch.atanh(sqrt_c * r) / (r + geom_eps)
    # print("log0 coef:", coef)
    return x * coef

def mobius_add(x, y, c):
    # inner products and norms along last dim
    xy  = torch.sum(x*y, dim=-1, keepdim=True)  
    nx2 = torch.sum(x*x, dim=-1, keepdim=True)
    ny2 = torch.sum(y*y, dim=-1, keepdim=True)

    num = (1 + 2 * c * xy + c * ny2) * x + (1 - c * nx2) * y
    den = 1 + 2 * c * xy + (c ** 2) * nx2 * ny2
    # avoid divide-by-zero and keep inside ball
    return num / (den + geom_eps)

def mobius_scalar_mul(t, x, c):
    nx = torch.linalg.norm(x, dim=-1, keepdim=True) 
    sqrtc = c ** 0.5
    ax = sqrtc * nx                                       
    ax_clamped = torch.clamp(ax, 0.0, 1.0 - geom_eps)
    coef = torch.tanh(t * torch.atanh(ax_clamped)) / (ax + geom_eps)  
    return coef * x

# scaled dot product attention described in https://arxiv.org/abs/1706.03762
# implemented in https://github.com/sooftware/attentions
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return context

class GNN_pooling_backbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, conv_type='GAT'):
        super().__init__()
        self.conv_type=conv_type
        # gnn convolution alternatives
        if conv_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=1, edge_dim=1)
            self.conv2 = GATConv(hidden_channels, out_channels, heads=1, edge_dim=1)
        elif conv_type == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif conv_type == "SAGE":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif conv_type == "Transformer":
            self.conv1 = TransformerConv(in_channels, hidden_channels,heads=1)
            self.conv2 = TransformerConv(hidden_channels, out_channels,heads=1)
        else:
            # linear layer 
            self.conv1 = nn.Linear(in_channels, hidden_channels)
            self.conv2 = nn.Linear(hidden_channels, out_channels)

        self.pool1 = TopKPooling(hidden_channels, ratio=0.5)
        self.pool2 = TopKPooling(out_channels, ratio=0.5)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        batch = data.batch
        # First convolutional layer
        if self.conv_type == 'GAT':
            x1 = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        elif self.conv_type == "GCN" or self.conv_type == "SAGE" or self.conv_type == "Transformer":
            x1 = self.conv1(x=x, edge_index=edge_index)
        else:
            x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1, edge_index, edge_weight, batch1, _, _ = self.pool1(x1, edge_index, edge_weight, batch)
        # Second convolutional layer
        if self.conv_type == 'GAT':
            x2 = self.conv2(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        elif self.conv_type == "GCN" or self.conv_type == "SAGE" or self.conv_type == "Transformer":
            x2 = self.conv2(x=x1, edge_index=edge_index)
        else:
            x2 = self.conv2(x1)
        x2 = F.relu(x2)
        x2, edge_index, edge_weight, batch2, _, _ = self.pool2(x2, edge_index, edge_weight, batch1)
        out = torch.cat([gmp(x1, batch1), gap(x1, batch1), gmp(x2, batch2), gap(x2, batch2)], dim=1)
        return out

class GNN_backbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, conv_type='GAT'):
        super().__init__()
        self.conv_type=conv_type
        # gnn convolution alternatives
        if conv_type == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=1, edge_dim=1)
            self.conv2 = GATConv(hidden_channels, out_channels, heads=1, edge_dim=1)
        elif conv_type == "GCN":
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif conv_type == "SAGE":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        elif conv_type == "Transformer":
            self.conv1 = TransformerConv(in_channels, hidden_channels,heads=1)
            self.conv2 = TransformerConv(hidden_channels, out_channels,heads=1)
        else:
            self.conv1 = nn.Linear(in_channels, hidden_channels)
            self.conv2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        batch = data.batch
        # First convolutional layer
        if self.conv_type == 'GAT':
            x1 = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_weight)
        elif self.conv_type == "GCN" or self.conv_type == "SAGE" or self.conv_type == "Transformer":
            x1 = self.conv1(x=x, edge_index=edge_index)
        else:
            x1 = self.conv1(x)
        x1 = F.relu(x1)
        # Second convolutional layer
        if self.conv_type == 'GAT':
            x2 = self.conv2(x=x1, edge_index=edge_index, edge_attr=edge_weight)
        elif self.conv_type == "GCN" or self.conv_type == "SAGE" or self.conv_type == "Transformer":
            x2 = self.conv2(x=x1, edge_index=edge_index)
        else:
            x2 = self.conv2(x1)
        x2 = F.relu(x2)
        out = torch.cat([gmp(x1, batch), gap(x1, batch), gmp(x2, batch), gap(x2, batch)], dim=1)
        return out


class CNN_backbone(nn.Module):
    def __init__(self, in_channels=1, out_channels=1024):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv3d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 48, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(48, 96, kernel_size=3, stride=1, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(14400, 512)
        self.fc2 = nn.Linear(512, out_channels)

    def forward(self, x):  
        x=x[:,None,:,:,:]
        # Apply convolutional layers with ReLU activation and pooling layers
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x))) 
        x = self.pool(F.relu(self.conv4(x)))  
        # Flatten the tensor
        x = x.view(x.size(0), -1) 
        # Fully connected layers with ReLU activation and Dropout
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)  
        return x
    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=2):
        super().__init__()
        ''' 
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.softmax(x, dim=1)
        return x 
    
class model_fn(nn.Module):
    def __init__(self, backbone, emb_dim):
        super().__init__()
        self.backbone = backbone
        self.mlp = MLP(in_dim=emb_dim)

    def forward(self, data):
        out = self.backbone(data)
        out = self.mlp(out)
        return out

class mlp_fn(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=2):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//4),
            # nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim//4),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//16),
            # nn.BatchNorm1d(hidden_dim),
            nn.LayerNorm(hidden_dim//16),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dim//16, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.softmax(x, dim=1)
        return x 

# class vae_fn(nn.Module):
#     """
#     VAE with an MLP encoder/decoder.
#     - Encoder mirrors your MLP down to a compact hidden, then predicts mu/logvar.
#     - Decoder is symmetric back to in_dim.
#     """
#     def __init__(self, in_dim, hidden_dim=1024, latent_dim=32, out_activation=None):
#         super().__init__()
#         # ----- Encoder (your blocks) -----
#         self.enc1 = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True)
#         )
#         self.enc2 = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 4),
#             nn.LayerNorm(hidden_dim // 4),
#             nn.ReLU(inplace=True)
#         )
#         self.enc3 = nn.Sequential(
#             nn.Linear(hidden_dim // 4, hidden_dim // 16),
#             nn.LayerNorm(hidden_dim // 16),
#             nn.ReLU(inplace=True)
#         )

#         # latent heads
#         h = hidden_dim // 16
#         self.mu = nn.Linear(h, latent_dim)
#         self.logvar = nn.Linear(h, latent_dim)

#         # ----- Decoder (symmetric) -----
#         self.dec1 = nn.Sequential(
#             nn.Linear(latent_dim, h),
#             nn.LayerNorm(h),
#             nn.ReLU(inplace=True)
#         )
#         self.dec2 = nn.Sequential(
#             nn.Linear(h, hidden_dim // 4),
#             nn.LayerNorm(hidden_dim // 4),
#             nn.ReLU(inplace=True)
#         )
#         self.dec3 = nn.Sequential(
#             nn.Linear(hidden_dim // 4, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(inplace=True)
#         )
#         self.dec_out = nn.Linear(hidden_dim, in_dim)

#         # How to squash reconstruction: 'sigmoid' for BCE on [0,1] data; None for MSE
#         self.out_activation = out_activation

#         self.classifier = nn.Linear(latent_dim*2, 2)

#     @staticmethod
#     def reparameterize(mu, logvar):
#         # z = mu + std * eps
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def encode(self, x):
#         h = self.enc3(self.enc2(self.enc1(x)))
#         mu = self.mu(h)
#         logvar = self.logvar(h)
#         return mu, logvar

#     def decode(self, z):
#         x_hat = self.dec_out(self.dec3(self.dec2(self.dec1(z))))
#         if self.out_activation == "sigmoid":
#             x_hat = torch.sigmoid(x_hat)
#         elif self.out_activation == "tanh":
#             x_hat = torch.tanh(x_hat)
#         return x_hat

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         x_hat = self.decode(z)

#         latent = torch.cat([mu, logvar], dim=1)
#         logits = self.classifier(latent)

#         return logits, x_hat, mu, logvar


# def vae_loss(x_hat, x, mu, logvar, recon="mse", beta=1.0):
#     """
#     VAE loss = reconstruction + beta * KL
#     recon: 'mse' (default) or 'bce'
#         - Use 'bce' if your inputs are in [0,1] and decoder uses sigmoid.
#         - Use 'mse' for real-valued inputs without bounded range.
#     reduction: 'mean' or 'sum' across batch.
#     """
#     if recon == "mse":
#         recon_term = F.mse_loss(x_hat, x, reduction="none").sum(dim=1)
#     elif recon == "bce":
#         recon_term = F.binary_cross_entropy_with_logits(x_hat, x, reduction="none").sum(dim=1)
#     else:
#         raise ValueError("recon must be 'mse' or 'bce'")

#     # KL per sample: D_KL(q(z|x) || N(0, I))
#     kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

#     loss = recon_term + beta * kl

#     return {
#         "loss": loss.mean(),
#         "recon": recon_term.mean(),
#         "kl": kl.mean()
#     }


# class VAEServer(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.global_model = copy.deepcopy(model).cpu()
    
#     @torch.no_grad()
#     def broadcast(self):
#         # (SHARED) send global weights
#         return {k: v.clone() for k, v in self.global_model.state_dict().items()}

#     @torch.no_grad()
#     def aggregate(self, client_payloads):
#         # client_payload = {"state": state_dict, "n": num_samples}
#         total = sum(p["n"] for p in client_payloads)
#         new_state = OrderedDict()
#         # weighted average of parameters
#         for k in client_payloads[0]["state"].keys():
#             new_state[k] = sum((p["n"] / total) * p["state"][k] for p in client_payloads)
#         self.global_model.load_state_dict(new_state)

# class VAEClient:
#     def __init__(self, model, dataset, device="cpu", lr=1e-4, batch_size=16):
#         super().__init__()
#         self.num_roi = 116
#         self.model = model.to(device)
#         self.device = device
#         self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001, momentum=0.9, nesterov=True)

#     @torch.no_grad()
#     def load_global(self, global_state):
#         # (RECEIVE) initialize local model from global weights
#         self.model.load_state_dict(global_state, strict=True)

#     def local_train(self, epochs=1):
#         self.model.train()
#         n_seen = 0
#         for _ in range(epochs):
#             for data in self.loader:
#                 # data = data.to(self.device)
#                 # n_seen += data.x.size(0)//self.num_roi
#                 x, y = data
#                 x = x.to(self.device)
#                 y = y.long().view(-1).to(self.device)
#                 n_seen += x.size(0)   
#                 logits, x_hat, mu, logvar = self.model(x)
#                 ae_loss = vae_loss(x_hat, x, mu, logvar)["loss"]
#                 label_loss = F.cross_entropy(logits, y)
#                 loss = label_loss + ae_loss
#                 self.opt.zero_grad()
#                 loss.backward()
#                 self.opt.step()

#         # (SHARE) return only weights (or deltas) + sample count; data never leaves
#         return {"state": {k: v.cpu().clone() for k, v in self.model.state_dict().items()},"n": n_seen}
    
#     @torch.no_grad()
#     def evaluate(self):
#         self.model.eval()
#         correct = total = 0
#         with torch.no_grad():
#             for data in self.loader:
#                 # data = data.to(self.device)
#                 # n_seen += data.x.size(0)//self.num_roi
#                 x, y = data
#                 x = x.to(self.device)
#                 y = y.long().view(-1).to(self.device)
#                 logits, _, _, _ = self.model(x)
#                 pred = logits.argmax(dim=1)
#                 correct += (pred == y).sum().item()
#                 total += y.size(0)
#         print("local train acc: ", correct/total)
#         return


class Server(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.global_model = copy.deepcopy(model).cpu()
        self.global_mask = None

    def _stack_param(self, client_payloads, key):
        return torch.stack([p["state"][key] for p in client_payloads], dim=0)
    
    @torch.no_grad()
    def broadcast(self):
        # (SHARED) send global weights
        return {k: v.clone() for k, v in self.global_model.state_dict().items()}
    
    # @torch.no_grad()
    # def broadcast_mask(self):
    #     # (SHARED) send global mask
    #     return self.global_mask
    
    # @torch.no_grad()
    # def aggregate_mask(self, client_masks, min_ratio=0.2):
    #     # aggregate clinet masks
    #     min_votes = len(client_masks)*min_ratio
    #     votes = torch.stack(client_masks, dim=0).sum(dim=0)
    #     self.global_mask = votes > min_votes
    #     print("Total channels on the global mask: ", self.global_mask.sum(dim=0).item())
    #     return 

    @torch.no_grad()
    def aggregate(self, client_payloads):
        # client_payload = {"state": state_dict, "n": num_samples}
        new_state = OrderedDict()
        total_clients = len(client_payloads)
        # weighted average of parameters
        for k in client_payloads[0]["state"].keys():
            new_state[k] = sum((1 / total_clients) * p["state"][k] for p in client_payloads)
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def weighted_aggregate(self, client_payloads):
        # client_payload = {"state": state_dict, "n": num_samples}
        total = sum(p["n"] for p in client_payloads)
        new_state = OrderedDict()
        # weighted average of parameters
        for k in client_payloads[0]["state"].keys():
            new_state[k] = sum((p["n"] / total) * p["state"][k] for p in client_payloads)
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def aggregate_coord_median(self, client_payloads):
        new_state = OrderedDict()
        for k in client_payloads[0]["state"].keys():
            X = self._stack_param(client_payloads, k)               
            new_state[k] = torch.median(X, dim=0).values
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def aggregate_trimmed_mean(self, client_payloads):
        tfrac = 0.25
        new_state = OrderedDict()
        n = len(client_payloads)
        t = int(n * tfrac)
        for k in client_payloads[0]["state"].keys():
            X = self._stack_param(client_payloads, k)                
            vals, _ = torch.sort(X, dim=0)
            start, end = t, max(t, n - t)
            # If trimming removes all, fall back to simple mean
            if end <= start:
                new_state[k] = X.mean(dim=0)
            else:
                new_state[k] = vals[start:end].mean(dim=0)
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def aggregate_par(self, client_payloads, gamma=0.001, c=1):
        new_state = OrderedDict()
        for k in client_payloads[0]["state"].keys():
            X = self._stack_param(client_payloads, k)   
            # Project each client delta
            Y = []
            for i in range(X.shape[0]):
                v = X[i] * gamma
                y = _exp0(v, c)
                Y.append(y)
            Y = torch.stack(Y, dim=0)
            y_bar = Y.mean(dim=0)
            # Retract and unscale
            c = torch.tensor(c) if isinstance(c, (int, float)) else c
            v_hat = _log0(y_bar, c) / (gamma + geom_eps)
            # print(X.mean(dim=0)-v_hat)
            new_state[k] = v_hat
        self.global_model.load_state_dict(new_state)
    
    @torch.no_grad()
    def aggregate_barycenter(self, client_payloads, gamma=0.001, c=1):
        """
        Gyrovector (Einstein) barycenter:  ⊕_i (α_i ⊗ Y_i)
        Left-fold using Möbius ops. For equal weights, pass weights=None.
        """
        new_state = OrderedDict()
        for k in client_payloads[0]["state"].keys():
            X = self._stack_param(client_payloads, k)
            # Project each client delta
            Y = []
            for i in range(X.shape[0]):
                v = X[i] * gamma
                y = _exp0(v, c)
                Y.append(y)
            Y = torch.stack(Y, dim=0)

            # Gyrovector (Möbius) barycenter: m = ⊕_i (α_i ⊗ Y_i), equal weights α_i=1/n
            n = Y.shape[0]
            Yf = Y.view(n, -1) 
            m = torch.zeros_like(Yf[0])
            alpha = 1.0 / n
            for i in range(n):
                m = mobius_add(m, mobius_scalar_mul(alpha, Yf[i], c=c), c=c)
            # Retract and unscale
            y_bar = m.view_as(Y[0])
            c = torch.tensor(c) if isinstance(c, (int, float)) else c
            v_hat = _log0(y_bar, c)/(gamma + geom_eps)
            new_state[k] =  v_hat
        self.global_model.load_state_dict(new_state)

class Client:
    def __init__(self, model, dataset, device="cpu", lr=1e-4, batch_size=16):
        super().__init__()
        self.num_roi = 116
        self.model = model.to(device)
        self.mask = None
        self.device = device
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001, momentum=0.9, nesterov=True)

    @torch.no_grad()
    def load_global(self, global_state):
        # (RECEIVE) initialize local model from global weights
        self.model.load_state_dict(global_state, strict=True)

    # @torch.no_grad()
    # def load_mask(self, global_mask):
    #     # (RECEIVE) initialize local model from global weights
    #     self.mask = global_mask

    def local_train(self, epochs=1):
        self.model.train()
        n_seen = 0
        for _ in range(epochs):
            for data in self.loader:
                # data = data.to(self.device)
                # n_seen += data.x.size(0)//self.num_roi
                x, y = data
                # apply mask
                # x = x * self.mask

                x = x.to(self.device)
                y = y.long().view(-1).to(self.device)
                n_seen += x.size(0)   
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        # (SHARE) return only weights (or deltas) + sample count; data never leaves
        return {"state": {k: v.cpu().clone() for k, v in self.model.state_dict().items()},"n": n_seen}
    
    # @torch.no_grad()
    # def get_vote(self, k = 1024):
    #     x = torch.stack(self.dataset.x)
    #     y = torch.stack(self.dataset.y)

    #     x0 = x - x.mean(dim=0, keepdim=True)
    #     y0 = y - y.mean()

    #     sx = x0.std(dim=0, unbiased=True)
    #     sy = y0.std(unbiased=True)

    #     # Avoid division by zero
    #     sx = torch.where(sx == 0, torch.tensor(float("inf")), sx)

    #     cov = (x0.t() @ y0) / (x.shape[0] - 1)
    #     corr = cov / (sx * sy)

    #     # Top-k by absolute value
    #     abs_corr = corr.abs()
    #     top_vals, top_idx = torch.topk(abs_corr, k, largest=True, sorted=True)
    #     # Recover signed correlations in the same order
    #     # top_corr = corr[top_idx]
    #     mask = torch.zeros((x.shape[1],))
    #     mask[top_idx] = 1

    #     return mask

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data in self.loader:
                # data = data.to(self.device)
                # n_seen += data.x.size(0)//self.num_roi
                x, y = data
                # apply mask
                # x = x * self.mask

                x = x.to(self.device)
                y = y.long().view(-1).to(self.device)
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print("local train acc: ", correct/total)
        return 


class GeomServer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.global_model = copy.deepcopy(model).cpu()
        self.global_mask = None

    def _stack_param(self, client_payloads, key):
        return torch.stack([p["state"][key] for p in client_payloads], dim=0)
    
    @torch.no_grad()
    def broadcast(self):
        # (SHARED) send global weights
        return {k: v.clone() for k, v in self.global_model.state_dict().items()}
    
    # @torch.no_grad()
    # def broadcast_mask(self):
    #     # (SHARED) send global mask
    #     return self.global_mask
    
    # @torch.no_grad()
    # def aggregate_mask(self, client_masks, min_ratio=0.2):
    #     # aggregate clinet masks
    #     min_votes = len(client_masks)*min_ratio
    #     votes = torch.stack(client_masks, dim=0).sum(dim=0)
    #     self.global_mask = votes > min_votes
    #     print("Total channels on the global mask: ", self.global_mask.sum(dim=0).item())
    #     return 

    @torch.no_grad()
    def aggregate(self, client_payloads):
        # client_payload = {"state": state_dict, "n": num_samples}
        new_state = OrderedDict()
        total_clients = len(client_payloads)
        # weighted average of parameters
        for k in client_payloads[0]["state"].keys():
            new_state[k] = self.global_model.state_dict()[k].cpu() + sum((1 / total_clients) * p["state"][k] for p in client_payloads)
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def weighted_aggregate(self, client_payloads):
        # client_payload = {"state": state_dict, "n": num_samples}
        total = sum(p["n"] for p in client_payloads)
        new_state = OrderedDict()
        # weighted average of parameters
        for k in client_payloads[0]["state"].keys():
            new_state[k] = self.global_model.state_dict()[k].cpu() + sum((p["n"] / total) * p["state"][k] for p in client_payloads)
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def aggregate_coord_median(self, client_payloads):
        new_state = OrderedDict()
        for k in client_payloads[0]["state"].keys():
            X = self._stack_param(client_payloads, k)               
            new_state[k] = self.global_model.state_dict()[k].cpu() + torch.median(X, dim=0).values
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def aggregate_trimmed_mean(self, client_payloads):
        tfrac = 0.25
        new_state = OrderedDict()
        n = len(client_payloads)
        t = int(n * tfrac)
        for k in client_payloads[0]["state"].keys():
            X = self._stack_param(client_payloads, k)                
            vals, _ = torch.sort(X, dim=0)
            start, end = t, max(t, n - t)
            # If trimming removes all, fall back to simple mean
            if end <= start:
                new_state[k] = self.global_model.state_dict()[k].cpu() + X.mean(dim=0)
            else:
                new_state[k] = self.global_model.state_dict()[k].cpu() + vals[start:end].mean(dim=0)
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def aggregate_geom_median(self, client_payloads, T=10):
        new_state = OrderedDict()
        for k in client_payloads[0]["state"].keys():
            if not torch.is_floating_point(client_payloads[0]["state"][k]):
                new_state[k] = client_payloads[0]["state"][k]
                continue
            U = self._stack_param(client_payloads, k)    # [n, ...]
            x = U.mean(dim=0)  # init
            for _ in range(T):
                diffs = U - x
                d = torch.sqrt(torch.clamp((diffs ** 2).reshape(diffs.size(0), -1).sum(dim=1), min=0.0)) + geom_eps
                alpha = (1/len(client_payloads) / d)
                alpha = alpha / (alpha.sum() + geom_eps)
                a = alpha.view([-1] + [1] * (U.ndim - 1))
                x = (U * a).sum(dim=0)
            new_state[k] = x
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def aggregate_par(self, client_payloads, gamma=0.001, c=1):
        new_state = OrderedDict()
        for k in client_payloads[0]["state"].keys():
            X = self._stack_param(client_payloads, k)   
            # Project each client delta
            Y = []
            for i in range(X.shape[0]):
                v = X[i] * gamma
                y = _exp0(v, c)
                Y.append(y)
            Y = torch.stack(Y, dim=0)
            y_bar = Y.mean(dim=0)
            # Retract and unscale
            v_hat = _log0(y_bar, c) / (gamma + geom_eps)
            # print(X.mean(dim=0)-v_hat)
            new_state[k] = self.global_model.state_dict()[k].cpu() + v_hat
        self.global_model.load_state_dict(new_state)

    @torch.no_grad()
    def aggregate_barycenter(self, client_payloads, gamma=0.001, c=1):
        """
        Gyrovector (Einstein) barycenter:  ⊕_i (α_i ⊗ Y_i)
        Left-fold using Möbius ops. For equal weights, pass weights=None.
        """
        new_state = OrderedDict()
        for k in client_payloads[0]["state"].keys():
            X = self._stack_param(client_payloads, k)
            # Project each client delta
            Y = []
            for i in range(X.shape[0]):
                v = X[i] * gamma
                y = _exp0(v, c)
                Y.append(y)
            Y = torch.stack(Y, dim=0)

            # Gyrovector (Möbius) barycenter: m = ⊕_i (α_i ⊗ Y_i), equal weights α_i=1/n
            n = Y.shape[0]
            Yf = Y.view(n, -1) 
            m = torch.zeros_like(Yf[0])
            alpha = 1.0 / n
            for i in range(n):
                m = mobius_add(m, mobius_scalar_mul(alpha, Yf[i], c=c), c=c)
            # Retract and unscale
            y_bar = m.view_as(Y[0])
            c = torch.tensor(c) if isinstance(c, (int, float)) else c
            v_hat = _log0(y_bar, c)/(gamma + geom_eps)
            new_state[k] = self.global_model.state_dict()[k].cpu() + v_hat
        self.global_model.load_state_dict(new_state)
    

class GeomClient(nn.Module):
    def __init__(self, model, dataset, device="cpu", lr=1e-4, batch_size=16):
        super().__init__()
        self.num_roi = 116
        self.model = model.to(device)
        self.global_log = None
        self.mask = None
        self.device = device
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001, momentum=0.9, nesterov=True)

    @torch.no_grad()
    def load_global(self, global_state):
        # (RECEIVE) initialize local model from global weights
        self.model.load_state_dict(global_state, strict=True)
        self.global_log = copy.deepcopy(global_state)

    # @torch.no_grad()
    # def load_mask(self, global_mask):
    #     # (RECEIVE) initialize local model from global weights
    #     self.mask = global_mask

    def local_train(self, epochs=1):
        self.model.train()
        n_seen = 0
        for _ in range(epochs):
            for data in self.loader:
                # data = data.to(self.device)
                # n_seen += data.x.size(0)//self.num_roi
                x, y = data
                # apply mask
                # x = x * self.mask

                x = x.to(self.device)
                y = y.long().view(-1).to(self.device)
                n_seen += x.size(0)   
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

        # (SHARE) return only weights (or deltas) + sample count; data never leaves
        return {"state": {k: v.cpu().clone()-self.global_log[k].cpu().clone() for k, v in self.model.state_dict().items()},"n": n_seen}
    
    # @torch.no_grad()
    # def get_vote(self, k = 1024):
    #     x = torch.stack(self.dataset.x)
    #     y = torch.stack(self.dataset.y)

    #     x0 = x - x.mean(dim=0, keepdim=True)
    #     y0 = y - y.mean()

    #     sx = x0.std(dim=0, unbiased=True)
    #     sy = y0.std(unbiased=True)

    #     # Avoid division by zero
    #     sx = torch.where(sx == 0, torch.tensor(float("inf")), sx)

    #     cov = (x0.t() @ y0) / (x.shape[0] - 1)
    #     corr = cov / (sx * sy)

    #     # Top-k by absolute value
    #     abs_corr = corr.abs()
    #     top_vals, top_idx = torch.topk(abs_corr, k, largest=True, sorted=True)
    #     # Recover signed correlations in the same order
    #     # top_corr = corr[top_idx]
    #     mask = torch.zeros((x.shape[1],))
    #     mask[top_idx] = 1

    #     return mask

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data in self.loader:
                # data = data.to(self.device)
                # n_seen += data.x.size(0)//self.num_roi
                x, y = data
                # apply mask
                # x = x * self.mask

                x = x.to(self.device)
                y = y.long().view(-1).to(self.device)
                pred = self.model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print("local train acc: ", correct/total)
        return 



    

    


