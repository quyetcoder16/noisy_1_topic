import random
import os
from tqdm import tqdm
import numpy as np
import glob
import pandas as pd
import torch
import scipy.sparse as sp
import torch.nn as nn
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from diffuse import IC

random.seed(1)
torch.manual_seed(1)

class GNN_skip_small(nn.Module):
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_hidden_3, dropout):
        super(GNN_skip_small, self).__init__()
        self.fc1 = nn.Linear(2 * n_feat, n_hidden_1)
        self.fc2 = nn.Linear(2 * n_hidden_1, n_hidden_2)
        self.fc4 = nn.Linear(n_feat + n_hidden_1 + n_hidden_2, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(n_hidden_1)
        self.bn2 = nn.BatchNorm1d(n_hidden_2)

    def forward(self, adj, x_in, idx):
        lst = list()
        lst.append(x_in)
        x = self.relu(self.fc1(torch.cat((x_in, torch.mm(adj, x_in)), 1)))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        x = self.relu(self.fc2(torch.cat((x, torch.mm(adj, x)), 1)))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        x = torch.cat(lst, dim=1)
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx) + 1, x.size(1)).to(x_in.device)
        x = out.scatter_add_(0, idx, x)
        x = self.relu(self.fc4(x))
        return x


def gnn_eval(model, A, tmp, feature, idx, device):
    feature[tmp, :] = 1
    output = model(A, feature, idx).squeeze()
    return output.cpu().detach().numpy().item()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

    feat_d = 50
    dropout = 0.4
    hidden = 64
    model = GNN_skip_small(feat_d, hidden, int(hidden / 2), int(hidden / 4), dropout).to(device)
    checkpoint = torch.load('models/model_best1.pth.tar' ,map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded")

    seed_size = 100
    alpha = 0.5

    fw = open("celf_glie_results.csv", "a")
    fw.write("graph,nodes,infl20,time20,infl100,time100\n")

    # for g in tqdm(["CA"]):
    for g in tqdm(["CA", "YT", "EN", "FB", "TW"]):

        print(g)
        if "l" in g:
            path = "data/sim_graphs/train/" + g + ".txt"
        else:
            if g == "CA":
                path = "data/real/CA-GrQc.inf"
            elif g == "YT":
                path = "data/real/youtube_combined.inf"
            elif g == "EN":
                path = "data/real/Email-Enron.inf"
            elif g == "FB":
                path = "data/real/facebook_combined.inf"
            elif g == "TW":
                path = "data/real/twitter_combined.inf"


        start = time.time()

        if not os.path.exists(path):
            print(f"File {path} không tồn tại")
            continue
        # G = pd.read_csv(path, header=None, sep=" ")
        # nodes = set(G[0].unique()).union(set(G[1].unique()))

        G_orig = pd.read_csv(path, header=None, sep=" ")
        G_orig.columns = ["source", "target", "weight"]

        # Ánh xạ ID nút thành chỉ số liên tục
        nodes = sorted(set(G_orig["source"].unique()).union(set(G_orig["target"].unique())))
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        G = G_orig.copy()
        G["source"] = G["source"].map(node_to_idx)
        G["target"] = G["target"].map(node_to_idx)

        # adj = sp.coo_matrix((G[2], (G[1], G[0])), shape=(len(nodes), len(nodes)))
        # adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
        # G.columns = ["source", "target", "weight"]
        adj = sp.coo_matrix((G["weight"], (G["source"], G["target"])), shape=(len(nodes), len(nodes)))
        adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)

        outdegree = G.groupby("source").agg({'target': 'count'}).reset_index()
        if g != "YT":
            deg_thres = np.histogram(outdegree.target, 20)[1][1]
        else:
            deg_thres = np.histogram(outdegree.target, 30)[1][1]

        nodes = outdegree.source[outdegree.target > deg_thres].values
        idx = torch.LongTensor(np.array([0] * adj.shape[0])).to(device)
        feature = torch.FloatTensor(np.zeros([adj.shape[0], feat_d])).to(device)

        S = []
        Q = []
        nid = 0
        mg = 1
        iteration = 2
        with torch.no_grad():
            for u in nodes:
                temp_l = []
                temp_l.append(u)
                temp_l.append(gnn_eval(model, adj, S + [u], feature.clone(), idx, device))
                temp_l.append(0)
                Q.append(temp_l)

        Q = sorted(Q, key=lambda x: x[1], reverse=True)
        S.append(Q[0][0])
        infl_spread = Q[0][1]
        Q = Q[1:]

        while len(S) < seed_size:
            u = Q[0]
            if u[iteration] != len(S):
                with torch.no_grad():
                    u[mg] = gnn_eval(model, adj, S + [u[nid]], feature.clone(), idx, device) - infl_spread
                u[iteration] = len(S)
                Q = sorted(Q, key=lambda x: x[1], reverse=True)
            else:
                infl_spread = u[mg] + infl_spread
                S.append(u[nid])
                if len(S) == 20:
                    x20 = time.time() - start
                Q = Q[1:]

        x100 = time.time() - start
        print("Xong, đang đánh giá..")

        x_ic100 = IC(G, S[:100], mc=100, alpha=alpha)
        x_ic20 = IC(G, S[:20], mc=100, alpha=alpha)

        fw.write(g.replace("\n", "") + ',"' + ",".join([str(i) for i in S]) + '",' +
                 str(x_ic20) + "," + str(x20) + "," + str(x_ic100) + "," + str(x100) + "\n")
        fw.flush()

    fw.close()


if __name__ == "__main__":
    main()