import random
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from IC_noisy_alpha_parallel import IC

random.seed(1)
np.random.seed(1)


def main():
    seed_size = 100
    alpha = 0.5
    mc = 1000

    fw = open("celf_ic_results.csv", "a")
    fw.write("graph,nodes,infl20,time20,infl100,time100\n")

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
        G_orig = pd.read_csv(path, header=None, sep=" ")
        G_orig.columns = ["source", "target", "weight"]

        # Ánh xạ ID nút thành chỉ số liên tục
        nodes = sorted(set(G_orig["source"].unique()).union(set(G_orig["target"].unique())))
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        outdegree = G_orig.groupby("source").agg({'target': 'count'}).reset_index()
        if g != "YT":
            deg_thres = np.histogram(outdegree.target, 20)[1][1]
        else:
            deg_thres = np.histogram(outdegree.target, 30)[1][1]

        nodes_idx = [node_to_idx[node] for node in outdegree.source[outdegree.target > deg_thres].values]

        S = []
        Q = []
        nid = 0
        mg = 1
        iteration = 2
        for u in nodes_idx:
            temp_l = []
            temp_l.append(u)
            temp_l.append(IC(G_orig, [nodes[u]], mc=mc, alpha=alpha))
            temp_l.append(0)
            Q.append(temp_l)

        Q = sorted(Q, key=lambda x: x[1], reverse=True)
        S.append(Q[0][0])
        infl_spread = Q[0][1]
        Q = Q[1:]

        while len(S) < seed_size:
            u = Q[0]
            if u[iteration] != len(S):
                u[mg] = IC(G_orig, [nodes[s] for s in S] + [nodes[u[nid]]], mc=mc,
                           alpha=alpha) - infl_spread
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

        # Ánh xạ ngược chỉ số về ID nút gốc
        S_orig = [nodes[idx] for idx in S]
        x_ic100 = IC(G_orig, S_orig[:100], mc=mc, alpha=alpha)
        x_ic20 = IC(G_orig, S_orig[:20], mc=mc, alpha=alpha)

        fw.write(g.replace("\n", "") + ',"' + ",".join([str(i) for i in S_orig]) + '",' +
                 str(x_ic20) + "," + str(x20) + "," + str(x_ic100) + "," + str(x100) + "\n")
        fw.flush()

    fw.close()


if __name__ == "__main__":
    main()
