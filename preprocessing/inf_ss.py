import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import random
from IC_noisy_alpha_parallel import IC



# Code chính cho Greedy Influence Maximization
def run_greedy_influence_maximization():
    random.seed(1)
    labels = open("../data/influence_labels.csv", "a")
    seed_size = 10
    alpha = 0.5
    mc = 1000

    for g in tqdm(glob.glob("../data/sim_graphs/*.txt")):
        print(g)
        G = pd.read_csv(g, header=None, sep=" ")
        G.columns = ["source", "target", "weight"]
        nodes = set(G["target"].unique()).union(set(G["source"].unique()))

        Q = []
        S = []
        nid = 0
        mg = 1
        iteration = 2

        for u in tqdm(nodes):
            temp_l = []
            temp_l.append(u)
            temp_l.append(IC(G, [u], mc=mc, alpha=alpha))
            temp_l.append(0)
            Q.append(temp_l)

        Q = sorted(Q, key=lambda x: x[1], reverse=True)

        S = [Q[0][0]]
        infl_spread = Q[0][1]
        labels.write(g.replace(".txt", "") + ',"' + ','.join([str(tm) for tm in S]) + '",' + str(infl_spread) + "\n")
        labels.flush()
        Q = Q[1:]

        while len(S) < seed_size:
            u = Q[0]
            if u[iteration] != len(S):
                u[mg] = IC(G, S + [u[nid]], mc=mc, alpha=alpha) - infl_spread
                u[iteration] = len(S)
                Q = sorted(Q, key=lambda x: x[1], reverse=True)
            else:
                print(f"Kích thước S: {len(S)}")
                infl_spread = u[mg] + infl_spread
                S.append(u[nid])
                labels.write(g.replace(".txt", "") + ',"' + ','.join([str(tm) for tm in S]) + '",' + str(infl_spread) + "\n")
                labels.flush()
                Q = Q[1:]

    labels.close()

if __name__ == '__main__':
    run_greedy_influence_maximization()