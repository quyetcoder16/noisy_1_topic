import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import random

# Hàm single_simulation và IC (từ code bạn cung cấp)
def single_simulation(G, S, alpha, seed=None):
    if seed is not None:
        np.random.seed(seed)
    new_active, A = S[:], S[:]
    G_noisy = G.copy()
    noise = np.random.uniform(-alpha, alpha, len(G))
    G_noisy['weight'] = G['weight'] + G['weight'] * noise
    G_noisy['weight'] = np.clip(G_noisy['weight'], 0, 1)
    while new_active:
        temp = G_noisy.loc[G_noisy['source'].isin(new_active)]
        targets = temp['target'].tolist()
        ic = temp['weight'].tolist()
        coins = np.random.uniform(0, 1, len(targets))
        choice = [ic[c] > coins[c] for c in range(len(coins))]
        new_ones = np.extract(choice, targets)
        new_active = list(set(new_ones) - set(A))
        A += new_active
    return len(A)

def IC(G, S, mc=1000, alpha=0.5, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()
    seeds = np.random.randint(0, 1000000, mc)
    sim_func = partial(single_simulation, G, S, alpha)
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(sim_func, [(seed,) for seed in seeds])
    return np.mean(results)