import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import random
import time
from IC_noisy_alpha_parallel import IC

# Code chính để tạo tập huấn luyện
def generate_influence_train_set():
    random.seed(1)
    t = time.time()

    # Đọc file influence_labels.csv
    x = pd.read_csv("../data/influence_labels.csv", header=None)
    x.columns = ["graph", "node", "infl"]
    x["len"] = x.node.apply(lambda x: len(x.split(",")))

    gs = x.graph.unique()
    neg_samples = 30
    alpha = 0.5  # Tham số nhiễu
    mc = 1000

    # Mở file để ghi kết quả
    labels = open("../data/influence_train_set.csv", "a")

    # Lặp qua các đồ thị
    for g in gs:
        print(g)
        tmp = x[x.graph == g]

        # Đọc đồ thị
        G = pd.read_csv( "../"+g + ".txt", header=None, sep=" ")
        G.columns = ["source", "target", "weight"]
        nodes = set(G["target"].unique()).union(set(G["source"].unique()))

        # Lặp qua các tập hạt giống tối ưu
        for i, row in tmp.iterrows():
            seeds = row["node"].split(",")
            # Tạo các tập hạt giống ngẫu nhiên
            for k in range(neg_samples):
                neg_seeds = set(random.sample(list(nodes), row["len"]))
                counter = 0
                while neg_seeds == set(seeds):
                    counter += 1
                    neg_seeds = set(random.sample(list(nodes), row["len"]))
                    if counter > 10:
                        print("Stuck at generating negative sample")
                        break;

                # Tính độ lan truyền với nhiễu
                sigma = IC(G, list(neg_seeds), mc=mc, alpha=alpha)
                # Ghi tập hạt giống ngẫu nhiên
                labels.write(row["graph"] + ',"' + ",".join([str(ng) for ng in neg_seeds]) + '",' + str(sigma) + "\n")

            # Ghi tập hạt giống tối ưu
            labels.write(row["graph"] + ',"' + row["node"] + '",' + str(row["infl"]) + "\n")

        labels.flush()

    labels.close()
    print(f"Labeling time: {time.time() - t} seconds")

if __name__ == '__main__':
    generate_influence_train_set()