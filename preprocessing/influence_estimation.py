import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
from multiprocessing import Pool, cpu_count
from functools import partial
import random

from diffuse import IC

# Thiết lập seed để tái lập
random.seed(1)

# Mở file để ghi kết quả
labels = open("../influence_labels.csv", "a")

seed_size = 10
alpha = 0.5  # Tham số nhiễu
mc = 1000

# Lặp qua các file đồ thị
for g in tqdm(glob.glob("../data/sim_graphs/*.txt")):
    print(g)
    # Đọc đồ thị
    G = pd.read_csv(g, header=None, sep=" ")
    G.columns = ["source", "target", "weight"]
    # Lấy tập hợp nút
    nodes = set(G["target"].unique()).union(set(G["source"].unique()))

    Q = []
    S = []
    nid = 0  # Chỉ số nút trong danh sách Q
    mg = 1  # Chỉ số marginal gain trong danh sách Q
    iteration = 2  # Chỉ số iteration trong danh sách Q

    # Khởi tạo Q với ảnh hưởng của từng nút
    for u in tqdm(nodes):  # Sử dụng nodes thay vì G.nodes()
        temp_l = []
        temp_l.append(u)  # Nút
        temp_l.append(IC(G, [u], mc=100, alpha=alpha))  # Ảnh hưởng
        temp_l.append(0)  # Iteration
        Q.append(temp_l)

    # Sắp xếp Q theo ảnh hưởng giảm dần
    Q = sorted(Q, key=lambda x: x[1], reverse=True)

    # Chọn nút đầu tiên
    S = [Q[0][0]]
    infl_spread = Q[0][1]
    labels.write(g.replace(".txt", "") + ',"' + ','.join([str(tm) for tm in S]) + '",' + str(infl_spread) + "\n")
    labels.flush()
    Q = Q[1:]

    # Lặp cho đến khi đạt seed_size
    while len(S) < seed_size:
        u = Q[0]
        # Kiểm tra nếu nút đã được cập nhật cho kích thước S hiện tại
        if u[iteration] != len(S):
            # Cập nhật marginal gain
            u[mg] = IC(G, S + [u[nid]], mc=100, alpha=alpha) - infl_spread
            u[iteration] = len(S)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
        else:
            print(f"Kích thước S: {len(S)}")
            # Thêm nút vào S
            infl_spread = u[mg] + infl_spread
            S.append(u[nid])
            labels.write(
                g.replace(".txt", "") + ',"' + ','.join([str(tm) for tm in S]) + '",' + str(infl_spread) + "\n")
            labels.flush()
            # Xóa nút khỏi Q
            Q = Q[1:]

labels.close()