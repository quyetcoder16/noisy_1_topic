import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial


def single_simulation(G, S, alpha, seed=None):
    """
    Chạy một mô phỏng Monte-Carlo duy nhất.
    Input:
        G: DataFrame pandas biểu diễn đồ thị
        S: danh sách nút hạt giống
        alpha: tham số nhiễu
        seed: hạt ngẫu nhiên để đảm bảo tái lập (nếu cần)
    Output: số nút bị ảnh hưởng trong mô phỏng này
    """
    if seed is not None:
        np.random.seed(seed)

    new_active, A = S[:], S[:]

    # Tạo bản sao đồ thị với trọng số có nhiễu
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
    print(f"S = {S}")
    """
    Independent Cascade model với trọng số nhiễu, chạy song song.
    Trọng số mới = Trọng số hiện tại + Trọng số hiện tại * random[-alpha, +alpha].
    Input:
        G: DataFrame pandas biểu diễn đồ thị với cột 'source', 'target', 'weight'
        S: danh sách nút hạt giống
        mc: số lần mô phỏng Monte-Carlo (mặc định: 1000)
        alpha: tham số kiểm soát biên độ nhiễu (mặc định: 0.5)
        num_processes: số tiến trình song song (mặc định: số CPU khả dụng)
    Output: số nút trung bình bị ảnh hưởng
    """
    # Thiết lập số tiến trình (mặc định là số CPU)
    if num_processes is None:
        num_processes = cpu_count()

    # Tạo danh sách seed ngẫu nhiên để đảm bảo các mô phỏng khác nhau
    seeds = np.random.randint(0, 1000000, mc)

    # Tạo hàm partial với các tham số cố định
    sim_func = partial(single_simulation, G, S, alpha)

    # Chạy song song với Pool
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(sim_func, [(seed,) for seed in seeds])

    # Tính trung bình độ lan truyền
    return np.mean(results)