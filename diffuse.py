import pandas as pd
import numpy as np


def IC(G, S, mc=1000, alpha=0.5):
    """
    Independent Cascade model với trọng số cạnh có nhiễu dựa trên alpha.
    Trọng số mới = Trọng số hiện tại + Trọng số hiện tại * random[-alpha, +alpha].
    Input:
        G: DataFrame pandas biểu diễn đồ thị với các cột 'source', 'target', 'weight'
        S: danh sách các nút hạt giống (seed nodes)
        mc: số lần mô phỏng Monte-Carlo (mặc định: 1000)
        alpha: tham số kiểm soát biên độ nhiễu (mặc định: 0.5)
    Output: số nút trung bình bị ảnh hưởng bởi các nút hạt giống
    """

    # Lặp qua các mô phỏng Monte-Carlo
    spread = []
    for i in range(mc):
        # Mô phỏng quá trình lan truyền
        new_active, A = S[:], S[:]

        # Tạo bản sao đồ thị với trọng số có nhiễu
        G_noisy = G.copy()
        # Tạo nhiễu: random trong [-alpha, +alpha]
        noise = np.random.uniform(-alpha, alpha, len(G))
        # Tính trọng số mới: weight + weight * noise
        G_noisy['weight'] = G['weight'] + G['weight'] * noise
        # Giới hạn trọng số trong khoảng [0, 1]
        G_noisy['weight'] = np.clip(G_noisy['weight'], 0, 1)

        while new_active:
            # Lọc các cạnh có nguồn nằm trong new_active
            temp = G_noisy.loc[G_noisy['source'].isin(new_active)]
            # Lấy danh sách mục tiêu và trọng số nhiễu
            targets = temp['target'].tolist()
            ic = temp['weight'].tolist()
            # Xác định các hàng xóm được kích hoạt
            coins = np.random.uniform(0, 1, len(targets))
            choice = [ic[c] > coins[c] for c in range(len(coins))]

            # Lấy các nút mới được kích hoạt
            new_ones = np.extract(choice, targets)

            # Cập nhật new_active với các nút chưa được kích hoạt
            new_active = list(set(new_ones) - set(A))

            # Thêm các nút mới kích hoạt vào tập hợp nút đã kích hoạt
            A += new_active

        spread.append(len(A))

    return np.mean(spread)