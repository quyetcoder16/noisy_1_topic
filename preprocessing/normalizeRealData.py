import pandas as pd
import networkx as nx
import os
import numpy as np

def convert_edgelist_to_ic_graph(input_path, output_path, weight_model="weighted_cascade", min_weight=0.01):
    """
    Đọc file edgelist, chuyển thành đồ thị có hướng, tính trọng số cạnh, và lưu dưới dạng file .inf.

    Parameters:
    - input_path (str): Đường dẫn file đầu vào (edgelist).
    - output_path (str): Đường dẫn file đầu ra (.inf).
    - weight_model (str): Mô hình trọng số ("weighted_cascade" hoặc "trivalency").
    - min_weight (float): Trọng số tối thiểu cho các cạnh (mặc định 0.01).

    Returns:
    - bool: True nếu thành công, False nếu thất bại.
    """
    try:
        # Kiểm tra file đầu vào
        if not os.path.exists(input_path):
            print(f"Lỗi: File đầu vào không tồn tại: {input_path}")
            return False

        # Kiểm tra quyền ghi file đầu ra
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.access(output_dir, os.W_OK):
            print(f"Lỗi: Không có quyền ghi vào thư mục: {output_dir}")
            return False

        # Đọc file edgelist
        G = nx.read_edgelist(input_path, nodetype=int, create_using=nx.DiGraph())
        print(f"Đọc đồ thị từ {input_path}: {G.number_of_nodes()} nút, {G.number_of_edges()} cạnh")

        # Tính trọng số cạnh
        edges = []
        if weight_model == "weighted_cascade":
            for u, v in G.edges():
                in_degree = G.in_degree(v)
                weight = 1.0 / in_degree if in_degree > 0 else min_weight
                edges.append([u, v, weight])
        elif weight_model == "trivalency":
            np.random.seed(0)  # Đảm bảo tái lập
            for u, v in G.edges():
                weight = np.random.choice([0.1, 0.01, 0.001])
                edges.append([u, v, weight])
        else:
            raise ValueError(f"Mô hình trọng số không hỗ trợ: {weight_model}")

        # Lưu file .inf
        df = pd.DataFrame(edges, columns=["source", "target", "weight"])
        df.to_csv(output_path, header=None, sep=" ", index=False)
        print(f"Đã tạo file: {output_path}")
        return df

    except Exception as e:
        print(f"Lỗi khi xử lý {input_path}: {str(e)}")
        return None

def main():
    # Danh sách dataset
    datasets = [
        ("CA-GrQc.txt", "CA-GrQc.inf"),
        ("com-youtube.ungraph.txt", "com-youtube.ungraph.inf"),
        ("Email-Enron.txt","Email-Enron.inf"),
        ("facebook_combined.txt", "facebook_combined.inf"),
        ("twitter_combined.txt", "twitter_combined.inf"),
    ]

    # Đường dẫn thư mục
    raw_dir = "../data/raw"
    output_dir = "../data/real"

    # Tạo thư mục
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Chuyển đổi từng dataset
    for input_file, output_file in datasets:
        input_path = os.path.join(raw_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        convert_edgelist_to_ic_graph(input_path, output_path, weight_model="weighted_cascade", min_weight=0.01)

if __name__ == "__main__":
    main()