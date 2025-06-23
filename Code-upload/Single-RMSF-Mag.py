#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:54:25 2025

@author: wanyingbi
"""

import numpy as np
import csv
from feature_extract_single import WeightingCompute
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


class PdbAtom:
    def __init__(self, filepath):
        self.atom_coordinates = []
        self._load_ca_atoms(filepath)

    def _load_ca_atoms(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                if (line.startswith("ATOM") or line.startswith("HETATM")) and line[12:16].strip() == "CA":
                    coord = list(map(float, line[30:54].split()))
                    if np.all(np.isfinite(coord)):
                        self.atom_coordinates.append(coord)

    def get_coordinates(self):
        return np.array(self.atom_coordinates)


def read_rmsf_column(tsv_file, column_name="RMSF_R1", skip_header_lines=1):
    """读取RMSF文件中的指定列"""
    with open(tsv_file, 'r') as file:
        reader = csv.reader(file, delimiter="\t")

        for _ in range(skip_header_lines):
            header = next(reader)
        header = [h.strip() for h in header]  # 去除列名空格

        try:
            col_idx = header.index(column_name.strip())
        except ValueError:
            raise ValueError(f"列 '{column_name}' 不存在于文件，实际列为: {header}")

        values = []
        for row in reader:
            if len(row) > col_idx:
                try:
                    values.append(float(row[col_idx]))
                except ValueError:
                    print(f"警告：非浮点值：{row[col_idx]}")
        return np.array(values)


def compute_predicted_bfactors(coordinates, features):
    weightings = []
    for alpha, r in features:
        wc = WeightingCompute(alpha=alpha, r=r)
        weightings.append(wc.compute_weighting_vector(coordinates))
    X = np.column_stack(weightings)

    # 对权重做 PCA 或线性拟合等建模都可以。这里只是归一处理作为演示
    predicted = np.mean(X, axis=1)  # 简单平均作为预测
    return predicted


def compare_with_rmsf(predicted, rmsf):
    pearson_corr = pearsonr(predicted, rmsf)[0]
    print(f"Pearson correlation (Predicted B-factor vs RMSF_R1): {pearson_corr:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(predicted, label="Predicted B-factor")
    plt.plot(rmsf, label="RMSF_R1", linestyle='--')
    plt.xlabel("CA Atom Index")
    plt.ylabel("Value")
    plt.title("Predicted B-factor vs RMSF_R1")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pdb_file = "1fs1_C.pdb"
    tsv_rmsf_file = "1fs1_C_RMSF.tsv"

    features = [(0.3, 1), (0.4, 1), (0.5, 1), (0.6, 1), (0.7, 1),(0.8, 1),
                (0.3, 2), (0.4, 2), (0.5, 2), (0.6, 2), (0.7, 2),(0.8, 2)]

    # 提取CA原子坐标
    coords = PdbAtom(pdb_file).get_coordinates()

    # 提取RMSF值
    rmsf_values = read_rmsf_column(tsv_rmsf_file, column_name="RMSF_R1", skip_header_lines=1)

    if len(coords) != len(rmsf_values):
        raise ValueError(f"数量不一致：坐标点数 {len(coords)} 与 RMSF 值数 {len(rmsf_values)}")

    # 使用你的算法预测B-factor
    predicted_bfactors = compute_predicted_bfactors(coords, features)

    # 相关性分析
    compare_with_rmsf(predicted_bfactors, rmsf_values) 
    
    
    
    
    
    
    