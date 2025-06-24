"""
Created on Thu Jun 19 15:54:25 2025
Author: wanyingbi
"""

import os
import numpy as np
import csv
from feature_extract_single_rmsf_mag import WeightingCompute
from scipy.stats import pearsonr


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
    """Read a specific column from RMSF TSV file."""
    with open(tsv_file, 'r') as file:
        reader = csv.reader(file, delimiter="\t")

        for _ in range(skip_header_lines):
            header = next(reader)
        header = [h.strip() for h in header]

        try:
            col_idx = header.index(column_name.strip())
        except ValueError:
            raise ValueError(f"Column '{column_name}' not found. Available columns: {header}")

        values = []
        for row in reader:
            if len(row) > col_idx:
                try:
                    values.append(float(row[col_idx]))
                except ValueError:
                    print(f"Warning: Non-numeric value found: {row[col_idx]}")
        return np.array(values)


def compute_predicted_bfactors(coordinates, features):
    weightings = []
    for alpha, r in features:
        wc = WeightingCompute(alpha=alpha, r=r)
        weightings.append(wc.compute_weighting_vector(coordinates))
    X = np.column_stack(weightings)
    predicted = np.mean(X, axis=1)
    return predicted


def compare_with_rmsf(predicted, rmsf):
    pearson_corr = pearsonr(predicted, rmsf)[0]
    print(f"Pearson correlation (Predicted B-factor vs RMSF_R1): {pearson_corr:.4f}")


if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))  

    pdb_file = os.path.normpath(os.path.join(script_dir, "..", "..", "data-upload", "MD-data", "1fd3_A.pdb"))
    tsv_rmsf_file = os.path.normpath(os.path.join(script_dir, "..", "..", "data-upload", "MD-data", "1fd3_A_RMSF.tsv"))

    print("pdb_file:", pdb_file)
    print("tsv_rmsf_file:", tsv_rmsf_file)

    features = [
        (0.3, 1), (0.4, 1), (0.5, 1), (0.6, 1), (0.7, 1), (0.8, 1),
        (0.3, 2), (0.4, 2), (0.5, 2), (0.6, 2), (0.7, 2), (0.8, 2)
    ]

    coords = PdbAtom(pdb_file).get_coordinates()
    rmsf_values = read_rmsf_column(tsv_rmsf_file, column_name="RMSF_R1", skip_header_lines=1)

    if len(coords) != len(rmsf_values):
        raise ValueError(f"Mismatch: {len(coords)} coordinates vs {len(rmsf_values)} RMSF values")

    predicted_bfactors = compute_predicted_bfactors(coords, features)
    compare_with_rmsf(predicted_bfactors, rmsf_values)
