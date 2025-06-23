
import os
import csv
import numpy as np
from feature_extract_364 import WeightingCompute
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from openpyxl import Workbook


class PdbAtom:
    def __init__(self, filepath):
        self.filepath = filepath
        self.atom_coordinates = []
        self.b_factors = []
        self._load_ca_atoms_from_file()

    def _load_ca_atoms_from_file(self):
        """Load C-alpha atom coordinates and B-factors from the file"""
        with open(self.filepath, "r") as file:
            for line in file:
                if line.startswith("ATOM") and "CA" in line:
                    coord = list(map(float, line[30:54].split()))
                    bfactor = float(line[60:66].strip())
                    if np.all(np.isfinite(coord)) and not np.isnan(bfactor):
                        self.atom_coordinates.append(coord)
                        self.b_factors.append(bfactor)

    def get_atom_coordinates(self):
        return np.array(self.atom_coordinates)

    def get_bfactors(self):
        return np.array(self.b_factors)

    def get_ca_count(self):
        return len(self.atom_coordinates)


def process_pdb_file(pdb_file, features):
    """Process a single PDB file, including loading C-alpha atoms and calculating Pearson correlation coefficient"""
    try:
        pdb_atom = PdbAtom(pdb_file)
        atom_coords = pdb_atom.get_atom_coordinates()
        b_factors = pdb_atom.get_bfactors()

        # Calculate the weighting vector
        weightings = []
        for alpha, r in features:
            magnitude = WeightingCompute(alpha=alpha, r=r)
            weightings.append(magnitude.compute_weighting_vector(atom_coords))

        # Linear regression model f = a * x1 + b * x2 + c * x3 + ... + z
        X = np.column_stack(weightings)
        model = LinearRegression()
        model.fit(X, b_factors)
        predicted_b_factors = model.predict(X)

        # Calculate Pearson correlation coefficient
        pearson_corr, _ = pearsonr(predicted_b_factors, b_factors)

        return os.path.basename(pdb_file), pearson_corr, len(atom_coords)
    except Exception as e:
        print(f"Error processing file {pdb_file}: {e}")
        return os.path.basename(pdb_file), None, 0


def compute_pearson_and_ca(pdb_folder, features, excel_output_file, csv_output_file):
    """Process all PDB files, calculate Pearson correlation coefficients and C-alpha atom counts"""
    results = []
    ca_counts = []
    pdb_files = [
        os.path.join(pdb_folder, f) for f in os.listdir(pdb_folder) if f.endswith(".pdb")
    ]

    # Process each file
    for pdb_file in pdb_files:
        pdb_filename, pearson_corr, ca_count = process_pdb_file(pdb_file, features)
        results.append((pdb_filename, pearson_corr))
        ca_counts.append((pdb_filename, ca_count))

    # Calculate average Pearson correlation coefficient
    pearson_values = [pearson_corr for _, pearson_corr in results if pearson_corr is not None]
    average_pearson_corr = np.mean(pearson_values) if pearson_values else None

    print(f"Average Pearson correlation coefficient: {average_pearson_corr}")

    # Save results to CSV and Excel
    save_results_to_csv(results, average_pearson_corr, csv_output_file)
    save_results_to_excel(ca_counts, excel_output_file)
    print(f"Results have been saved to {csv_output_file} and {excel_output_file}")


def save_results_to_csv(results, average_pearson_corr, output_file):
    """Save Pearson correlation results to a CSV file"""
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["PDB_File", "Pearson_Correlation"])
        writer.writerows(results)
        writer.writerow(["Average", average_pearson_corr])


def save_results_to_excel(results, output_file):
    """Save C-alpha atom counts to an Excel file"""
    wb = Workbook()
    ws = wb.active
    ws.title = "CA Atom Counts"

    # Write header
    ws.append(["PDB_File", "CA_Atom_Count"])

    # Write results for each file
    for pdb_filename, ca_count in results:
        ws.append([pdb_filename, ca_count])

    # Save Excel file
    wb.save(output_file)


if __name__ == "__main__":
    # Set paths and parameters
    pdb_folder = "/Users/Paths to 364\small\medium\large protein files./364/"
    excel_output_file = "ca_atom_counts.xlsx"
    csv_output_file = "pearson_results.csv"

    # Input feature list (alpha, r)
    # features = [(0.3, 1), (0.4, 1), (0.5, 1), (0.6, 1), (0.7, 1), (0.8, 1), (0.9, 1), (1, 1), (1.1, 1), (1.2, 1)]
    features = [(0.3, 1), (0.4, 1), (0.5, 1), (0.6, 1), (0.7, 1), (0.8, 1),
                (0.3, 2), (0.4, 2), (0.5, 2), (0.6, 2), (0.7, 2), (0.8, 2)]

    # Process PDB files
    compute_pearson_and_ca(pdb_folder, features, excel_output_file, csv_output_file)
