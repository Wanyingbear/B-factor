
# Topological Magnitude for Protein Flexibility Analysis
<div align='center'>
[![License: MIT](https://opensource.org/licenses/MIT)](https://opensource.org/licenses/MIT)
</div>

**Title** - Topological Magnitude for Protein Flexibility Analysis

**Authors** - Wanying Bi, Hongsong Feng, Jie Wu, Jingyan Li, and Guo-Wei Wei

## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Datasets](#datasets)
- [Model Files](#model-files)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Introduction
This work presents a novel method for predicting protein flexibility based on the magnitude of finite metric spaces. Protein flexibility is critical for structural adaptability and biological function, influencing enzymatic catalysis, molecular recognition, and signal transduction. The magnitude, an invariant from category theory and algebraic topology, has been extended to graphs, hypergraphs, and material science. This study models protein structures as finite metric spaces with biologically relevant distance functions and uses magnitude to capture their geometric and topological properties, improving the prediction of B-factors, which represent atomic fluctuation. Experimental results on the Superset dataset show that our method achieves a Pearson correlation coefficient (PCC) of 0.725, outperforming GNM, pfFRI, ASPH, opFRI, and EH. Compared to GNM, our approach improves PCC by 28.32%, demonstrating its performance advantage. Additionally, by combining magnitude with global and local structural features, we validate the method’s robustness in blind tests, proving its effectiveness in capturing protein flexibility and offering a novel approach to protein dynamics analysis.

**Keywords**: Protein flexibility, Magnitude topology, Finite metric spaces, B-factor prediction.

## Model Architecture

The model architecture is based on the concept of magnitude from algebraic topology and category theory. It involves the following key components:

- **Finite Metric Space Representation**: Protein structures are modeled as finite metric spaces using biologically relevant distance functions.
- **Magnitude Calculation**: The magnitude of these finite metric spaces is calculated to capture geometric and topological properties.
- **Feature Extraction**: Magnitude values are used as features to predict protein flexibility.
- **Prediction Model**: A linear regression model is employed to predict B-factors based on the extracted magnitude features.

Further details about the model architecture and its components will be provided in the upcoming publication.

## Getting Started

### Prerequisites
The code in this repo has been tested with the following software versions:
- Python>=3.7.0
- numpy>=1.21.5
- scipy>=1.7.3
- scikit-learn>=0.24.2
- pandas>=1.3.5
- openpyxl>=3.0.9
- argparse>=1.1

You can install these packages using pip:
```bash
pip install numpy scipy scikit-learn pandas openpyxl argparse


We recommend using the Anaconda Python distribution, which is available for Windows, MacOS, and Linux. Installation for all required packages (listed above) has been tested using the standard instructions from the providers of each package.

# Datasets

## Task 1: Protein Analysis

The datasets used in the first task are available in the following directories:

- The Superset dataset, which includes 364 proteins, is located in:
  ```plaintext
  ./TASK1_Protein_analysis/data_upload/364
  ```
- The Large dataset is located in:
  ```plaintext
  ./TASK1_Protein_analysis/data_upload/Large
  ```
- The MD-data dataset, used for MD and Magnitude comparison, is located in:
  ```plaintext
  ./TASK1_Protein_analysis/data_upload/MD-data
  ```
- The Medium dataset is located in:
  ```plaintext
  ./TASK1_Protein_analysis/data_upload/Medium
  ```
- The Small dataset is located in:
  ```plaintext
  ./TASK1_Protein_analysis/data_upload/Small
  ```

## Task 2: Blind Test

The datasets used in the second task (blind test) are available in the following directories:

- The Large dataset is located in:
  ```plaintext
  ./TASK2_blind_test/Large
  ```
- The Medium dataset is located in:
  ```plaintext
  ./TASK2_blind_test/Medium
  ```
- The Small dataset is located in:
  ```plaintext
  ./TASK2_blind_test/Small
  ```
- The labels for the blind test dataset are located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-2-s/datasets/labels
  ```
- The features for the blind test dataset are located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-2-s/Bfactor-Set364/features
  ```
- The 364 dataset is located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-1-p/datasets/364
  ```
- The final list of 364 proteins is located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-1-p/datasets/list-364-final.txt
  ```
- The list of large PDBs is located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-1-p/datasets/list-PDBs_CA_wanying-large.txt
  ```
- The list of medium PDBs is located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-1-p/datasets/list-PDBs_CA_wanying-medium.txt
  ```
- The list of small PDBs is located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-1-p/datasets/list-PDBs_CA_wanying-small.txt
  ```
- The list of superset PDBs is located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-1-p/datasets/list-PDBs_CA_wanying-superset.txt
  ```
- The filtered magnitude features for the blind test are located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-1-p/features/mag_filtered
  ```
- The cross-validated filtered magnitude features for the blind test are located in:
  ```plaintext
  ./TASK2_blind_test/B-factor-1-p/features/mag_filtered_CV
  ```

To obtain the full datasets, please contact the corresponding authors at wanyingbi1015@163.com.

## Model Files

This repository contains the following Python files:

### Task 1: Protein Analysis

1. **`feature_extract_mag.py`** - This script is used to extract magnitude features from protein data.
   - Path: `TASK1_Protein_analysis/Code_upload/feature_extract_mag.py`

2. **`multi_protein_analysis.py`** - This script performs multi-protein analysis.
   - Path: `TASK1_Protein_analysis/Code_upload/multi_protein_analysis.py`

3. **`Single-RMSF-Mag.py`** - This script calculates single RMSF (Root Mean Square Fluctuation) and magnitude for proteins.
   - Path: `TASK1_Protein_analysis/Code_upload/MD-MAG/Single-RMSF-Mag.py`

4. **`feature_extract_single_rmsf_mag.py`** - This script extracts single RMSF and magnitude features from protein data.
   - Path: `TASK1_Protein_analysis/Code_upload/MD-MAG/feature_extract_single_rmsf_mag.py`

### Task 2: Blind Test

1. **`Bfactor-model-gbdt-DG-blind-CV-atom.py`** - This script trains a GBRT (Gradient Boosting Regression Tree) model for blind cross-validation on atomic level data.
   - Path: `TASK2_blind_test/B-factor-1-p/bin/Bfactor-model-gbdt-DG-blind-CV-atom.py`

2. **`Bfactor-model-gbdt-DG-blind-CV-protein.py`** - This script trains a GBRT model for blind cross-validation on protein level data.
   - Path: `TASK2_blind_test/B-factor-1-p/bin/Bfactor-model-gbdt-DG-blind-CV-protein.py`

3. **`Bfactor-model-gbdt-DG-blind-LOO.py`** - This script trains a GBRT model for blind leave-one-out validation.
   - Path: `TASK2_blind_test/B-factor-1-p/bin/Bfactor-model-gbdt-DG-blind-LOO.py`

4. **`submit_Bfactor-model-gbdt-DG-blind-CV.py`** - This script submits the results of the GBRT model for blind cross-validation.
   - Path: `TASK2_blind_test/submit_Bfactor-model-gbdt-DG-blind-CV.py`

5. **`submit_Bfactor-model-gbdt-DG-blind-LOO.py`** - This script submits the results of the GBRT model for blind leave-one-out validation.
   - Path: `TASK2_blind_test/submit_Bfactor-model-gbdt-DG-blind-LOO.py`

If you encounter any issues or have questions, please contact the authors at wanyingbi1015@163.com.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##Citation
If you use this code or the pre-trained models in your work, please cite our work:
Wanying Bi, Hongsong Feng, Jie Wu, Jingyan Li, and Guo-Wei Wei. "Topological Magnitude for Protein Flexibility Analysis" 

## Acknowledgements
This work was supported in part by Natural Science Foundation of China (NSFC) grant (11971144), High level Scientific Research Foundation of Hebei Province, the Start-up Research Fund from Yanqi Lake Beijing Institute of Mathematical Sciences and Applications, NIH grants R01AI164266, and R35GM148196, NSF grants DMS-2052983, DMS-2245903, and IIS-1900473, and MSU Research Foundation.

