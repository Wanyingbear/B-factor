import numpy as np
import argparse
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os
from statistics import mean
from sklearn.preprocessing import StandardScaler
import sys

main_dir = os.path.join(os.getcwd(), "blind_test", "B-factor-1-p")
stride_main_dir = os.path.join(os.getcwd(), "blind_test", "B-factor-2-s")
stride_fp_dir = os.path.join(stride_main_dir, "Bfactor-Set364", "features")
stride_label_dir = os.path.join(stride_main_dir, "datasets", "labels")
results_path = os.path.join(main_dir, "results", "blind")


_list_PDBs = open(f"{main_dir}/datasets/list-PDBs_CA_wanying-superset.txt").read().splitlines()
list_PDBs = [pdbid for pdbid in _list_PDBs if pdbid != "3P6J"]
list_PDBs = np.array(list_PDBs)

def main(args):
    use_norm = False

    mag_feature_path = f"{main_dir}/features/mag_filtered_CV"
    print(f"mag feature path: {mag_feature_path}")

    X_train_all = []
    for pdbid in list_PDBs:
        save_feature_path = f"{mag_feature_path}/{pdbid}_CA-Mag.csv"
        if not os.path.exists(save_feature_path):
            raise FileNotFoundError(f"Feature file '{save_feature_path}' does not exist.")

        X_train = pd.read_csv(save_feature_path, header=None).values
        X_train_all.append(X_train)
    X_train_all = np.concatenate(X_train_all, axis=0)

    # Global and local features
    feature_train_GL = []
    for pdbid in list_PDBs:
        feature_path = f"{stride_fp_dir}/{pdbid}-onehot.csv"
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature file '{feature_path}' does not exist.")

        features_pdb = pd.read_csv(feature_path, header=None, index_col=None)
        feature_train_GL.append(features_pdb)
    feature_train_GL = np.concatenate(feature_train_GL, axis=0)
    print("Global total shape", np.shape(feature_train_GL))

    Feature_ALL = np.concatenate((X_train_all, feature_train_GL), axis=1)

    kf = KFold(n_splits=args.nkf, random_state=args.icycle, shuffle=True)
    train_index, test_index = list(kf.split(Feature_ALL))[args.ikf]

    Feature_train = Feature_ALL[train_index]
    Feature_test = Feature_ALL[test_index]

    print("Train shape", np.shape(Feature_train))
    print("Test shape", np.shape(Feature_test))

    labels_all = []
    for pdbid in list_PDBs:
        label_path = f"{stride_label_dir}/{pdbid}.csv"
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file '{label_path}' does not exist.")

        label_pdb = pd.read_csv(label_path, header=None, index_col=None).values
        labels_all.append(label_pdb)
    labels_all = np.concatenate(labels_all).ravel()

    labels_train = labels_all[train_index]
    labels_test = labels_all[test_index]

    if use_norm:
        scaler = StandardScaler()
        scaler.fit(Feature_train)
        Feature_train = scaler.transform(Feature_train)
        Feature_test = scaler.transform(Feature_test)

    if args.ml_method == "GBDT":
        clf = GradientBoostingRegressor(
            n_estimators=1000,
            max_depth=7,
            min_samples_split=5,
            learning_rate=0.002,
            subsample=0.8,
            max_features="sqrt",
            random_state=args.icycle,
        )
    elif args.ml_method == "RF":
        clf = RandomForestRegressor(
            n_estimators=1000,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=args.icycle,
        )
    else:
        raise ValueError(f"Unsupported ml_method '{args.ml_method}'. Use 'GBDT' or 'RF'.")

    clf.fit(Feature_train, labels_train)
    yP = clf.predict(Feature_test)
    corr, _ = pearsonr(yP, labels_test)
    rmse = np.sqrt(mean_squared_error(yP, labels_test))

    os.makedirs(results_path, exist_ok=True)
    with open(f"{results_path}/R-n{args.nkf}-ikf{args.ikf}-c{args.icycle}-psl-{args.pslk}-blind-CV-atom-{args.ml_method}-ntree1000.csv", "w") as fw:
        print(f"R={corr:.3f},rmse={rmse:.3f}", file=fw)

    with open(f"{results_path}/pred-n{args.nkf}-ikf{args.ikf}-c{args.icycle}-psl-{args.pslk}-blind-CV-atom-{args.ml_method}-ntree1000.csv", "w") as fw:
        for ll in yP:
            print(ll, file=fw)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blind prediction for B-factor")
    parser.add_argument("--ml_method", type=str, required=True, help="Machine learning method: 'GBDT' or 'RF'")
    parser.add_argument("--nkf", type=int, required=True, help="Number of K-Folds for cross-validation")
    parser.add_argument("--icycle", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--ikf", type=int, required=True, help="Index of K-Fold")
    parser.add_argument("--pslk", type=str, required=True, help="PSL feature selection index")
    args = parser.parse_args()

    print(args)
    main(args)
