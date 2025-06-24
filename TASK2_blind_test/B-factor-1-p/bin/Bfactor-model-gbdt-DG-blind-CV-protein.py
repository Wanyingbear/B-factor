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
import argparse
import sys


main_dir = os.path.join(os.getcwd(), "blind_test", "B-factor-1-p")
stride_main_dir = os.path.join(os.getcwd(), "blind_test", "B-factor-2-s")
stride_fp_dir = os.path.join(stride_main_dir, "Bfactor-Set364", "features")
stride_label_dir = os.path.join(stride_main_dir, "datasets", "labels")
results_path = os.path.join(main_dir, "results", "blind")
# ml_method = "GradientBoostingRegressor"

# list_PDBs = open(f"{main_dir}/datasets/list-avail.txt").read().splitlines()
_list_PDBs = open(f"{main_dir}/datasets/list-PDBs_CA_wanying-superset.txt").read().splitlines()
list_PDBs = []
for pdbid in _list_PDBs:
    if pdbid != "3P6J":
        list_PDBs.append(pdbid)

list_PDBs = np.array(list_PDBs)


def main(args):
    use_norm = True
    use_norm = False

    kf = KFold(n_splits=args.nkf, random_state=args.icycle, shuffle=True)
    train_index, test_index = list(kf.split(list_PDBs))[args.ikf]

    list_PDBs_train = list_PDBs[train_index]
    list_PDBs_test = list_PDBs[test_index]

    mag_feature_path = f"{main_dir}/features/mag_filtered_CV"
    X_train_all = []
    for pdbid in list_PDBs_train:
        save_feature_path = f"{mag_feature_path}/{pdbid}_CA-Mag.csv"  
        X_train = []
        save_path = f"{save_feature_path}"
       
        X_train.append(pd.read_csv(save_path, header=None).values)  
        X_train = np.concatenate(X_train, axis=1)  
        X_train_all.append(X_train)
    X_train_all = np.concatenate(X_train_all, axis=0)  

    print("DG train shape", np.shape(X_train_all))

    X_test_all = []
    for pdbid in list_PDBs_test:
        
        save_feature_path = f"{mag_feature_path}/{pdbid}_CA-Mag.csv"  
        X_test = []
        save_path = f"{save_feature_path}"
        # Use pandas to read the CSV file, and specify header=None to avoid reading the header
        X_test.append(pd.read_csv(save_path, header=None).values)  # Read CSV and convert to NumPy array
        X_test = np.concatenate(X_test, axis=1)  # If the CSV contains multiple columns, concatenate them along axis 1
        X_test_all.append(X_test)
    X_test_all = np.concatenate(X_test_all, axis=0)  # Combine all test data into a large array

    print("DG test shape", np.shape(X_test_all)) 

    # Global and local features
    feature_train_GL = []
    feature_test_GL = []
    for pdbid in list_PDBs_train:
        feature_path = f"{stride_fp_dir}/{pdbid}-onehot.csv"
        features_pdb = pd.read_csv(feature_path, header=None, index_col=None)
        feature_train_GL.append(features_pdb)
    feature_train_GL = np.concatenate(feature_train_GL, axis=0)
    print("Global train shape", np.shape(feature_train_GL))

    Feature_train = np.concatenate((X_train_all, feature_train_GL), axis=1)

    for pdbid in list_PDBs_test:
        feature_path = f"{stride_fp_dir}/{pdbid}-onehot.csv"
        features_pdb = pd.read_csv(feature_path, header=None, index_col=None).values
        feature_test_GL.append(features_pdb)
    feature_test_GL = np.concatenate(feature_test_GL, axis=0)
    print("Global test shape", np.shape(feature_test_GL))

    Feature_test = np.concatenate((X_test_all, feature_test_GL), axis=1)

    labels_train = []
    labels_test = []
    for pdbid in list_PDBs_train:
        label_path = f"{stride_label_dir}/{pdbid}.csv"
        label_pdb = pd.read_csv(label_path, header=None, index_col=None).values
        labels_train.append(label_pdb)
    labels_train = np.concatenate(labels_train).ravel()
    for pdbid in list_PDBs_test:
        label_path = f"{stride_label_dir}/{pdbid}.csv"
        label_test = pd.read_csv(label_path, header=None, index_col=None).values
        labels_test.append(label_test)
    labels_test = np.concatenate(labels_test).ravel()

    if use_norm:
        scaler = StandardScaler()
        scaler.fit(Feature_train)
        X_train = scaler.transform(Feature_train)
        X_test = scaler.transform(Feature_test)

    if args.ml_method == "GBDT":
        ml_method = "GradientBoostingRegressor"
        i = 20000
        i = 2000
        i = 1000
        # i = 100
        ntree = i
        j = 7
        k = 5
        m = 8
        lr = 0.002
        clf = globals()["%s" % ml_method](
            n_estimators=i,
            max_depth=j,
            min_samples_split=k,
            learning_rate=lr,
            subsample=0.1 * m,
            max_features="sqrt",
            random_state=args.icycle,
        )

    elif args.ml_method == "RF":
        i = 1000
        # i = 100
        ntree = i
        j = 8
        k = 4
        m = 2
        lr = 0.002
        ml_method = "RandomForestRegressor"
        clf = globals()["%s" % ml_method](
            n_estimators=i,
            max_depth=j,
            min_samples_split=k,
            # criterion="mse",
            min_samples_leaf=m,
        )

    print(np.shape(Feature_train), np.shape(labels_train))
    print(np.shape(Feature_test), np.shape(labels_test))

    clf.fit(Feature_train, labels_train)
    yP = clf.predict(Feature_test)
    corr, _ = pearsonr(yP, labels_test)
    rmse = np.sqrt(mean_squared_error(yP, labels_test))

    fw = open(
        f"{results_path}/R-n{args.nkf}-ikf{args.ikf}-c{args.icycle}-psl-{args.pslk}-blind-CV-protein-{args.ml_method}-ntree{ntree}.csv",
        "w",
    )
    print("R=%.3f,rmse=%.3f" % (corr, rmse), file=fw)
    fw.close()

    fw = open(
        f"{results_path}/pred-n{args.nkf}-ikf{args.ikf}-c{args.icycle}-psl-{args.pslk}-blind-CV-protein-{args.ml_method}-ntree{ntree}.csv",
        "w",
    )
    for ll in yP:
        print(ll, file=fw)
    fw.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Blind prediction for B-factor")
    parser.add_argument("--ml_method", type=str)
    parser.add_argument("--nkf", type=int)
    parser.add_argument("--icycle", type=int)
    parser.add_argument("--ikf", type=int)
    parser.add_argument("--pslk", type=str)
    args = parser.parse_args()
    print(args)

    main(args)
