import os

def path_config():
    

    class PathConfig:
        def __init__(self):
            # Get the current working directory (project root)
            self.project_root = os.getcwd()

            # Relative paths
            self.main_dir = os.path.join(self.project_root, "blind_test", "B-factor-1-p")
            self.stride_main_dir = os.path.join(self.project_root, "blind_test", "B-factor-2-s")
            
            # Dataset path for the first code
            self.dataset_path = os.path.join(self.main_dir, "datasets")
            # PSL feature path for the first code
            self.psl_feature_path = os.path.join(self.main_dir, "features", "PSL")
            # Stride feature path for the second code
            self.stride_fp_dir = os.path.join(self.stride_main_dir, "Bfactor-Set364", "features")
            # Stride label path for the second code
            self.stride_label_dir = os.path.join(self.stride_main_dir, "datasets", "labels")
            
            # Results path
            self.results_path = os.path.join(self.main_dir, "results", "blind")

        def __repr__(self):
            return f"PathConfig(main_dir={self.main_dir}, stride_main_dir={self.stride_main_dir}, " \
                   f"dataset_path={self.dataset_path}, psl_feature_path={self.psl_feature_path}, " \
                   f"stride_fp_dir={self.stride_fp_dir}, stride_label_dir={self.stride_label_dir}, " \
                   f"results_path={self.results_path})"

    return PathConfig()
