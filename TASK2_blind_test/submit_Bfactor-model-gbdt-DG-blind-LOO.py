import os
import sys

sys.path.append(os.path.join(os.getcwd(), "blind_test", "B-factor-1-p", "src"))

import mag_path

mypath = mag_path.path_config()



#ml_method = "RF"
ml_method = "GBDT"


def Bfactor_features_job():

    pslk = "1"

    datasets = ["small", "medium", "lagre", "superset"]

    ii = 0
    for dataset in datasets:
        dataset_path = "/mnt/home/biwanyin/blind_test/B-factor-1-p"  # 修改为正确的路径
        _list_PDBs = (
            open(f"{dataset_path}/datasets/list-PDBs_CA_wanying-{dataset}.txt")  # 修改文件路径为正确的
            .read()
            .splitlines()
        )

        list_PDBs = []
        for pdbid in _list_PDBs:
            if pdbid != "3P6J":
                list_PDBs.append(pdbid)

        ncycle = 4
        use_sbatch = False
        use_sbatch = True
        if not os.path.exists('jobs'):
            os.makedirs('jobs')

        batch_file = f"jobs/Bf-features-{dataset}.sh"
        fo = open(batch_file, "w")
        for icycle in range(ncycle, ncycle + 1):
            for pdbid_loo in list_PDBs:
                job_file_name = f"{dataset}-c{icycle}-{pdbid_loo}-{ml_method}-psl{pslk}"
                job_file = f"jobs/{job_file_name}.sh"
                fjob = open(job_file, "w")
                lines = []
                lines += ["#!/bin/bash"]
                lines += ["#SBATCH --time=2:00:00"]
                lines += ["#SBATCH --ntasks=1"]
                lines += ["#SBATCH --cpus-per-task=1"]
                lines += ["#SBATCH --mem=10gb"]
                lines += ["#SBATCH -o jobs/%s.out" % (job_file_name)]
                lines += ["#SBATCH --job-name %s" % job_file_name]
                lines += ["#SBATCH --account=general"]
                lines += [
                    f"python {mypath.main_dir}/bin/Bfactor-model-gbdt-DG-blind-LOO.py --dataset {dataset} --icycle {icycle} --pdbid_loo {pdbid_loo} --ml_method {ml_method} --pslk {pslk}"
                ]
                if use_sbatch:
                    lines += ["scontrol show job $SLURM_JOB_ID"]

                for line in lines:
                    fjob.write(line + "\n")
                fjob.close()

                if use_sbatch:
                    print(f"sbatch {job_file}&", file=fo)

                if not use_sbatch:
                    os.system(f"bash {job_file}")
                    exit()
                ii += 1
        fo.close()

        # exit()
        os.system(f"bash {batch_file}")
    print(ii)


Bfactor_features_job()
