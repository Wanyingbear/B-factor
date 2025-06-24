import os
import sys


sys.path.append(os.path.join(os.getcwd(), "blind_test", "B-factor-1-p", "src"))


import mag_path

mypath = mag_path.path_config()


def Bfactor_features_job():

    kernel_types = ["exp", "lor"]

    cv_target = "protein"
    #cv_target = "atom"

    ml_methods = ["RF", "GBDT"]

    nkf = 10
    ncycle = 5
    pslks = ["0", "1"][:1]
    use_sbatch = False
    use_sbatch = True
    ii = 0
    if not os.path.exists('jobs'):
            os.makedirs('jobs')

    batch_file = f"jobs/Bf-features.sh"
    fo = open(batch_file, "w")
    for pslk in pslks:
        for ml_method in ml_methods:
            for icycle in range(ncycle):
                for ikf in range(nkf):
                    for kernel_type in kernel_types[:1]:
                        job_file_name = f"{kernel_type}-nkf{nkf}-ikf{ikf}-c{icycle}-{ml_method}-{cv_target}-psl{pslk}"
                        job_file = f"jobs/{job_file_name}.sh"
                        fjob = open(job_file, "w")
                        lines = []
                        lines += ["#!/bin/bash"]
                        lines += ["#SBATCH --time=4:00:00"]
                        lines += ["#SBATCH --ntasks=1"]
                        lines += ["#SBATCH --cpus-per-task=1"]
                        lines += ["#SBATCH --mem=20gb"]
                        lines += ["#SBATCH -o jobs/%s.out" % (job_file_name)]
                        lines += ["#SBATCH --job-name %s" % job_file_name]
                        lines += ["#SBATCH --account=general"]
                        lines += [
                            f"python {mypath.main_dir}/bin/Bfactor-model-gbdt-DG-blind-CV-{cv_target}.py --ikf {ikf} --nkf {nkf} --icycle {icycle} --ml_method {ml_method} --pslk {pslk}"
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

    print(ii)
    os.system(f"bash {batch_file}")


Bfactor_features_job()
