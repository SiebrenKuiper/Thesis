#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
# preprocessing through fmriprep
# wsl
# conda activate IMAGE (WSL, not Windows)
# bash /mnt/d/Research/MS_Thesis/Study2/Analysis/fMRIPrep.sh
# Wait for fMRIPrep to finish, very loooong time

conda activate IMAGE
python D:/Research/MS_Thesis/Study2/Analysis/Org_BU_beh.py
python D:/Research/MS_Thesis/Study2/Analysis/Org_BU_fMRI.py
python D:/Research/MS_Thesis/Study2/Analysis/Mask.py
python D:/Research/MS_Thesis/Study2/Analysis/Clean_Ques.py
python D:/Research/MS_Thesis/Study2/Analysis/Clean_BU_beh.py
python D:/Research/MS_Thesis/Study2/Analysis/HyphTest_BU_beh.py
python D:/Research/MS_Thesis/Study2/Analysis/HyphTest_BU_run1_Contrast_level1.py
python D:/Research/MS_Thesis/Study2/Analysis/HyphTest_BU_run1_Contrast_level2.py
python D:/Research/MS_Thesis/Study2/Analysis/HyphTest_BU_run2_Contrast_level1.py
python D:/Research/MS_Thesis/Study2/Analysis/HyphTest_BU_run2_Contrast_level2.py
python D:/Research/MS_Thesis/Study2/Analysis/BU_run1_LSS_fMRI.py
python D:/Research/MS_Thesis/Study2/Analysis/BU_run2_LSS_fMRI.py
matlab -batch "run('D:/Research/MS_Thesis/Study2/Analysis/HyphTest_BU_run1_Mediation.m')"
matlab -batch "run('D:/Research/MS_Thesis/Study2/Analysis/HyphTest_BU_run2_Mediation.m')"
python HyphTest_BU_run1_PPI_level1.py

# matlab -batch "run('D:/Research/MS_Thesis/Study2/Analysis/Transmat2csv_DOT.m')"

echo "all scripts executed successfully!"