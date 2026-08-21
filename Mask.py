import glob
import os
import re
import shutil
import numpy as np
from nilearn import image, maskers, datasets

INPUT_DIR = r"D:/Research/MS_Thesis/Study2/Output/Image/cleaned"
MASK_DIR = r"D:/Research/MS_Thesis/Study2/Output/Image/masks"
if os.path.exists(MASK_DIR):
    shutil.rmtree(MASK_DIR)
os.makedirs(MASK_DIR, exist_ok=True)
Atlas_MNI = datasets.load_mni152_template(resolution=1) 
Skip_list = [1, 22, 26, 28, 39, 41, 64, 67, 56]

def make_group_mask(task, run):
    pat = f"sub-*_task-{task}_run-{run:02d}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
    all_files = glob.glob(os.path.join(INPUT_DIR, pat))
    filtered_files = []
    for f in all_files:
        m = re.search(r"sub-(\d+)_", os.path.basename(f))
        if m:
            sub_id = int(m.group(1))
            if sub_id not in Skip_list:
                filtered_files.append(f)
            else:
                print(f"  Skipping {os.path.basename(f)} (sub-{sub_id} in skip list)")
        else:
            filtered_files.append(f)

    print(f"Task {task} run {run}: found {len(all_files)} masks, using {len(filtered_files)} after skipping.")
    mean_img = image.mean_img(filtered_files)
    group_mask = image.math_img("img > 0.2", img=mean_img)
    return group_mask

if __name__ == "__main__":
    combos = [
        #("dot", 1),
        #("dot", 2),
        ("belief", 1),
        ("belief", 2),
    ]
    for task, run in combos:
        group_mask = make_group_mask(task, run)
        group_mask.to_filename(
            os.path.join(MASK_DIR, f"group_mask_{task}_run{run:02d}.nii")
        )
    #optimism_masker = maskers.NiftiSpheresMasker(
    #    seeds=[(-58, 21, -1)], 
    #    radius=8,
    #    mask_img=Atlas_MNI,
    #)
    #optimism_masker.fit()
    #optimism_masker_img = optimism_masker.inverse_transform(np.array([[1]]))
    #optimism_masker_img.to_filename(os.path.join(MASK_DIR, "IFG_mask.nii"))