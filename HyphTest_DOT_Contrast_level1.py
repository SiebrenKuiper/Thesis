import glob
import os
import re
import shutil
import numpy as np
import pandas as pd
from nilearn import glm, image, plotting

from config import (
    BASE_IMG_DIR, SKIP_LIST, CONFOUND_COLS,
    HRF_MODEL, DRIFT_MODEL, HIGH_PASS, NOISE_MODEL, OVERSAMPLING,
    SMOOTHING_FWHM, N_JOBS, MINIMIZE_MEMORY, VERBOSE
)

IMG_DIR = BASE_IMG_DIR
CLEANED_DIR = os.path.join(IMG_DIR, "cleaned")
FIRST_LEVEL_DIR = os.path.join(IMG_DIR, "contrast/DOT_level1")
if os.path.exists(FIRST_LEVEL_DIR):
    shutil.rmtree(FIRST_LEVEL_DIR)
os.makedirs(FIRST_LEVEL_DIR, exist_ok=True)

CONTRASTS = [
    ("resp_correct-incorrect", "resp_correct - resp_incorrect"),
    ("confi_correct_rating", "confi_correct"),
    ("confi_incorrect_rating", "confi_incorrect"),
    ("rating_correct-incorrect", "confi_correct - confi_incorrect"),
]

def create_design_matrix(
    func_file: str,
    beh_file: str,
    confound_file: str,
    mask_file: str
):
    func_img = image.load_img(func_file)
    beh_data = pd.read_csv(beh_file, sep="\t")
    confound_data = pd.read_csv(confound_file, sep="\t")
    mask_img = image.load_img(mask_file)
    n_scans = func_img.shape[-1]
    tr = func_img.header.get_zooms()[-1]
    frame_times = np.arange(n_scans) * tr
    beh_data["duration"] = 0
    beh_data["modulation"] = np.where(
        beh_data["trial_type"].isin(["confi_correct", "confi_incorrect"]),
        beh_data["ConfRating"],
        1
    )
    beh_data = beh_data.sort_values(by="onset").reset_index(drop=True)
    beh_data = beh_data[["onset", "duration", "trial_type", "modulation"]]
    confound_data = confound_data[CONFOUND_COLS].fillna(0)
    design_matrix = glm.first_level.make_first_level_design_matrix(
        frame_times=frame_times,
        events=beh_data,
        hrf_model=HRF_MODEL,
        drift_model=DRIFT_MODEL,
        high_pass=HIGH_PASS,
        oversampling=OVERSAMPLING,
        add_regs=confound_data,
        add_reg_names=CONFOUND_COLS
    )
    return func_img, design_matrix, mask_img


def first_level_analysis():
    print("**** Starting First Level Analysis (MultiRun, fixed-effect model) ****")
    run_list = ["01", "02"]
    func_pattern = os.path.join(
        CLEANED_DIR,
        "sub-*_task-dot_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )
    func_files = sorted(glob.glob(func_pattern))
    sub_ids = sorted({
        m.group(1)
        for f in func_files
        if (m := re.search(r"sub-(\d+)_", os.path.basename(f)))
    })
    print(f"Found {len(sub_ids)} subjects")

    for sub_id in sub_ids:
        if int(sub_id) in SKIP_LIST:
            print(f"  Skipping sub-{sub_id} (in skip list)")
            continue
        print(f"\nProcessing subject: sub-{sub_id}")
        sub_run_imgs = []
        sub_design_matrices = []
        sub_valid_runs = []
        for run_id in run_list:
            func_file = os.path.join(
                CLEANED_DIR,
                f"sub-{sub_id}_task-dot_run-{run_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            )
            beh_file = os.path.join(
                CLEANED_DIR,
                f"sub-{sub_id}_task-dot_run-{run_id}_events.tsv"
            )
            confound_file = os.path.join(
                CLEANED_DIR,
                f"sub-{sub_id}_task-dot_run-{run_id}_desc-confounds_timeseries.tsv"
            )
            mask_file = os.path.join(
                IMG_DIR,
                f"masks/group_mask_dot_run{run_id}.nii"
            )
            missing_files = [
                f for f in [func_file, beh_file, confound_file, mask_file]
                if not os.path.exists(f)
            ]
            if missing_files:
                print(f"  Run {run_id}: missing files {missing_files} - skipping")
                continue
            try:
                func_img, design_matrix, mask_img = create_design_matrix(
                    func_file,
                    beh_file,
                    confound_file,
                    mask_file
                )
                plotting.plot_design_matrix(
                    design_matrix=design_matrix,
                    output_file=os.path.join(
                        FIRST_LEVEL_DIR,
                        f"sub-{sub_id}_run-{run_id}_design_matrix.svg"
                    )
                )

                print("  Fitting first-level GLM...")
                sub_run_imgs.append(func_img)
                sub_design_matrices.append(design_matrix)
                sub_valid_runs.append(run_id)
            except Exception as e:
                print(f"  Run {run_id}: error building design matrix: {e}")
                continue
        if not sub_run_imgs:
            print("  No valid runs - skipping subject")
            continue
        try:
            print(f"  Valid runs: {sub_valid_runs} - fitting FirstLevelModel...")
            fmri_glm = glm.first_level.FirstLevelModel(
                verbose=VERBOSE,
                noise_model=NOISE_MODEL,
                minimize_memory=MINIMIZE_MEMORY,
                n_jobs=N_JOBS,
                mask_img=mask_img,
                smoothing_fwhm=SMOOTHING_FWHM,
            )
            fmri_glm_fit = fmri_glm.fit(
                run_imgs=sub_run_imgs,
                design_matrices=sub_design_matrices
            )
            for name, contrast_def in CONTRASTS:
                cmap = fmri_glm_fit.compute_contrast(
                    contrast_def=contrast_def,
                    stat_type="t",
                    output_type="effect_size",
                )
                cmap.to_filename(
                    os.path.join(
                        FIRST_LEVEL_DIR,
                        f"sub-{sub_id}_contrast-{name}-cmap.nii.gz"
                    )
                )
                z_map = fmri_glm_fit.compute_contrast(
                    contrast_def=contrast_def,
                    stat_type="t",
                    output_type="z_score"
                )
                plotting.plot_stat_map(
                    stat_map_img=z_map,
                    threshold=1.96,
                    title=f"sub-{sub_id} — {name}-zmap",
                    output_file=os.path.join(
                        FIRST_LEVEL_DIR,
                        f"sub-{sub_id}_contrast-{name}-zmap.svg"
                    ),
                )
            print(f"  sub-{sub_id} done.")
        except Exception as e:
            print(f"  Error processing sub-{sub_id}: {e} — skipping.")
            continue
    print("=" * 70)
    print("First-level analysis complete.")

if __name__ == "__main__":
    first_level_analysis()