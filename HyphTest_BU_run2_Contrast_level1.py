import glob
import os
import re
import shutil
import numpy as np
import pandas as pd
from nilearn import glm, image, plotting

from config import (
    BASE_IMG_DIR, SKIP_LIST, CONFOUND_COLS,
    HRF_MODEL, DRIFT_MODEL, HIGH_PASS, NOISE_MODEL,OVERSAMPLING,
    SMOOTHING_FWHM, N_JOBS, MINIMIZE_MEMORY, VERBOSE
)

IMG_DIR = BASE_IMG_DIR
CLEANED_DIR = os.path.join(IMG_DIR, "cleaned")
FIRST_LEVEL_DIR = os.path.join(IMG_DIR, "contrast/BU_run2_level1")
if os.path.exists(FIRST_LEVEL_DIR):
    shutil.rmtree(FIRST_LEVEL_DIR)
os.makedirs(FIRST_LEVEL_DIR, exist_ok=True)

CONTRASTS = [
    ("event2_desirable", "event2_desirable"),
    ("event2_undesirable", "event2_undesirable"),
    ("confi2_desirable", "confi2_desirable"),
    ("confi2_undesirable", "confi2_undesirable"),
    ("event2_desirable-undesirable", "event2_desirable - event2_undesirable"),
    ("Confi2_desirable-undesirable", "confi2_desirable - confi2_undesirable"),
]

def create_design_matrix(func_file: str, beh_file: str, confound_file: str):
    func_img = image.load_img(func_file)
    beh_data = pd.read_csv(beh_file, sep="\t")
    confound_data = pd.read_csv(confound_file, sep="\t")
    n_scans = func_img.shape[-1]
    tr = func_img.header.get_zooms()[-1]
    frame_times = np.arange(n_scans) * tr

    events_df = beh_data[["onset", "duration", "trial_type_contrast", "EE", "Confi2"]].copy()
    events_df["modulation"] = 1.0
    events_df.rename(columns={"trial_type_contrast": "trial_type"}, inplace=True)
    events_df["EE"] = events_df["EE"].abs()
    events_df["EE"] -= events_df["EE"].mean()
    events_df["Confi2"] -= events_df["Confi2"].mean()
    events_df.loc[events_df["trial_type"].str.startswith("event2_"), "modulation"] = events_df["EE"]
    events_df.loc[events_df["trial_type"].str.startswith("confi2_"), "modulation"] = events_df["Confi2"]
    events_df = events_df[["onset", "duration", "trial_type", "modulation"]]
    confounds_df = confound_data[CONFOUND_COLS].fillna(0)
    design_matrix = glm.first_level.make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model=HRF_MODEL,
        drift_model=DRIFT_MODEL,
        high_pass=HIGH_PASS,
        oversampling=OVERSAMPLING,
        add_regs=confounds_df,
        add_reg_names=CONFOUND_COLS,
    )
    return func_img, design_matrix

def first_level_analysis() -> None:
    print("**** Starting First-Level Analysis (Run 02) ****")
    func_pattern = os.path.join(
        CLEANED_DIR,
        "sub-*_task-belief_run-02_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
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

        func_file = os.path.join(
            CLEANED_DIR,
            f"sub-{sub_id}_task-belief_run-02_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
        beh_file = os.path.join(
            CLEANED_DIR,
            f"sub-{sub_id}_task-belief_run-02_events.tsv"
        )
        confound_file = os.path.join(
            CLEANED_DIR,
            f"sub-{sub_id}_task-belief_run-02_desc-confounds_timeseries.tsv"
        )

        missing = [f for f in (func_file, beh_file, confound_file) if not os.path.exists(f)]
        if missing:
            print(f"  Skipping sub-{sub_id} — missing files: {missing}")
            continue

        try:
            func_img, design_matrix = create_design_matrix(
                func_file,
                beh_file,
                confound_file
            )
            plotting.plot_design_matrix(
                design_matrix=design_matrix,
                output_file=os.path.join(
                    FIRST_LEVEL_DIR,
                    f"sub-{sub_id}_design_matrix.svg"
                ),
            )
            
            print("  Fitting first-level GLM...")
            fmri_glm = glm.first_level.FirstLevelModel(
                verbose=VERBOSE,
                noise_model=NOISE_MODEL,
                minimize_memory=MINIMIZE_MEMORY,
                n_jobs=N_JOBS,
                mask_img=False,
                smoothing_fwhm=SMOOTHING_FWHM,
            )
            fmri_glm_fit = fmri_glm.fit(
                run_imgs=func_img,
                design_matrices=design_matrix
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

                zmap = fmri_glm_fit.compute_contrast(
                    contrast_def=contrast_def,
                    stat_type="t",
                    output_type="z_score",
                )
                plotting.plot_stat_map(
                    stat_map_img=zmap,
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