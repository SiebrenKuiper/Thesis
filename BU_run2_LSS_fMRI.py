import glob
import os
import re
import traceback
from typing import List
import pandas as pd
from nilearn import image, plotting, glm

from config import (
    BASE_IMG_DIR, SKIP_LIST, CONFOUND_COLS,
    HRF_MODEL, DRIFT_MODEL, HIGH_PASS, NOISE_MODEL,
    SMOOTHING_FWHM, N_JOBS, MINIMIZE_MEMORY, VERBOSE
)

IMG_DIR = BASE_IMG_DIR
CLEANED_DIR = os.path.join(IMG_DIR, "cleaned")
LSS_OUTPUT_DIR = os.path.join(IMG_DIR, "LSS/lss_run02")
if not os.path.exists(LSS_OUTPUT_DIR):
    os.makedirs(LSS_OUTPUT_DIR, exist_ok=True)


def process_subject_lss(sub_id, cleaned_dir, output_root):
    print(f"\nProcessing subject: sub-{sub_id}")

    func_file = os.path.join(cleaned_dir,
        f"sub-{sub_id}_task-belief_run-02_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    events_file = os.path.join(cleaned_dir,
        f"sub-{sub_id}_task-belief_run-02_events.tsv")
    confounds_file = os.path.join(cleaned_dir,
        f"sub-{sub_id}_task-belief_run-02_desc-confounds_timeseries.tsv")

    for f in [func_file, events_file, confounds_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")
    beh_data = pd.read_csv(events_file, sep="\t")
    confound_data = pd.read_csv(confounds_file, sep="\t")
    events_df = beh_data.copy()
    events_df.rename(columns={"trial_type_mediation": "trial_type"}, inplace=True)
    # events_df["modulation"] = 1.0
    # events_df["Confi2"] -= events_df["Confi2"].mean()
    # events_df.loc[events_df["trial_type"] == "confi2", "modulation"] = events_df["Confi2"]
    events_df=events_df[["onset", "duration", "trial_type"]]
    confound_df  = confound_data[CONFOUND_COLS].fillna(0)
    
    TARGET_CONDITION = "event2"
    feedback_indices = events_df.index[
        events_df["trial_type"] == TARGET_CONDITION
    ].tolist()

    if not feedback_indices:
        print(f"  No '{TARGET_CONDITION}' trials found for sub-{sub_id}, skipping.")
        return False

    func_img     = image.load_img(func_file)
    lss_beta_maps = []

    for trial_number, i_trial in enumerate(feedback_indices, start=1):
        trial_name   = f"{TARGET_CONDITION}_{trial_number:02d}"
        dm_img_path  = os.path.join(output_root,
            f"sub-{sub_id}_design_matrix_run-02_trial-{trial_name}.svg")
        events_df_lss = events_df.copy()
        events_df_lss.at[i_trial, "trial_type"] = trial_name
        lss_glm = glm.first_level.FirstLevelModel(
            t_r=1,
            verbose=VERBOSE,
            noise_model=NOISE_MODEL,
            hrf_model=HRF_MODEL,
            drift_model=DRIFT_MODEL,
            high_pass=HIGH_PASS,
            minimize_memory=MINIMIZE_MEMORY,
            n_jobs=N_JOBS,
            smoothing_fwhm=SMOOTHING_FWHM,
        )
        lss_glm.fit(
            run_imgs=func_img,
            events=events_df_lss,
            confounds=confound_df,
        )
        design_matrix = lss_glm.design_matrices_[0]

        plotting.plot_design_matrix(
            design_matrix=design_matrix,
            output_file=dm_img_path,
        )
        beta_map = lss_glm.compute_contrast(
            contrast_def=trial_name, output_type="effect_size"
        )
        beta_map_path = os.path.join(output_root,
            f"sub-{sub_id}_task-belief_run-02_trial-{trial_name}_beta_map.nii")
        beta_map.to_filename(beta_map_path)

        lss_beta_maps.append(beta_map)

        z_map = lss_glm.compute_contrast(
            contrast_def=trial_name,
            stat_type="t",
            output_type="z_score"
        )
        plotting.plot_stat_map(
            stat_map_img=z_map,
            threshold=1.96,
            title=f"sub-{sub_id} — {trial_name} (z-map)",
            output_file=os.path.join(output_root, f"sub-{sub_id}_zmap_run-02_trial-{trial_name}.svg")
        )
        
    fourd_img = image.concat_imgs(lss_beta_maps)
    out_path  = os.path.join(output_root,
        f"sub-{sub_id}_task-belief_run-02_condition_{TARGET_CONDITION}_lss.nii")
    fourd_img.to_filename(out_path)
    print(f"  Saved {len(lss_beta_maps)} beta maps → {out_path}")

    return True

def collect_subject_ids(cleaned_dir: str) -> List[str]:
    func_pattern = os.path.join(cleaned_dir,
        "sub-*_task-belief_run-02_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    sub_ids = []
    for f in sorted(glob.glob(func_pattern)):
        match = re.search(r"sub-(\d+)", os.path.basename(f))
        if match:
            sub_ids.append(match.group(1))
    return sorted(set(sub_ids))

def main():
    print("**** Starting LSS first-level analysis for Study 2, Run 2 ****")
    sub_ids = collect_subject_ids(CLEANED_DIR)
    print(f"Found {len(sub_ids)} total subjects")

    success_subjects = []
    for sub_id in sub_ids:
        if int(sub_id) in SKIP_LIST:
            print(f"  Skipping sub-{sub_id} (in skip list)")
            continue
        try:
            ok = process_subject_lss(
                sub_id=sub_id,
                cleaned_dir=CLEANED_DIR,
                output_root=LSS_OUTPUT_DIR,
            )
            if ok:
                success_subjects.append(sub_id)
        except Exception as e:
            print(f"Error processing subject sub-{sub_id}: {e}")
            print("-" * 60)
            traceback.print_exc()
            print("-" * 60)
            continue

    print("\n" + "=" * 60)
    print(f"First-level analysis complete: {len(success_subjects)} / {len(sub_ids)} valid subjects")
    print("=" * 80)

if __name__ == "__main__":
    main()