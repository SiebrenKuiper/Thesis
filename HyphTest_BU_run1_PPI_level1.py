import glob
import os
import re
import shutil
import numpy as np
import pandas as pd
from nilearn import glm, image, plotting, maskers
import numpy as np
from scipy import signal, interpolate
from sklearn import linear_model


## Ref:  https://osf.io/w86nt
def events_to_impulse_series(events_run, n_scans, t_r, condition=None, y_feature="trial_type"):
    cols = ["onset", "duration", y_feature]
    reg_vals = events_run.loc[:, cols]
 
    if condition is not None:
        mask = reg_vals[y_feature] == condition
        reg_vals = reg_vals[mask]
 
    reg_vals = reg_vals.values.T
    reg_vals_onsets = reg_vals[0, :].astype(int)

    total_duration = int(n_scans * t_r)
    predictor_all = np.zeros(total_duration)

    # sub-77 miss the last two volumes
    valid_mask = reg_vals_onsets < total_duration
    if not valid_mask.all():
        dropped = reg_vals_onsets[~valid_mask]
        print(
            f"    Warning: dropping {len(dropped)} event onset(s) "
            f"{dropped.tolist()}s — run duration is only {total_duration}s"
        )
        reg_vals_onsets = reg_vals_onsets[valid_mask]

    predictor_all[reg_vals_onsets] = 1
 
    original_scale = np.arange(0, len(predictor_all), 1)
    resampler = interpolate.interp1d(original_scale, predictor_all, kind="linear", fill_value="extrapolate")
    desired_scale = np.linspace(0, len(predictor_all), int(n_scans))
    event_scans = np.ceil(resampler(desired_scale))
 
    return event_scans


def dct_mat(N_, K_):
    n = np.array(range(N_)).T
    C_ = np.zeros((n.shape[0], K_))
    C_[:, 0] = np.ones(n.shape[0]) / np.sqrt(N_)
    for q in range(1, K_):
        C_[:, q] = np.sqrt(2 / N_) * np.cos(np.pi * (2 * n) * q / (2 * N_))
    return C_

def deconvolve_glover_HRF(x, t_r):
    N = x.shape[0]
    hrf = glm.first_level.hemodynamic_models.glover_hrf(
        t_r=t_r, 
        oversampling=1,
        time_length=32)
    hrf /= np.max(hrf) 
    xb = dct_mat(N, N)
    Hxb = np.zeros((N, N))
    for i in range(N):
        Hxb[:, i] = signal.convolve(xb[:, i], hrf)[:N]
    reg = linear_model.Ridge(alpha=1, solver="lsqr", fit_intercept=False, max_iter=1000)
    reg.fit(Hxb, x)
    neurosignal = np.matmul(xb, np.ravel(reg.coef_)[:N])
    return neurosignal

def convolve_glover_HRF(x, t_r):
    hrf = glm.first_level.hemodynamic_models.glover_hrf(
        t_r=t_r, 
        oversampling=1,
        time_length=32)
    hrf /= np.max(hrf) 
    boldsignal = signal.convolve(x, hrf)[:len(x)]
    return boldsignal

from config import (
    BASE_IMG_DIR, SKIP_LIST, CONFOUND_COLS, CONFOUND_MOTION,
    HRF_MODEL, DRIFT_MODEL, HIGH_PASS, NOISE_MODEL, OVERSAMPLING,
    SMOOTHING_FWHM, N_JOBS, MINIMIZE_MEMORY, VERBOSE
)

CONTRASTS = [
    ("seed_signal_BOLD", "seed_signal_BOLD"),
    ("ppi_desirFB", "ppi_desirFB"),
    ("ppi_undesirFB", "ppi_undesirFB"),
    ("ppi_undesirFB - ppi_desirFB", "ppi_undesirFB - ppi_desirFB")
]

IMG_DIR = BASE_IMG_DIR
CLEANED_DIR = os.path.join(IMG_DIR, "cleaned")
FIRST_LEVEL_DIR = os.path.join(IMG_DIR, "PPI/BU_run1_level1")
if os.path.exists(FIRST_LEVEL_DIR):
    shutil.rmtree(FIRST_LEVEL_DIR)
os.makedirs(FIRST_LEVEL_DIR, exist_ok=True)
GROUP_MASK = None
# os.path.join(IMG_DIR, "masks/group_mask_belief_run01.nii")
# IFG_MASK = os.path.join(IMG_DIR, "masks/IFG_mask.nii")


def create_design_matrix(func_file: str, beh_file: str, confound_file: str):
    func_img = image.load_img(func_file)
    beh_data = pd.read_csv(beh_file, sep="\t")
    confound_data = pd.read_csv(confound_file, sep="\t")
    n_scans = func_img.shape[-1]
    tr = func_img.header.get_zooms()[-1]
    frame_times = np.arange(n_scans) * tr
    events_df = beh_data.copy()
    confounds_df = confound_data[CONFOUND_COLS].fillna(0)
    events_df.rename(columns={"trial_type_contrast": "trial_type"}, inplace=True)
    events_df["modulation"] = 1.0
    events_df["EE"] = events_df["EE"].abs()
    events_df["EE"] -= events_df["EE"].mean()
    events_df["Confi1"] -= events_df["Confi1"].mean()
    events_df["Estim1"] -= events_df["Estim1"].mean()
    events_df.loc[events_df["trial_type"].str.startswith("feedback_"), "modulation"] = events_df["EE"]
    # events_df.loc[events_df["trial_type"] == "confi1", "modulation"] = events_df["Confi1"]
    # events_df.loc[events_df["trial_type"] == "event", "modulation"] = events_df["Estim1"]
    events_df = events_df[["onset", "duration", "trial_type"]]
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

    masker = maskers.NiftiSpheresMasker(
        seeds=[(-58, 21, -1)],
        radius=8, 
        # mask_img=IFG_MASK,
        # target_affine=func_img.affine,
        # target_shape=func_img.shape[:3],
        standardize=False,
        detrend=True,
        smoothing_fwhm=SMOOTHING_FWHM,
        verbose=VERBOSE,
        t_r=tr,
    )
    roi_ts = masker.fit_transform(
        func_img,
        confounds=confounds_df[CONFOUND_MOTION].values
    )

    seed_bold = np.mean(roi_ts, axis=1)
    seed_neural = deconvolve_glover_HRF(seed_bold, t_r=tr)

    impulse_desir = events_to_impulse_series(
        events_run=events_df, n_scans=n_scans, t_r=tr, condition="feedback_desirable"
    )
    impulse_undesir = events_to_impulse_series(
        events_run=events_df, n_scans=n_scans, t_r=tr, condition="feedback_undesirable"
    )
    seed_neural -= seed_neural.mean()
    # impulse_desir -= impulse_desir.mean()
    # impulse_undesir -= impulse_undesir.mean()

    ppi_desir = convolve_glover_HRF(impulse_desir * seed_neural, t_r=tr)[:n_scans]
    ppi_undesir = convolve_glover_HRF(impulse_undesir * seed_neural, t_r=tr)[:n_scans]

    design_matrix = pd.concat([
        pd.Series(seed_bold, name="seed_signal_BOLD"), 
        pd.Series(ppi_desir, name="ppi_desirFB"),
        pd.Series(ppi_undesir, name="ppi_undesirFB"),
        design_matrix
    ], axis=1)
    return func_img, design_matrix



def Psycho_Physio_Interaction():
    print("Starting first-level PPI analysis...")
    func_pattern = os.path.join(
        CLEANED_DIR,
        "sub-*_task-belief_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    )
    func_files = sorted(glob.glob(func_pattern))

    sub_ids = sorted({
        m.group(1)
        for f in func_files
        if (m := re.search(r"sub-(\d+)_", os.path.basename(f)))
    })
    print(f"Found {len(sub_ids)} subjects")
    brain_mask = image.load_img(GROUP_MASK)

    for sub_id in sub_ids:
        if int(sub_id) in SKIP_LIST:
            print(f"  Skipping sub-{sub_id} (in skip list)")
            continue

        print(f"\nProcessing subject: sub-{sub_id}")

        func_file = os.path.join(
            CLEANED_DIR,
            f"sub-{sub_id}_task-belief_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        )
        beh_file = os.path.join(
            CLEANED_DIR,
            f"sub-{sub_id}_task-belief_run-01_events.tsv"
        )
        confound_file = os.path.join(
            CLEANED_DIR,
            f"sub-{sub_id}_task-belief_run-01_desc-confounds_timeseries.tsv"
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
                    f"sub-{sub_id}_design_matrix_PPI.svg"
                ),
            )
            fmri_glm = glm.first_level.FirstLevelModel(
                verbose=VERBOSE,
                noise_model=NOISE_MODEL,
                minimize_memory=MINIMIZE_MEMORY,
                n_jobs=N_JOBS,
                mask_img=brain_mask,
                smoothing_fwhm=SMOOTHING_FWHM
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
                    output_type="z_score"
                )
                plotting.plot_stat_map(
                    stat_map_img=zmap,
                    threshold=1.96,
                    title=f"sub-{sub_id} — {name}-zmap",
                    output_file=os.path.join(
                        FIRST_LEVEL_DIR,
                        f"sub-{sub_id}_contrast-{name}-zmap.svg"
                    )
                )
            print(f"  sub-{sub_id} done.")
        except Exception as e:
            print(f"  Error processing sub-{sub_id}: {e} — skipping.")
            continue

    print("=" * 70)
    print("First-level PPI analysis completed.")

if __name__ == "__main__":
    Psycho_Physio_Interaction()