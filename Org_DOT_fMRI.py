import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from nilearn import image

BEH_INPUT_DIR = r"D:/Research/MS_Thesis/Study2/Data/BEH/DOT"
PREPROC_PATH = r"D:/Research/MS_Thesis/Study2/Output/Image/preproc"
OUTPUT_DIR = r"D:/Research/MS_Thesis/Study2/Output/Image/cleaned"

MAX_WORKERS = 4
MFD_THRESHOLD = 0.55

FILE_NAMES = {
    "fmri": "sub-{subid:02d}/func/sub-{subid:02d}_task-dot_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "beh": "perceptData{subid:02d}.csv",
    "confounds": "sub-{subid:02d}/func/sub-{subid:02d}_task-dot_run-{run:02d}_desc-confounds_timeseries.tsv",
    "mask": "sub-{subid:02d}/func/sub-{subid:02d}_task-dot_run-{run:02d}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
    "out_fmri": "sub-{subid:02d}_task-dot_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "out_beh": "sub-{subid:02d}_task-dot_run-{run:02d}_events.tsv",
    "out_conf": "sub-{subid:02d}_task-dot_run-{run:02d}_desc-confounds_timeseries.tsv",
    "out_mask": "sub-{subid:02d}_task-dot_run-{run:02d}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
}


class RunExcludedError(Exception):
    """Raised when a run is excluded due to excessive head motion (mFD > threshold)."""
    def __init__(self, subid: int, run: int, mfd: float):
        self.subid = subid
        self.run = run
        self.mfd = mfd
        super().__init__(
            f"sub-{subid:02d} run-{run:02d} excluded: mFD={mfd:.4f}mm > {MFD_THRESHOLD}mm"
        )


def compute_mfd(conf_df: pd.DataFrame) -> float:
    """Return mean framewise displacement, ignoring the first-volume NaN."""
    if "framewise_displacement" not in conf_df.columns:
        raise ValueError("'framewise_displacement' column not found in confounds file.")
    return conf_df["framewise_displacement"].dropna().mean()


def correct_time_point(beh_data: pd.DataFrame):
    df = beh_data.copy()
    for start_idx, end_idx in [(0, 49), (50, len(df) - 1)]:
        base_time = df.loc[start_idx, "TrialStart"]
        df.loc[start_idx:end_idx, ["EndTime", "TrialStart"]] -= base_time
    df["ConfTime"] += df["TrialStart"]
    df["RespTime"] += df["TrialStart"]
    df["RespStart"] = df["RespTime"] - df["RespRT"]
    df["ConfStart"] = df["ConfTime"] - df["ConfRT"]
    df = df.dropna(subset=["Resp", "Correct", "ConfRating"])
    return df.loc[:49].copy(), df.loc[50:].copy()


def align_beh_img_conf(fmri_data, beh_data, conf_data, mask_data):
    n_vol = fmri_data.shape[3]
    max_tr = math.ceil(beh_data["EndTime"].max())
    trunc_pt = min(n_vol, max_tr)
    fmri_aligned = fmri_data.slicer[..., :trunc_pt]
    beh_aligned = beh_data[beh_data["EndTime"] <= trunc_pt].reset_index(drop=True)
    conf_aligned = conf_data.iloc[:trunc_pt].reset_index(drop=True)
    mask_aligned = mask_data.slicer[..., :trunc_pt]
    return fmri_aligned, beh_aligned, conf_aligned, mask_aligned


def process_single_run(
    subid: int, run: int, beh_run: pd.DataFrame, preproc_path: str, output_path: str
) -> float:
    """
    Process one run. Returns the mFD for this run.
    Raises RunExcludedError if mFD exceeds the threshold (no files are written).
    """
    fmri_path = os.path.join(preproc_path, FILE_NAMES["fmri"].format(subid=subid, run=run))
    conf_path = os.path.join(preproc_path, FILE_NAMES["confounds"].format(subid=subid, run=run))
    mask_path = os.path.join(preproc_path, FILE_NAMES["mask"].format(subid=subid, run=run))

    conf_df = pd.read_csv(conf_path, sep="\t")

    # ── mFD check: bail out before loading the heavy NIfTI files ──────────────
    mfd = compute_mfd(conf_df)
    if mfd > MFD_THRESHOLD:
        raise RunExcludedError(subid, run, mfd)

    fmri_img = image.load_img(fmri_path)
    mask_img = image.load_img(mask_path)

    fmri_aligned, beh_aligned, conf_aligned, mask_aligned = align_beh_img_conf(
        fmri_img, beh_run, conf_df, mask_img
    )

    # Reference: https://neurostars.org/t/parametric-modulation-in-nistats/3392/2
    beh_aligned = pd.melt(
        beh_aligned,
        id_vars=["Correct", "ConfRating", "ConfRT", "RespRT"],
        value_vars=["RespStart", "ConfStart"],
        var_name="trial_type",
        value_name="onset",
    )
    conditions = [
        (beh_aligned["trial_type"] == "RespStart") & (beh_aligned["Correct"] == 1),
        (beh_aligned["trial_type"] == "RespStart") & (beh_aligned["Correct"] == 0),
        (beh_aligned["trial_type"] == "ConfStart") & (beh_aligned["Correct"] == 1),
        (beh_aligned["trial_type"] == "ConfStart") & (beh_aligned["Correct"] == 0),
    ]
    choices = ["resp_correct", "resp_incorrect", "confi_correct", "confi_incorrect"]
    beh_aligned["trial_type"] = np.select(conditions, choices, default="NA")
    beh_aligned = beh_aligned.sort_values(by="onset")

    fmri_aligned.to_filename(
        os.path.join(output_path, FILE_NAMES["out_fmri"].format(subid=subid, run=run))
    )
    beh_aligned.to_csv(
        os.path.join(output_path, FILE_NAMES["out_beh"].format(subid=subid, run=run)),
        sep="\t",
        index=False,
    )
    conf_aligned.to_csv(
        os.path.join(output_path, FILE_NAMES["out_conf"].format(subid=subid, run=run)),
        sep="\t",
        index=False,
    )
    mask_aligned.to_filename(
        os.path.join(output_path, FILE_NAMES["out_mask"].format(subid=subid, run=run))
    )

    return mfd


def main(subid: int, output_path: str) -> dict:
    """
    Returns a result dict:
        status  : "OK" | "skipped" | "excluded" | "ERROR"
        details : human-readable string
        mfd     : {run: mfd_value} for runs that were checked
    """
    beh_path = os.path.join(BEH_INPUT_DIR, FILE_NAMES["beh"].format(subid=subid))
    required = (
        [os.path.join(PREPROC_PATH, FILE_NAMES["fmri"].format(subid=subid, run=r)) for r in (1, 2)]
        + [os.path.join(PREPROC_PATH, FILE_NAMES["confounds"].format(subid=subid, run=r)) for r in (1, 2)]
        + [os.path.join(PREPROC_PATH, FILE_NAMES["mask"].format(subid=subid, run=r)) for r in (1, 2)]
        + [beh_path]
    )
    if not all(os.path.exists(p) for p in required):
        return {"status": "skipped", "details": f"sub-{subid:02d}: skipped (missing files)", "mfd": {}}

    try:
        beh_data = pd.read_csv(beh_path)
        beh_run1, beh_run2 = correct_time_point(beh_data)

        # BUG FIX: compute grand mean once before any subtraction
        # (original code re-used already-modified beh_run1 in the second mean call)
        grand_mean = np.nanmean(
            np.concatenate([beh_run1["ConfRating"].values, beh_run2["ConfRating"].values])
        )
        if not np.isnan(grand_mean):
            beh_run1["ConfRating"] -= grand_mean
            beh_run2["ConfRating"] -= grand_mean

        mfd_by_run = {}
        excluded_runs = []

        for run, beh_run in [(1, beh_run1), (2, beh_run2)]:
            try:
                mfd = process_single_run(subid, run, beh_run, PREPROC_PATH, output_path)
                mfd_by_run[run] = mfd
            except RunExcludedError as exc:
                mfd_by_run[run] = exc.mfd
                excluded_runs.append(run)

        if excluded_runs and len(excluded_runs) < 2:
            # At least one run processed, at least one excluded
            exc_str = ", ".join(f"run-{r:02d}" for r in excluded_runs)
            return {
                "status": "excluded",
                "details": f"sub-{subid:02d}: OK (partial) — {exc_str} excluded (mFD > {MFD_THRESHOLD}mm)",
                "mfd": mfd_by_run,
            }
        elif len(excluded_runs) == 2:
            return {
                "status": "excluded",
                "details": f"sub-{subid:02d}: all runs excluded (mFD > {MFD_THRESHOLD}mm)",
                "mfd": mfd_by_run,
            }
        else:
            return {
                "status": "OK",
                "details": f"sub-{subid:02d}: OK",
                "mfd": mfd_by_run,
            }

    except Exception:
        return {
            "status": "ERROR",
            "details": f"sub-{subid:02d}: ERROR\n{traceback.format_exc()}",
            "mfd": {},
        }


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subject_ids = list(range(1, 78))
    n = len(subject_ids)

    print(f"Processing {n} subjects with {MAX_WORKERS} workers (mFD threshold: {MFD_THRESHOLD}mm) ...\n")

    t_start = time.time()
    all_results = {}
    completed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_sub = {executor.submit(main, sid, OUTPUT_DIR): sid for sid in subject_ids}

        for future in as_completed(future_to_sub):
            sid = future_to_sub[future]
            completed += 1
            elapsed = time.time() - t_start
            remaining = elapsed / completed * (n - completed)

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "ERROR",
                    "details": f"sub-{sid:02d}: ERROR -- {exc}",
                    "mfd": {},
                }

            all_results[sid] = result
            print(
                f"  [{completed:>2}/{n}]  {result['details']:<55}  "
                f"{elapsed:>6.1f}s elapsed  ~{remaining:>5.1f}s left",
                flush=True,
            )

    total = time.time() - t_start
    ok       = sum(1 for r in all_results.values() if r["status"] == "OK")
    excluded = sum(1 for r in all_results.values() if r["status"] == "excluded")
    skipped  = sum(1 for r in all_results.values() if r["status"] == "skipped")
    errors   = sum(1 for r in all_results.values() if r["status"] == "ERROR")

    print(f"\nFinished in {total:.1f}s — {ok} OK, {excluded} excluded, {skipped} skipped, {errors} errors")


    mfd_records = []
    for sid, result in sorted(all_results.items()):
        for run, mfd_val in result["mfd"].items():
            mfd_records.append({
                "sub_id":   f"sub-{sid:02d}",
                "run":      f"run-{run:02d}",
                "mFD":      round(mfd_val, 4),
                "excluded": mfd_val > MFD_THRESHOLD,
            })

    if mfd_records:
        mfd_df = pd.DataFrame(mfd_records)
        print(f"\n  {'Subject':<12} {'Run':<10} {'mFD (mm)':<12} {'Status'}")
        print(f"  {'-'*46}")
        for _, row in mfd_df.iterrows():
            status = "EXCLUDED" if row["excluded"] else "ok"
            print(f"  {row['sub_id']:<12} {row['run']:<10} {row['mFD']:<12} {status}")

    if errors:
        print("\nErrors:")
        for r in all_results.values():
            if r["status"] == "ERROR":
                print(r["details"])