import glob
import os
import shutil

import numpy as np
import pandas as pd

INPUT_DIR_BEH = r"D:/Research/MS_Thesis/Study2/Data/BEH/BU"
INPUT_DIR_PREPROC = r"D:/Research/MS_Thesis/Study2/Output/Image/preproc"
OUTPUT_DIR = r"D:/Research/MS_Thesis/Study2/Output/Image/cleaned"
if not os.path.exists(INPUT_DIR_BEH):
    os.makedirs(INPUT_DIR_BEH, exist_ok=True)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

ORG_BEH = True
ORG_IMG = False
MFD_THRESHOLD = 0.55  # mm; scans with mFD > this value are excluded

FILTER_LIST = {"P04_BURecall.csv", "P05_BURecall.csv"}
if FILTER_LIST:
    print(f"Filtering out behavioural files: {', '.join(FILTER_LIST)}")

print("\n=== Organising behavioural data ===")
e1_data = pd.DataFrame()
e2_data = pd.DataFrame()

e1_count = 0
e2_count = 0
skipped_count = 0

for file_name in os.listdir(INPUT_DIR_BEH):
    if (not file_name.endswith(".csv")
            or "Loop" in file_name
            or "Prac" in file_name
            or file_name in FILTER_LIST):
        if file_name.endswith(".csv") and ("Loop" in file_name or "Prac" in file_name or file_name in FILTER_LIST):
            print(f"  Skipping file: {file_name}")
            skipped_count += 1
        continue

    file_path = os.path.join(INPUT_DIR_BEH, file_name)

    if "BUE1" in file_name:
        bue1 = pd.read_csv(file_path)
        start_time1 = bue1["Cross1.started"].dropna().iloc[0]
        bue1 = bue1.rename(columns={
            "EstimScale.response": "Estim1",
            "ConfiResp": "Confi1",
            "SubID(P/C+Num": "SubID",
            "ImagePath": "EventID",
            "Events.started": "onset_event1",
            "Estim.started": "onset_estim1",
            "Fedbak.started": "onset_feedback",
            "Confi.started": "onset_confi1",
        })
        onset_cols1 = ["onset_event1", "onset_estim1", "onset_feedback", "onset_confi1"]
        bue1 = bue1[["BaseP", "SubID", "EventID", "Estim1", "Confi1"] + onset_cols1]
        bue1[onset_cols1] -= start_time1
        e1_data = pd.concat([e1_data, bue1], ignore_index=True)
        e1_count += 1
        print(f"  Loaded E1 data from: {file_name} (n={len(bue1)} rows)")

    elif "BUE2" in file_name:
        bue2 = pd.read_csv(file_path)
        start_time2 = bue2["Cross1.started"].dropna().iloc[0]
        bue2 = bue2.rename(columns={
            "EstimScale.response": "Estim2",
            "ConfiResp": "Confi2",
            "SubID(P/C+Num": "SubID",
            "ImagePath": "EventID",
            "Events.started": "onset_event2",
            "Estim.started": "onset_estim2",
            "Confi.started": "onset_confi2",
        })
        onset_cols2 = ["onset_event2", "onset_estim2", "onset_confi2"]
        bue2 = bue2[["SubID", "EventID", "Estim2", "Confi2"] + onset_cols2]
        bue2[onset_cols2] -= start_time2
        e2_data = pd.concat([e2_data, bue2], ignore_index=True)
        e2_count += 1
        print(f"  Loaded E2 data from: {file_name} (n={len(bue2)} rows)")

print(f"Processed {e1_count} E1 files and {e2_count} E2 files (skipped {skipped_count} non-matching files).")

for df in [e1_data, e2_data]:
    df["SubID"] = df["SubID"].str[-2:]
print("  Standardised SubID to last two characters.")

merged_data = e1_data.merge(e2_data, on=["SubID", "EventID"])
print(f"  Merged data: {len(merged_data)} rows (unique subject-event pairs).")

# 同时计算 contrast 和 mediation 所需的列
merged_data["EE"] = (merged_data["Estim1"] - merged_data["BaseP"])
merged_data["Valence"] = np.where(
    merged_data["Estim1"] > merged_data["BaseP"],
    "desirable",
    "undesirable",
)

merged_data["Update"] = np.where(
    merged_data["Estim1"] > merged_data["BaseP"],
    merged_data["Estim1"] - merged_data["Estim2"],
    merged_data["Estim2"] - merged_data["Estim1"],
)
print("  Computed 'Update' (belief update).")

pre_filter_len = len(merged_data)
merged_data = merged_data[merged_data["EE"] != 0]
print(f"  Removed {pre_filter_len - len(merged_data)} rows where EE == 0.")
merged_data = merged_data.dropna(subset=["EventID"])
merged_data["EventID"] = merged_data["EventID"].str.extract(r"(\d{2})\.png").astype(float)
print("  Extracted numeric EventID from filenames.")

numeric_cols = merged_data.select_dtypes(include="number").columns
merged_data[numeric_cols] = merged_data[numeric_cols].round(2)
merged_data["Confi1"] -= 1
merged_data["Confi2"] -= 1
print("  Rounded numeric columns and shifted confidence ratings (1-based → 0-based).")

print(f"\n=== Computing mean framewise displacement (threshold: mFD > {MFD_THRESHOLD} mm) ===")
confound_pattern = os.path.join(
    INPUT_DIR_PREPROC,
    "sub-*/func/sub-*_task-belief_run-*_desc-confounds_timeseries.tsv",
)
confound_files = glob.glob(confound_pattern)

if not confound_files:
    print("  WARNING: No confound files found — skipping mFD exclusion.")
    excluded_runs = {}
else:
    mfd_records = []
    for cf in sorted(confound_files):
        basename = os.path.basename(cf)
        sub_match = pd.Series([basename]).str.extract(r"(sub-\d+)")
        run_match = pd.Series([basename]).str.extract(r"(run-\d+)")
        if sub_match.iloc[0, 0] is None or run_match.iloc[0, 0] is None:
            print(f"  WARNING: Could not parse sub/run from {basename} — skipping.")
            continue

        sub_label = sub_match.iloc[0, 0]
        run_label = run_match.iloc[0, 0]
        sub_id    = sub_label.replace("sub-", "")

        confounds_df = pd.read_csv(cf, sep="\t")
        if "framewise_displacement" not in confounds_df.columns:
            print(f"  WARNING: 'framewise_displacement' column missing in {basename} — skipping.")
            continue

        fd_values = confounds_df["framewise_displacement"].dropna()
        mfd = fd_values.mean()

        mfd_records.append({
            "sub_id":    sub_id,
            "run_label": run_label,
            "sub_label": sub_label,
            "mFD":       round(mfd, 4),
            "n_volumes": len(fd_values),
            "excluded":  mfd > MFD_THRESHOLD,
        })

    mfd_df = pd.DataFrame(mfd_records)
    print(f"\n  {'Subject':<12} {'Run':<10} {'mFD (mm)':<12} {'Status'}")
    print(f"  {'-'*44}")
    for _, row in mfd_df.iterrows():
        status = "EXCLUDED" if row["excluded"] else "ok"
        print(f"  {row['sub_label']:<12} {row['run_label']:<10} {row['mFD']:<12} {status}")

    excluded_mask = mfd_df["excluded"]
    n_excluded = excluded_mask.sum()
    print(f"\n  Excluded {n_excluded} / {len(mfd_df)} runs exceeding mFD > {MFD_THRESHOLD} mm.")

    excluded_runs = set()
    for _, row in mfd_df[excluded_mask].iterrows():
        run_num = int(row["run_label"].replace("run-", ""))
        excluded_runs.add((row["sub_id"], run_num))

subjects = merged_data["SubID"].unique()
print(f"\n  Exporting events TSV files for {len(subjects)} subjects...")

for sub_id in subjects:
    sub_id_int = int(sub_id)
    sub_df = merged_data[merged_data["SubID"] == sub_id].copy()

    # ── Run 1 ──────────────────────────────────────────────────────────
    if (sub_id, 1) in excluded_runs:
        print(f"    sub-{sub_id_int:02d} run-01 SKIPPED (mFD exceeded threshold)")
    else:
        id_vars = ["Confi1", "Estim1", "EE", "Update", "Valence"]
        df_run1 = sub_df[id_vars + ["onset_event1", "onset_estim1",
                                    "onset_feedback", "onset_confi1"]].copy()
        df_run1[["Confi1", "Estim1", "EE", "Update"]] = \
            df_run1[["Confi1", "Estim1", "EE", "Update"]].round(2)

        run1 = pd.melt(
            df_run1,
            id_vars=id_vars,
            value_vars=["onset_event1", "onset_estim1", "onset_feedback", "onset_confi1"],
            var_name="_onset_col",
            value_name="onset",
        )
        run1["_base"] = run1["_onset_col"].str.replace("onset_", "", regex=False)
        run1["trial_type_mediation"] = run1["_base"]
        run1["trial_type_contrast"] = run1["_base"]
        mask_fb = run1["_base"] == "feedback"
        run1.loc[mask_fb, "trial_type_contrast"] = "feedback_" + run1.loc[mask_fb, "Valence"]

        run1["duration"] = pd.NA
        run1.loc[run1["_base"] == "event1",    "duration"] = 0
        run1.loc[run1["_base"] == "feedback",  "duration"] = 0
        run1.loc[run1["_base"] == "confi1",    "duration"] = 0

        run1 = run1.sort_values("onset")
        run1 = run1[["trial_type_mediation", "trial_type_contrast",
                        "onset", "duration",
                        "Confi1", "Estim1", "EE", "Update"]]
        run1 = run1.dropna()

        out1 = os.path.join(OUTPUT_DIR,
                            f"sub-{sub_id_int:02d}_task-belief_run-01_events.tsv")
        run1.to_csv(out1, sep="\t", index=False)
        print(f"    Exported sub-{sub_id_int:02d} run-01")

    # ── Run 2 ──────────────────────────────────────────────────────────
    if (sub_id, 2) in excluded_runs:
        print(f"    sub-{sub_id_int:02d} run-02 SKIPPED (mFD exceeded threshold)")
    else:
        id_vars = ["Estim2", "EE", "Confi2", "Update", "Valence"]
        df_run2 = sub_df[id_vars + ["onset_event2", "onset_estim2",
                                    "onset_confi2"]].copy()
        df_run2[["Confi2", "Estim2", "EE", "Update"]] = \
            df_run2[["Confi2", "Estim2", "EE", "Update"]].round(2)

        run2 = pd.melt(
            df_run2,
            id_vars=id_vars,
            value_vars=["onset_event2", "onset_estim2", "onset_confi2"],
            var_name="_onset_col",
            value_name="onset",
        )
        run2["_base"] = run2["_onset_col"].str.replace("onset_", "", regex=False)

        # trial_type_mediation：event2 / estim2 / confi2，不拆分
        run2["trial_type_mediation"] = run2["_base"]

        # trial_type_contrast：event2 和 confi2 拆分 valence
        run2["trial_type_contrast"] = run2["_base"]
        for base_name in ["event2", "confi2"]:
            mask = run2["_base"] == base_name
            run2.loc[mask, "trial_type_contrast"] = base_name + "_" + run2.loc[mask, "Valence"]

        run2["duration"] = pd.NA
        run2.loc[run2["_base"] == "event2", "duration"] = 0
        run2.loc[run2["_base"] == "confi2", "duration"] = 0

        run2 = run2.sort_values("onset")
        run2 = run2[["trial_type_mediation", "trial_type_contrast",
                        "onset", "duration",
                        "Estim2", "EE", "Confi2", "Update"]]
        run2 = run2.dropna()

        out2 = os.path.join(OUTPUT_DIR,
                            f"sub-{sub_id_int:02d}_task-belief_run-02_events.tsv")
        run2.to_csv(out2, sep="\t", index=False)
        print(f"    Exported sub-{sub_id_int:02d} run-02")

print("Behavioural organisation finished.\n")


print("\n=== Copying imaging files from preproc to cleaned ===")
patterns = [
    "sub-*/func/sub-*_task-belief_run-*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    "sub-*/func/sub-*_task-belief_run-*_desc-confounds_timeseries.tsv",
    "sub-*/func/sub-*_task-belief_run-*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
]
total_copied = 0
for pattern in patterns:
    matches = glob.glob(os.path.join(INPUT_DIR_PREPROC, pattern))
    if matches:
        print(f"  Pattern: {pattern} → found {len(matches)} file(s)")
        for f in matches:
            dest = os.path.join(OUTPUT_DIR, os.path.basename(f))
            shutil.copy2(f, dest)
            total_copied += 1
            print(f"    Copied: {os.path.basename(f)}")
    else:
        print(f"  Pattern: {pattern} → no files found")
print(f"Total imaging files copied: {total_copied}\n")

print("All requested operations completed.")