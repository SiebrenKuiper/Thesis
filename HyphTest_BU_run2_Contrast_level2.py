import glob
import os
import re
import traceback
import numpy as np
import pandas as pd
from nilearn import glm, image

ques_data = pd.read_csv("D:/Research/MS_Thesis/Study2/Output/Questionnaire/ques_data.csv")
IMG_DIR = r"D:/Research/MS_Thesis/Study2/Output/Image"
# GROUP_MASK = image.load_img(os.path.join(IMG_DIR, "masks/group_mask_belief_run01.nii"))
FIRST_LEVEL_DIR = os.path.join(IMG_DIR, "contrast/BU_run2_level1")
SECOND_LEVEL_DIR = os.path.join(IMG_DIR, "contrast/BU_run2_level2")
os.makedirs(SECOND_LEVEL_DIR, exist_ok=True)

SECOND_LEVEL_CONTRASTS = [
    "event2_desirable",
    "event2_undesirable",
    "confi2_desirable",
    "confi2_undesirable",
    "event2_desirable-undesirable",
]


def load_first_level_maps(contrast_name: str, group_map: dict[str, str]) -> pd.DataFrame:
    """
    Scan FIRST_LEVEL_DIR for z-maps of *contrast_name*.

    Returns a DataFrame with columns:
        SubID, group, Group (0/1 int), img (Nifti1Image)
    Only subjects with group labels "pain" or "control" are included.
    """
    pattern = os.path.join(
        FIRST_LEVEL_DIR,
        f"sub-*_contrast-{contrast_name}-cmap.nii.gz",
    )
    cmap_files = sorted(glob.glob(pattern))

    if not cmap_files:
        print(f"  [WARN] No maps found for contrast \"{contrast_name}\" — skipping.")
        return pd.DataFrame()

    subject_re = re.compile(
        rf"sub-(\d+)_contrast-{re.escape(contrast_name)}-cmap\.nii\.gz"
    )

    rows = []
    for path in cmap_files:
        match = subject_re.search(os.path.basename(path))
        if not match:
            print(f"  [WARN] Filename does not match expected pattern — skipping: {path}")
            continue

        sub_id = match.group(1)
        group = group_map.get(sub_id)
        if group not in ("pain", "control"):
            print(f"  [WARN] sub-{sub_id}: unknown group {group!r} — skipping.")
            continue

        try:
            img = image.load_img(path)
        except Exception:
            print(f"  [ERROR] Failed to load {path} — skipping.")
            traceback.print_exc()
            continue

        rows.append({
            "SubID": sub_id,
            "group": group,
            "Group": 1 if group == "pain" else 0,
            "img": img,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"  Loaded {len(df)} subjects  "
              f"(pain={int((df['Group'] == 1).sum())}, "
              f"control={int((df['Group'] == 0).sum())})")
    return df


def run_second_level(contrast_name: str, sample_df: pd.DataFrame) -> None:
    """
    Fit a second-level GLM for *contrast_name* and save:
      - Intercept z-map + HTML report  (whole-brain mean activation)
      - Group contrast z-map + HTML report  (pain > control)
    Outputs go into a dedicated subdirectory:
        SECOND_LEVEL_DIR/<contrast_name>/
    """
    out_dir = os.path.join(SECOND_LEVEL_DIR, contrast_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"  Fitting second-level GLM (n={len(sample_df)})...")

    design_matrix = pd.DataFrame({
        "Intercept": np.ones(len(sample_df), dtype=float),
        "Group": sample_df["Group"].astype(int).to_numpy(),
    })

    second_level_model = glm.second_level.SecondLevelModel(
        mask_img=False,
        n_jobs=-1,
        verbose=1,
        minimize_memory=True,
    )
    second_level_model.fit(
        second_level_input=sample_df["img"].tolist(),
        design_matrix=design_matrix,
    )

    intercept_zmap = second_level_model.compute_contrast(
        second_level_contrast="Intercept",
        output_type="z_score",
    )
    intercept_zmap_path = os.path.join(out_dir, "intercept_whole_brain_zmap.nii.gz")
    intercept_zmap.to_filename(intercept_zmap_path)
    print(f"  Intercept z-map saved: {intercept_zmap_path}")

    intercept_report = second_level_model.generate_report(
        contrasts="Intercept",
        alpha=0.05,
        cluster_threshold=10,
        height_control="fdr",
        two_sided=True,
        min_distance=8,
        plot_type="slice",
    )
    intercept_report.save_as_html(os.path.join(out_dir, "intercept_whole_brain.html"))

    group_zmap = second_level_model.compute_contrast(
        second_level_contrast="Group",
        output_type="z_score",
    )
    group_zmap_path = os.path.join(out_dir, "group_contrast_pain_vs_control_zmap.nii.gz")
    group_zmap.to_filename(group_zmap_path)
    print(f"  Group z-map saved:     {group_zmap_path}")

    group_report = second_level_model.generate_report(
        contrasts="Group",
        alpha=0.05,
        cluster_threshold=10,
        height_control="fdr",
        two_sided=True,
        min_distance=8,
        plot_type="slice",
    )
    group_report.save_as_html(
        os.path.join(out_dir, "group_contrast_pain_vs_control.html")
    )


def main() -> None:
    print("**** Starting Second-Level Analysis (Pain vs Control) ****")
    tmp = ques_data[["SubID", "Group"]].copy()
    tmp["SubID"] = tmp["SubID"].apply(lambda x: f"{int(x):02d}")
    group_map = dict(zip(tmp["SubID"], tmp["Group"]))

    for contrast_name in SECOND_LEVEL_CONTRASTS:
        print(f"\n--- Contrast: {contrast_name} ---")
        sample_df = load_first_level_maps(contrast_name, group_map)
        if sample_df.empty:
            continue
        try:
            run_second_level(contrast_name, sample_df)
        except Exception as e:
            print(f"  [ERROR] Failed for contrast \"{contrast_name}\": {e} — skipping.")
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Second-level analysis complete.")


if __name__ == "__main__":
    main()