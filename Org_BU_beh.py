import os

import numpy as np
import pandas as pd

INPUT_DIR = r"D:/Research/MS_Thesis/Study2/Data/BEH/BU"
OUTPUT_DIR = r"D:/Research/MS_Thesis/Study2/Output/Behavior"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILTER_LIST = {"P04_BURecall.csv", "P05_BURecall.csv"}

e1_data = pd.DataFrame()
e2_data = pd.DataFrame()
recall_data = pd.DataFrame()
accept_data = pd.DataFrame()

for file_name in os.listdir(INPUT_DIR):
    if (not file_name.endswith(".csv")
            or "Loop" in file_name
            or "Prac" in file_name
            or file_name in FILTER_LIST):
        continue

    file_path = os.path.join(INPUT_DIR, file_name)

    if "BUE1" in file_name:
        bue1 = pd.read_csv(file_path)
        bue1 = bue1.rename(columns={
            "EstimScale.response": "Estim1",
            "ConfiResp": "Confi1",
            "SubID(P/C+Num": "SubID",
            "ImagePath": "EventID",
            "EstimResponse": "RecordingResp1",
        })
        e1_data = pd.concat(
            [e1_data, bue1[["BaseP", "Estim1", "RecordingResp1", "Confi1", "SubID", "EventID"]]],
            ignore_index=True,
        )

    elif "BUE2" in file_name:
        bue2 = pd.read_csv(file_path)
        bue2 = bue2.rename(columns={
            "EstimScale.response": "Estim2",
            "ConfiResp": "Confi2",
            "SubID(P/C+Num": "SubID",
            "ImagePath": "EventID",
            "EstimResponse": "RecordingResp2",
        })
        e2_data = pd.concat(
            [e2_data, bue2[["Estim2", "RecordingResp2", "Confi2", "SubID", "EventID"]]],
            ignore_index=True,
        )

    elif "Recall" in file_name:
        bu_recall = pd.read_csv(file_path)
        bu_recall = bu_recall.rename(columns={
            "RecallScale.response": "Recall",
            "ConfiResp": "RecallConfi",
            "SubID(P/C+Num": "SubID",
            "ImagePath": "EventID",
        })
        recall_data = pd.concat(
            [recall_data, bu_recall[["Recall", "RecallConfi", "SubID", "EventID"]]],
            ignore_index=True,
        )

    elif "Accept" in file_name:
        bu_accept = pd.read_csv(file_path)
        bu_accept = bu_accept.rename(columns={
            "AcceptResp": "Accept",
            "SubID(P/C+Num": "SubID",
            "ImagePath": "EventID",
        })
        accept_data = pd.concat(
            [accept_data, bu_accept[["Accept", "SubID", "EventID"]]],
            ignore_index=True,
        )
for df in [e1_data, e2_data, recall_data, accept_data]:
    df["SubID"] = df["SubID"].str[-2:]

merged_data = (
    e1_data
    .merge(e2_data, on=["SubID", "EventID"])
    .merge(recall_data, on=["SubID", "EventID"])
    .merge(accept_data, on=["SubID", "EventID"])
)

merged_data["Valence"] = np.where(
    merged_data["Estim1"] > merged_data["BaseP"], 0.5, -0.5
)
merged_data["Update"] = np.where(
    merged_data["Estim1"] > merged_data["BaseP"],
    merged_data["Estim1"] - merged_data["Estim2"],
    merged_data["Estim2"] - merged_data["Estim1"],
)
merged_data["EE"] = (merged_data["Estim1"] - merged_data["BaseP"]).abs()
merged_data["RecallE"] = (merged_data["Recall"] - merged_data["BaseP"]).abs()

numeric_cols = merged_data.select_dtypes(include="number").columns
merged_data[numeric_cols] = merged_data[numeric_cols].round(2)
merged_data["Confi1"] -= 1
merged_data["Confi2"] -= 1
merged_data["Confi_delta"] = merged_data["Confi2"] - merged_data["Confi1"]
merged_data["RecallConfi"] -= 1

merged_data = merged_data.dropna(subset=["EventID"])
merged_data["EventID"] = merged_data["EventID"].str.extract(r"(\d{2})\.png").astype(float)
merged_data = merged_data.sort_values(["SubID", "EventID"]).reset_index(drop=True)

output_path = os.path.join(OUTPUT_DIR, "BU_behavioral_merged.csv")
merged_data.to_csv(output_path, index=False)