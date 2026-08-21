from Org_BU_beh import merged_data
import matplotlib.pyplot as plt
import os

fig_output = "D:/Research/MS_Thesis/Study2/Output/Figures"
if not os.path.exists(fig_output):
    os.makedirs(fig_output)

bu_beh_cleaned = merged_data.dropna()
bu_beh_cleaned = bu_beh_cleaned[bu_beh_cleaned["EE"] != 0]
bu_beh_cleaned["SubID"] = bu_beh_cleaned["SubID"].astype(int)

bu_beh_cleaned.to_csv(
    "D:/Research/MS_Thesis/Study2/Output/Behavior/Cleaned_UB_beh.csv"
)

subids = sorted(bu_beh_cleaned["SubID"].unique())
n_sub = len(subids)
fig, axes = plt.subplots(n_sub, 2, figsize=(10, 3 * n_sub))
for i, subid in enumerate(subids):
    sub_data = bu_beh_cleaned[bu_beh_cleaned["SubID"] == subid]
    ax_left = axes[i, 0]
    ax_left.plot(sub_data["EventID"], sub_data["Estim1"], marker="o", color="green", label="Estim1")
    ax_left.plot(sub_data["EventID"], sub_data["Estim2"], marker="o", color="orange", label="Estim2")
    ax_left.set_title(f"Subject {subid} - Estimation")
    ax_left.set_ylabel("Estimation Value")
    ax_left.set_ylim(6, 60)
    ax_left.legend()
    ax_left.grid()
    ax_right = axes[i, 1]
    ax_right.plot(sub_data["EventID"], sub_data["Confi1"], linestyle="-.", marker=".", color="green", label="Confi1")
    ax_right.plot(sub_data["EventID"], sub_data["Confi2"], linestyle="-.", marker=".", color="orange", label="Confi2")
    ax_right.set_title(f"Subject {subid} - Confidence")
    ax_right.set_xlabel("EventID")
    ax_right.set_ylabel("Confidence Value")
    ax_right.set_ylim(1, 4)
    ax_right.legend()
    ax_right.grid()
plt.tight_layout()
plt.savefig(os.path.join(fig_output, "Subject_Estim_Confi.svg"))