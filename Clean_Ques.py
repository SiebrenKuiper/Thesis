import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import pingouin

ques_data = pd.read_csv(
    "D:/Research/MS_Thesis/Study2/Output/Questionnaire/Quest_notcleaned.csv"
)

fig_output = "D:/Research/MS_Thesis/Study2/Output/Figures"
if not os.path.exists(fig_output):
    os.makedirs(fig_output)

scores = pd.DataFrame({
    "MCQ": ques_data[[f"MCQ_{i}" for i in range(1, 31)]].sum(axis=1),
    "PSC": ques_data[[f"PSC_{i}" for i in range(1, 14)]].sum(axis=1),
    "BDI": ques_data[[f"BDI_{i}" for i in range(1, 21)]].sum(axis=1),
    "BAI": ques_data[[f"BAI_{i}" for i in range(1, 23)]].sum(axis=1),
})
ques_data = pd.concat([ques_data, scores], axis=1)
ques_data["SubID"] = ques_data["SubID"].astype(int)


# 需要手动删除的被试
# 1号被试: 测试人员
# 22号被试: 头动过度
# 26号被试: E1 扫了两遍
# 28号被试: run1缺失
# 67号被试:余祥平
# 56号被试:makexin，MRI 崩了
# 39号被试: 错误地理解实验任务，认为E2是recall
ques_data = ques_data[~ques_data["SubID"].isin([1, 22, 26, 28, 39, 67, 56])]

pain_data = ques_data[["SubID", "PainT0", "PainT1", "PainT2", "PainT3", "PainT4", "PainT5", "PainT6", "Group"]]

long_pain_data = pd.melt(
    pain_data,
    id_vars=["SubID", "Group"],
    value_vars=["PainT0", "PainT1", "PainT2", "PainT3", "PainT4", "PainT5", "PainT6"],
    var_name="Time",
    value_name="PainRating",
)

plt.style.use("default")
plt.figure(figsize=(10, 6))
ax = sns.pointplot(
    data=long_pain_data,
    x="Time",
    y="PainRating",
    hue="Group",
    hue_order=["pain", "control"],
    estimator=np.mean,
    errorbar="se",
    dodge=True,
    markers=["o", "s"],
    capsize=0.1,
    palette="Set1",
)
ax.set_title("Group Average Pain Ratings", fontsize=18, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(fig_output, "PainRting_group.svg"))

subjects = long_pain_data["SubID"].unique()
n_subs = len(subjects)
cols = 4
rows = (n_subs + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
axes = axes.flatten()
for i, subid in enumerate(subjects):
    ax = axes[i]
    sub_data = long_pain_data[long_pain_data["SubID"] == subid]
    group = sub_data["Group"].iloc[0]
    color = "red" if group == "pain" else "blue"
    ax.plot(sub_data["Time"], sub_data["PainRating"], color=color, linewidth=4, marker="o")
    ax.set_title(f"SubID: {subid} | Group: {group}", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Pain Over Time - Each Subject Separately", fontsize=16, fontweight="bold", y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig(os.path.join(fig_output, "PainRting_participant.svg"))

print("Mixed ANOVA (within: five time points, between: two groups) for Pain Ratings (before cleaning):")
pingouin.print_table(pingouin.mixed_anova(
    data=long_pain_data,
    dv="PainRating",
    within="Time",
    between="Group",
    subject="SubID",
))
pingouin.print_table(pingouin.pairwise_tests(
    data=long_pain_data,
    dv="PainRating",
    within="Time",
    between="Group",
    subject="SubID",
    padjust="fdr_bh",
    return_desc=True,
))

keep = (((ques_data["Group"] == "pain") & (ques_data[["PainT1", "PainT2", "PainT3", "PainT4", "PainT5"]].mean(axis=1) > 3)) |
       ((ques_data["Group"] == "control") & (ques_data[["PainT1", "PainT2", "PainT3", "PainT4", "PainT5"]].mean(axis=1) <= 3)))
print("excluded subjects:", ques_data.loc[~keep, "SubID"].tolist())
ques_data = ques_data[keep].copy()
ques_data.to_csv("D:/Research/MS_Thesis/Study2/Output/Questionnaire/ques_data.csv", index=False)