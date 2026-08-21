import pingouin
import pandas as pd

from Clean_BU_beh import bu_beh_cleaned
from Clean_Ques import ques_data

bu_beh = pd.merge(
    bu_beh_cleaned,
    ques_data,
    on="SubID",
    how="left",
)

bu_beh = bu_beh.dropna(subset=["Group", "Valence", "Update", "Confi_delta"])

print("===== Group * Valence --> Update =====")
print("== 2 * 2 mixed ANOVA ==")
pingouin.print_table(pingouin.mixed_anova(
    data=bu_beh,
    dv="Update",
    within="Valence",
    subject="SubID",
    between="Group",
    correction="auto",
    effsize="np2",
))
print("== post hoc ==")
pingouin.print_table(pingouin.pairwise_tests(
    data=bu_beh,
    dv="Update",
    within="Valence",
    subject="SubID",
    between="Group",
    return_desc=True,
))

print("===== Group * Valence --> ConfiChange =====")
print("== 2 * 2 mixed ANOVA ==")
pingouin.print_table(pingouin.mixed_anova(
    data=bu_beh,
    dv="Confi_delta",
    within="Valence",
    subject="SubID",
    between="Group",
    correction="auto"
))
print("== post hoc ==")
pingouin.print_table(pingouin.pairwise_tests(
    data=bu_beh,
    dv="Confi_delta",
    within="Valence",
    subject="SubID",
    between="Group",
    within_first=False,
    return_desc=True,
))