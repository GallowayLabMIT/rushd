# ruff: noqa
import matplotlib.pyplot as plt
import pandas as pd
import rushd as rd

# Example: Bad formatting for tick labels
data = pd.DataFrame(
    {
        "short_name": ["A", "B", "C", "D"],
        "long_name": ["no inducer", "doxycycline", "guanine", "doxycycline + guanine"],
        "data": [1, 3, 2, 4],
        "doxycycline": ["-", "+", "-", "+"],
        "guanine": ["-", "-", "+", "+"],
    }
)

fig, axes = plt.subplots(1, 2)
axes[1].bar(data.short_name, data.data)
axes[1].set(title="Short Name")
axes[1].annotate(
    text="\n".join(["A: no inducer", "B: doxycycline", "C: guanine", "D: doxycycline + guanine"]),
    xy=(1.1, 1),
    xycoords="axes fraction",
    va="top",
    fontsize=plt.rcParams["xtick.labelsize"],
)
axes[0].bar(data.long_name, data.data)
axes[0].tick_params(axis="x", labelrotation=90)
axes[0].set(title="Full Name")
plt.savefig(
    "../_static/built_output/generate_xticklabels/bad-formatting-examples.svg", bbox_inches="tight"
)


# Example: Desired output
fig, ax = plt.subplots(1, 1, figsize=(3, 4))
ax.bar(data.short_name, data.data)
ax.set(title="Desired Output")
rd.plot.generate_xticklabels(data, "short_name", ["doxycycline", "guanine"])
fig.savefig("../_static/built_output/generate_xticklabels/desired-output.svg", bbox_inches="tight")

# Example: Main
data = pd.DataFrame(
    {
        "condition": ["A", "B", "C", "D"] * 2,
        "cell_type": ["MEF"] * 4 + ["iMN"] * 4,
        "data": [1, 3, 2, 4, 2, 2, 4, 5],
        "doxycycline": ["–", "+", "–", "+"] * 2,
        "guanine": ["–", "–", "+", "+"] * 2,
    }
)
data["name"] = data.cell_type + data.condition
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.bar(data.name, data.data)

rd.plot.generate_xticklabels(data, "name", ["cell_type", "doxycycline", "guanine"])
fig.savefig("../_static/built_output/generate_xticklabels/main-example.svg", bbox_inches="tight")

# Example: Custom alignment
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.bar(data.name, data.data)
rd.plot.generate_xticklabels(
    data,
    "name",
    ["cell_type", "doxycycline", "guanine"],
    align_annotation="center",
    align_ticklabels="left",
)
fig.savefig(
    "../_static/built_output/generate_xticklabels/custom-alignment.svg", bbox_inches="tight"
)


# Example: Compatibility with Seaborn
import seaborn as sns

g = sns.catplot(
    data=data, x="condition", y="data", col="cell_type", kind="bar", height=4, aspect=0.7
)
ax_col = "condition"
label_cols = ["doxycycline", "guanine"]
df_labels = data.drop_duplicates([ax_col] + label_cols)

rd.plot.generate_xticklabels(df_labels, ax_col, label_cols, ax=g.axes_dict["MEF"])
g.fig.savefig("../_static/built_output/generate_xticklabels/seaborn-usage.svg", bbox_inches="tight")
