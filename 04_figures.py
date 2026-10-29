# Databricks notebook source
# MAGIC %md
# MAGIC # BBIM709 Assessment 2 - Notebook 04: Figures for the report
# MAGIC Draws the five charts used in the report, from your own Databricks results:
# MAGIC * **Figure 6** hourly trips and mean duration (from the silver Delta table)
# MAGIC * **Figure 7** ten busiest start stations (from the silver Delta table)
# MAGIC * **Figures 12, 13, 14** confusion matrices, actual vs predicted demand, feature importance (from the output of Notebook 03b)
# MAGIC
# MAGIC Run Notebooks 01 and 03b first. PNG files are also saved in the volume folder `figures/` so they can be downloaded.

# COMMAND ----------
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

try:
    dbutils  # exists only on Databricks
    ON_DATABRICKS = True
except NameError:
    ON_DATABRICKS = False

if ON_DATABRICKS:
    CATALOG, SCHEMA = "workspace", "default"                 # <- change if your catalog/schema differ
    VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/citibike"
    PREFIX = f"{CATALOG}.{SCHEMA}."
    FIG_DIR = f"{VOLUME}/figures"
    RESULTS_DIR = f"{VOLUME}/results"
else:                                                         # local validation run only
    PREFIX = ""
    FIG_DIR = "/tmp/citibike/figures"
    RESULTS_DIR = "/tmp/citibike/results"
T_SILVER = PREFIX + "citibike_silver"

def load_table(name):
    return spark.table(name) if ON_DATABRICKS else spark.read.parquet(f"/tmp/citibike/{name}")

os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#1f4e79", "#c55a11"

# COMMAND ----------
# MAGIC %md ## 1. Figure 6 - trips and mean trip length by hour and rider type (Job A result)

# COMMAND ----------
silver = load_table(T_SILVER)
job_a = (silver.groupBy("hour", "member_casual")
         .agg(F.count("*").alias("trips"), F.avg("trip_minutes").alias("avg_min"))
         .orderBy("hour").toPandas())
trips = job_a.pivot(index="hour", columns="member_casual", values="trips")
dur = job_a.pivot(index="hour", columns="member_casual", values="avg_min")

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].plot(trips.index, trips["member"], color=BLUE, label="Member")
ax[0].plot(trips.index, trips["casual"], color=ORANGE, ls="--", label="Casual")
ax[0].set_xlabel("Hour of day"); ax[0].set_ylabel("Trips (2021)"); ax[0].set_title("Trips by hour", fontsize=10); ax[0].legend(frameon=False)
ax[1].plot(dur.index, dur["member"], color=BLUE)
ax[1].plot(dur.index, dur["casual"], color=ORANGE, ls="--")
ax[1].set_xlabel("Hour of day"); ax[1].set_ylabel("Mean trip length (min)"); ax[1].set_title("Mean duration by hour", fontsize=10)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/figure_06_hourly.png", dpi=200); plt.show()

# COMMAND ----------
# MAGIC %md ## 2. Figure 7 - ten busiest start stations (Job B result)

# COMMAND ----------
top10 = (silver.groupBy("start_station_name").agg(F.count("*").alias("trips"))
         .orderBy(F.desc("trips")).limit(10).toPandas().iloc[::-1])
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.barh(top10["start_station_name"], top10["trips"], color=BLUE)
for i, v in enumerate(top10["trips"]):
    ax.text(v + 50, i, f"{v:,}", va="center", fontsize=8)
ax.set_xlabel("Trips started (2021)"); ax.set_xlim(0, top10["trips"].max() * 1.12)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/figure_07_top_stations.png", dpi=200); plt.show()

# COMMAND ----------
# MAGIC %md ## 3. Model results from Notebook 03b
# MAGIC Notebook 03b saves its results to `results/03b_results.json` in the volume. This cell reads that file, so run Notebook 03b first (re-run it if you ran an older version).

# COMMAND ----------
RES_PATH = f"{RESULTS_DIR}/03b_results.json"
if not os.path.exists(RES_PATH):
    raise FileNotFoundError(f"{RES_PATH} not found. Run Notebook 03b_ml_sklearn first (the latest version saves this file).")
with open(RES_PATH) as fh:
    res = json.load(fh)
print("Models loaded:", [m["model"] for m in res["clf_models"]], "|", [m["model"] for m in res["ts_models"]])

# COMMAND ----------
# MAGIC %md ## 4. Figure 12 - confusion matrices on the test set

# COMMAND ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
for a, mod in zip(axs, res["clf_models"]):
    cm = mod["confusion"]
    M = np.array([[cm["0_0"], cm["0_1"]], [cm["1_0"], cm["1_1"]]])
    P = M / M.sum(axis=1, keepdims=True)
    a.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            a.text(j, i, f"{M[i, j]:,}\n({P[i, j]:.0%})", ha="center", va="center",
                   color="white" if P[i, j] > 0.5 else "black", fontsize=9)
    a.set_xticks([0, 1]); a.set_yticks([0, 1])
    a.set_xticklabels(["Member", "Casual"]); a.set_yticklabels(["Member", "Casual"])
    a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.set_title(mod["model"], fontsize=10)
    for sp in a.spines.values():
        sp.set_visible(False)
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/figure_12_confusion.png", dpi=200); plt.show()

# COMMAND ----------
# MAGIC %md ## 5. Figure 13 - actual vs predicted hourly trips (first 14 days of the test period)

# COMMAND ----------
s = pd.DataFrame(res["ts_sample"]); s["t"] = pd.to_datetime(s["t"])
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot(s["t"], s["actual"], color=BLUE, lw=1.2, label="Actual")
ax.plot(s["t"], s["pred"], color=ORANGE, lw=1.2, ls="--", label="Predicted (GBT)")
ax.set_ylabel("Trips per hour"); ax.set_xlabel("Time (first 14 days of the test period)")
ax.legend(frameon=False, ncol=2, loc="upper right")
fig.autofmt_xdate(); plt.tight_layout(); plt.savefig(f"{FIG_DIR}/figure_13_forecast.png", dpi=200); plt.show()

# COMMAND ----------
# MAGIC %md ## 6. Figure 14 - feature importance

# COMMAND ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
for a, key, title in zip(axs, ["clf_importance", "ts_importance"],
                         ["Random Forest (user type)", "Gradient boosting (hourly demand)"]):
    d = pd.DataFrame(res[key]).iloc[::-1]
    a.barh(d["feature"], d["importance"], color=BLUE); a.set_title(title, fontsize=10); a.set_xlabel("Feature importance")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/figure_14_importance.png", dpi=200); plt.show()
print("Saved:", sorted(os.listdir(FIG_DIR)))
