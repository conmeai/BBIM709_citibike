# Databricks notebook source
# MAGIC %md
# MAGIC # BBIM709 Assessment 2 - Notebook 03b: Machine learning with scikit-learn (Task 3, serverless-compatible)
# MAGIC Use this notebook when Spark MLlib is not available (serverless compute / Free Edition).
# MAGIC Spark does the heavy data preparation (Delta tables, aggregation); the models are trained with scikit-learn on a pandas copy.
# MAGIC Same targets, features and chronological split as `03_ml_classification_prediction.py`.
# MAGIC * **Classification**: casual (1) vs member (0). * **Prediction**: hourly trip demand.
# MAGIC Validation = September, test = October-December (never shuffled).

# COMMAND ----------
import time, json
import numpy as np
import pandas as pd
from pyspark.sql import functions as F
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score)

try:
    dbutils  # exists only on Databricks
    ON_DATABRICKS = True
except NameError:
    ON_DATABRICKS = False

if ON_DATABRICKS:
    CATALOG, SCHEMA = "workspace", "default"                 # <- change if your catalog/schema differ
    VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/citibike"         # volume that holds /raw/<csv file>
    PREFIX = f"{CATALOG}.{SCHEMA}."
else:                                                         # local validation run only
    VOLUME, PREFIX = "/tmp/citibike", ""

T_BRONZE, T_SILVER, T_GOLD = (PREFIX + n for n in ("citibike_bronze", "citibike_silver", "citibike_gold_hourly"))

def save_table(df, name, partition_by=None):
    """Databricks: managed Delta table in Unity Catalog. Local test: parquet folder."""
    w = df.write.mode("overwrite")
    if partition_by:
        w = w.partitionBy(partition_by)
    if ON_DATABRICKS:
        w.format("delta").option("overwriteSchema", "true").saveAsTable(name)
    else:
        w.parquet(f"/tmp/citibike/{name}")

def load_table(name):
    return spark.table(name) if ON_DATABRICKS else spark.read.parquet(f"/tmp/citibike/{name}")
SEED = 42
results = {"engine": "scikit-learn"}

# COMMAND ----------
# MAGIC %md ## A. Classification: casual vs member

# COMMAND ----------
# MAGIC %md ### A1. Data preparation (Spark -> pandas, features, chronological split)

# COMMAND ----------
cols = ["label_casual", "trip_minutes", "distance_km", "hour", "dow", "is_weekend", "month", "rideable_type",
        "start_station_id", "end_station_id", "start_lat", "start_lng"]
pdf = load_table(T_SILVER).select(*cols).toPandas()
pdf["log_minutes"] = np.log1p(pdf["trip_minutes"])
pdf["hour_sin"] = np.sin(2 * np.pi * pdf["hour"] / 24)
pdf["hour_cos"] = np.cos(2 * np.pi * pdf["hour"] / 24)
pdf["is_electric"] = (pdf["rideable_type"] == "electric_bike").astype(int)
pdf["round_trip"] = (pdf["start_station_id"] == pdf["end_station_id"]).astype(int)
FEATURES = ["log_minutes", "distance_km", "hour_sin", "hour_cos", "dow", "is_weekend", "month",
            "is_electric", "round_trip", "start_lat", "start_lng"]
X, y, mth = pdf[FEATURES].astype("float64"), pdf["label_casual"].astype(int), pdf["month"]
tr_val, val, tr_full, te = (mth <= 8), (mth == 9), (mth <= 9), (mth >= 10)
results["clf_counts"] = {"train": int(tr_full.sum()), "validation": int(val.sum()), "test": int(te.sum()),
                         "casual_share_train": round(float(y[tr_full].mean()), 4), "casual_share_test": round(float(y[te].mean()), 4)}
print(results["clf_counts"])

# COMMAND ----------
# MAGIC %md ### A2. Model selection and hyper-parameter tuning (validated on September)

# COMMAND ----------
def lr_model(): return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, max_iter=300))
def rf_model(depth): return RandomForestClassifier(n_estimators=30, max_depth=depth, class_weight="balanced", n_jobs=-1, random_state=SEED)

tuning, t0 = [], time.time()
m = lr_model().fit(X[tr_val], y[tr_val])
tuning.append({"model": "LogisticRegression", "params": "C=1.0", "val_auc_pr": round(average_precision_score(y[val], m.predict_proba(X[val])[:, 1]), 4)})
for depth in (4, 8):
    m = rf_model(depth).fit(X[tr_val], y[tr_val])
    tuning.append({"model": "RandomForest", "params": f"numTrees=30, maxDepth={depth}", "val_auc_pr": round(average_precision_score(y[val], m.predict_proba(X[val])[:, 1]), 4)})
results["clf_tuning"], results["clf_tuning_seconds"] = tuning, round(time.time() - t0, 1)
for t in tuning: print(t)
best_depth = 8 if tuning[2]["val_auc_pr"] >= tuning[1]["val_auc_pr"] else 4
results["clf_best_depth"] = best_depth

# COMMAND ----------
# MAGIC %md ### A3. Final fit (Jan-Sep) and evaluation on the future hold-out (Oct-Dec)

# COMMAND ----------
def clf_report(name, model, sec):
    p = model.predict_proba(X[te])[:, 1]; yp = (p >= 0.5).astype(int); yt = y[te]
    cm = confusion_matrix(yt, yp)
    return {"model": name, "fit_seconds": round(sec, 1), "auc_roc": round(roc_auc_score(yt, p), 4), "auc_pr": round(average_precision_score(yt, p), 4),
            "accuracy": round(accuracy_score(yt, yp), 4), "weighted_f1": round(f1_score(yt, yp, average="weighted"), 4),
            "casual_precision": round(precision_score(yt, yp), 4), "casual_recall": round(recall_score(yt, yp), 4), "casual_f1": round(f1_score(yt, yp), 4),
            "confusion": {"0_0": int(cm[0, 0]), "0_1": int(cm[0, 1]), "1_0": int(cm[1, 0]), "1_1": int(cm[1, 1])}}

t0 = time.time(); lr_final = lr_model().fit(X[tr_full], y[tr_full]); r_lr = clf_report("Logistic Regression", lr_final, time.time() - t0)
t0 = time.time(); rf_final = rf_model(best_depth).fit(X[tr_full], y[tr_full]); r_rf = clf_report("Random Forest", rf_final, time.time() - t0)
results["clf_models"] = [r_lr, r_rf]
results["clf_baseline_accuracy"] = round(1 - results["clf_counts"]["casual_share_test"], 4)
results["clf_importance"] = sorted([{"feature": f, "importance": round(float(i), 4)} for f, i in zip(FEATURES, rf_final.feature_importances_)], key=lambda d: -d["importance"])
for r in (r_lr, r_rf): print(r)
print(results["clf_importance"])

# COMMAND ----------
# MAGIC %md ## B. Prediction: next-hour trip demand (time-series regression)

# COMMAND ----------
# MAGIC %md ### B1. Data preparation: lag and calendar features on the gold hourly table

# COMMAND ----------
g = load_table(T_GOLD).select("hour_ts", "trips").toPandas().sort_values("hour_ts").reset_index(drop=True)
g["hour_ts"] = pd.to_datetime(g["hour_ts"])
g["hour"] = g["hour_ts"].dt.hour
g["dow"] = (g["hour_ts"].dt.dayofweek + 1) % 7 + 1          # Spark convention: 1 = Sunday
g["month"] = g["hour_ts"].dt.month
g["is_weekend"] = g["dow"].isin([1, 7]).astype(int)
g["lag_1"], g["lag_24"], g["lag_168"] = g["trips"].shift(1), g["trips"].shift(24), g["trips"].shift(168)
g["roll_24"] = g["trips"].shift(1).rolling(24, min_periods=1).mean()
g = g.dropna().reset_index(drop=True)
TSF = ["hour", "dow", "month", "is_weekend", "lag_1", "lag_24", "lag_168", "roll_24"]
tv, vv, tf, tt = (g.month <= 8), (g.month == 9), (g.month <= 9), (g.month >= 10)
results["ts_counts"] = {"train": int(tf.sum()), "validation": int(vv.sum()), "test": int(tt.sum())}
print(results["ts_counts"])

# COMMAND ----------
# MAGIC %md ### B2. Tuning (validated on September) and final evaluation (Oct-Dec)

# COMMAND ----------
rmse = lambda a, b: float(np.sqrt(mean_squared_error(a, b)))
grid, best = [], None
for depth in (3, 5):
    for iters in (30, 80):
        m = GradientBoostingRegressor(max_depth=depth, n_estimators=iters, learning_rate=0.1, random_state=SEED).fit(g.loc[tv, TSF], g.loc[tv, "trips"])
        score = rmse(g.loc[vv, "trips"], m.predict(g.loc[vv, TSF]))
        grid.append({"maxDepth": depth, "maxIter": iters, "val_rmse": round(score, 2)})
        if best is None or score < best[0]: best = (score, depth, iters)
results["ts_tuning"], results["ts_best"] = grid, {"maxDepth": best[1], "maxIter": best[2]}
for r in grid: print(r)

def ts_report(name, pred, sec):
    yt = g.loc[tt, "trips"]
    return {"model": name, "fit_seconds": round(sec, 1), "rmse": round(rmse(yt, pred), 2), "mae": round(float(mean_absolute_error(yt, pred)), 2), "r2": round(float(r2_score(yt, pred)), 4)}

Xtr, ytr, Xte = g.loc[tf, TSF], g.loc[tf, "trips"], g.loc[tt, TSF]
r_naive = ts_report("Seasonal naive (lag 24)", g.loc[tt, "lag_24"], 0.0)
t0 = time.time(); lin = Ridge(alpha=0.1).fit(Xtr, ytr); r_lin = ts_report("Linear Regression", lin.predict(Xte), time.time() - t0)
t0 = time.time(); rfr = RandomForestRegressor(n_estimators=50, max_depth=8, n_jobs=-1, random_state=SEED).fit(Xtr, ytr); r_rfr = ts_report("Random Forest Regressor", rfr.predict(Xte), time.time() - t0)
t0 = time.time(); gbt = GradientBoostingRegressor(max_depth=best[1], n_estimators=best[2], learning_rate=0.1, random_state=SEED).fit(Xtr, ytr)
pred_gbt = gbt.predict(Xte); r_gbt = ts_report("Gradient-Boosted Trees", pred_gbt, time.time() - t0)
results["ts_models"] = [r_naive, r_lin, r_rfr, r_gbt]
for r in results["ts_models"]: print(r)
results["ts_importance"] = sorted([{"feature": f, "importance": round(float(i), 4)} for f, i in zip(TSF, gbt.feature_importances_)], key=lambda d: -d["importance"])
te_df = g.loc[tt].assign(pred=pred_gbt)
results["ts_sample"] = [{"t": str(r.hour_ts), "actual": float(r.trips), "pred": round(float(r.pred), 1)} for r in te_df.head(24 * 14).itertuples()]
byh = te_df.assign(err=(te_df.trips - te_df.pred).abs()).groupby("hour")["err"].mean()
results["ts_mae_by_hour"] = [{"hour": int(h), "mae": round(float(v), 1)} for h, v in byh.items()]
print("RESULTS_JSON:" + json.dumps(results))
