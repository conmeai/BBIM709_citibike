# Databricks notebook source
# MAGIC %md
# MAGIC # BBIM709 Assessment 2 - Notebook 02b: MapReduce with the DataFrame API (Task 2, serverless-compatible)
# MAGIC Use this notebook when the RDD API is not available (serverless compute, Free Edition, standard access mode).
# MAGIC The MapReduce model is kept explicit: **map** = emit (key, value) rows, **shuffle + reduce** = `groupBy(key).agg(...)`.
# MAGIC Results are identical to `02_mapreduce.py` (RDD version).

# COMMAND ----------
import time, json
from pyspark.sql import functions as F

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
results = {"mode": "dataframe_mapreduce"}

# COMMAND ----------
# MAGIC %md ## 1. Input handling: read Silver and keep only the needed columns

# COMMAND ----------
silver = load_table(T_SILVER).select("start_station_name", "hour", "dow", "member_casual", "trip_minutes")
n = silver.count()
results["input_records"] = n
print("Records:", n)

# COMMAND ----------
# MAGIC %md ## 2. Job A (temporal analysis): trips and mean duration by (hour, user type)
# MAGIC * **Map**: row -> key = (hour, member_casual), value = (trip_minutes, 1)
# MAGIC * **Shuffle + Reduce**: add partial sums and counts per key; mean = sum / count

# COMMAND ----------
t0 = time.time()
map_a = silver.select(F.struct("hour", "member_casual").alias("k"), F.col("trip_minutes").alias("v_sum"), F.lit(1).alias("v_cnt"))
red_a = map_a.groupBy("k").agg(F.sum("v_sum").alias("sum_min"), F.sum("v_cnt").alias("trips"))
rows = (red_a.select(F.col("k.hour").alias("hour"), F.col("k.member_casual").alias("user"), "trips",
                     (F.col("sum_min") / F.col("trips")).alias("avg_min"))
        .orderBy("hour", "user").collect())
t_a = time.time() - t0
results["job_a_seconds_df"] = round(t_a, 2)
results["job_a_keys"] = len(rows)
results["job_a"] = [{"hour": r["hour"], "user": r["user"], "trips": r["trips"], "avg_min": round(r["avg_min"], 2)} for r in rows]
print(f"Job A finished in {t_a:.2f}s, {len(rows)} keys")
for r in rows[:8]:
    print(((r["hour"], r["user"]), (r["trips"], r["avg_min"])))

# COMMAND ----------
# MAGIC %md ## 3. Job B (counting): top-10 busiest start stations

# COMMAND ----------
t0 = time.time()
top10 = (silver.select(F.col("start_station_name").alias("k"), F.lit(1).alias("v"))
         .groupBy("k").agg(F.sum("v").alias("trips")).orderBy(F.desc("trips")).limit(10).collect())
results["job_b_seconds_df"] = round(time.time() - t0, 2)
results["top10_stations"] = [{"station": r["k"], "trips": r["trips"]} for r in top10]
for r in top10:
    print(f"{r['trips']:>7,}  {r['k']}")

# COMMAND ----------
# MAGIC %md ## 4. Job C (anomaly detection): weekday-hour cells with mean duration > 2 standard deviations from the overall mean

# COMMAND ----------
cell = (silver.select(F.struct("dow", "hour").alias("k"), F.col("trip_minutes").alias("m"))
        .groupBy("k").agg(F.sum("m").alias("s"), F.count("*").alias("c")).collect())
cells = [((r["k"]["dow"], r["k"]["hour"]), r["s"] / r["c"], r["c"]) for r in cell]
means = [m for _, m, _ in cells]
mu = sum(means) / len(means)
sd = (sum((m - mu) ** 2 for m in means) / len(means)) ** 0.5
anomalies = sorted([(k, round(m, 1), c) for k, m, c in cells if abs(m - mu) > 2 * sd], key=lambda t: -t[1])
results["anomaly_mu"], results["anomaly_sd"] = round(mu, 2), round(sd, 2)
results["anomalies"] = [{"dow": k[0], "hour": k[1], "avg_min": m, "trips": c} for k, m, c in anomalies]
print(f"cells={len(cells)} mean-of-means={mu:.2f} sd={sd:.2f} anomalies={len(anomalies)}")
for a in anomalies[:10]:
    print(a)

# COMMAND ----------
# MAGIC %md ## 5. Partitioning experiment: how many partitions take part in the shuffle

# COMMAND ----------
exp = []
for p in (2, 8, 32):
    t0 = time.time()
    (silver.repartition(p, "hour", "member_casual")
     .groupBy("hour", "member_casual").agg(F.sum("trip_minutes"), F.count("*")).count())
    exp.append({"repartition": p, "seconds": round(time.time() - t0, 2)})
results["partition_experiment"] = exp
print(exp)

# COMMAND ----------
# MAGIC %md ## 6. Same Job A in one statement (what the optimiser does for you)

# COMMAND ----------
t0 = time.time()
one = (silver.groupBy("hour", "member_casual").agg(F.count("*").alias("trips"), F.avg("trip_minutes").alias("avg_min"))
       .orderBy("hour", "member_casual").collect())
results["job_a_seconds_single_statement"] = round(time.time() - t0, 2)
mismatch = sum(1 for a, b in zip(rows, one) if (a["hour"], a["user"], a["trips"]) != (b["hour"], b["member_casual"], b["trips"]))
results["rdd_vs_df_mismatches"] = mismatch
print(f"single statement: {results['job_a_seconds_single_statement']}s | mismatches vs explicit map/reduce: {mismatch}")

print("RESULTS_JSON:" + json.dumps(results))
