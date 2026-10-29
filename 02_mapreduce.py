# Databricks notebook source
# MAGIC %md
# MAGIC # BBIM709 Assessment 2 - Notebook 02: MapReduce (Task 2)
# MAGIC Needs a **classic** cluster in *Dedicated (single user)* access mode (the RDD API is not available on serverless).
# MAGIC No classic cluster (e.g. Free Edition)? Use `02b_mapreduce_dataframe.py` instead.
# MAGIC Section 6 runs the same Job A with the DataFrame API for comparison.

# COMMAND ----------
import time, json, operator
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
# RDD MapReduce needs the RDD API: classic cluster in *Dedicated (single user)* access mode.
try:
    spark.range(10).rdd.map(lambda r: r[0]).count()
except Exception as e:
    raise RuntimeError("RDD API is not available on this compute (serverless / standard access mode). "
                       "Attach a classic cluster in Dedicated access mode, or run 02b_mapreduce_dataframe.py instead.") from e
sc = spark.sparkContext
results = {}

# COMMAND ----------
# MAGIC %md ## 1. Input handling: read Silver and keep only the columns the job needs

# COMMAND ----------
silver = load_table(T_SILVER).select("start_station_name", "hour", "dow", "member_casual", "trip_minutes")
rdd = silver.rdd.map(lambda r: (r["start_station_name"], r["hour"], r["dow"], r["member_casual"], r["trip_minutes"]))
rdd = rdd.cache()
n = rdd.count()
results["input_records"] = n
results["input_partitions"] = rdd.getNumPartitions()
results["default_parallelism"] = sc.defaultParallelism
print("Records:", n, "| partitions:", rdd.getNumPartitions(), "| default parallelism:", sc.defaultParallelism)

# COMMAND ----------
# MAGIC %md ## 2. Job A (temporal analysis): trips and mean duration by (hour of day, user type)
# MAGIC * **Map**: record -> ((hour, member_casual), (trip_minutes, 1))
# MAGIC * **Combine/Reduce**: add partial sums and counts per key (`reduceByKey` pre-aggregates inside each partition = combiner)
# MAGIC * **Finalise**: sum / count -> mean

# COMMAND ----------
t0 = time.time()
mapped = rdd.map(lambda x: ((x[1], x[3]), (x[4], 1)))
reduced = mapped.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]), numPartitions=8)
job_a = reduced.mapValues(lambda v: (v[1], v[0] / v[1])).sortByKey().collect()
t_a = time.time() - t0
results["job_a_seconds_rdd"] = round(t_a, 2)
results["job_a_keys"] = len(job_a)
results["job_a"] = [{"hour": k[0], "user": k[1], "trips": v[0], "avg_min": round(v[1], 2)} for k, v in job_a]
print(f"Job A finished in {t_a:.2f}s, {len(job_a)} keys")
for k, v in job_a[:8]:
    print(k, v)

# COMMAND ----------
# MAGIC %md ## 3. Job B (counting): top-10 busiest start stations
# MAGIC * **Map**: record -> (station, 1) * **Reduce**: sum

# COMMAND ----------
t0 = time.time()
top10 = (rdd.map(lambda x: (x[0], 1)).reduceByKey(operator.add).takeOrdered(10, key=lambda kv: -kv[1]))
results["job_b_seconds_rdd"] = round(time.time() - t0, 2)
results["top10_stations"] = [{"station": k, "trips": v} for k, v in top10]
for k, v in top10:
    print(f"{v:>7,}  {k}")

# COMMAND ----------
# MAGIC %md ## 4. Job C (anomaly detection): hour/day cells whose mean duration is > 2 standard deviations from the overall mean
# MAGIC Two chained MapReduce stages: stage 1 builds the (dow, hour) means; stage 2 reduces over cells to get mean/std.

# COMMAND ----------
cell = (rdd.map(lambda x: ((x[2], x[1]), (x[4], x[4] ** 2, 1)))
           .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2]))
           .mapValues(lambda v: (v[0] / v[2], v[2])).collect())
means = [m for _, (m, c) in cell]
mu = sum(means) / len(means)
sd = (sum((m - mu) ** 2 for m in means) / len(means)) ** 0.5
anomalies = sorted([(k, round(m, 1), c) for k, (m, c) in cell if abs(m - mu) > 2 * sd], key=lambda t: -t[1])
results["anomaly_mu"], results["anomaly_sd"] = round(mu, 2), round(sd, 2)
results["anomalies"] = [{"dow": k[0], "hour": k[1], "avg_min": m, "trips": c} for k, m, c in anomalies]
print(f"cells={len(cell)} mean-of-means={mu:.2f} sd={sd:.2f} anomalies={len(anomalies)}")
for a in anomalies[:10]:
    print(a)

# COMMAND ----------
# MAGIC %md ## 5. Partitioning and shuffle experiments (RDD)

# COMMAND ----------
exp = []
for p in (2, 8, 32):
    t0 = time.time()
    r = rdd.map(lambda x: ((x[1], x[3]), (x[4], 1))).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]), numPartitions=p).count()
    exp.append({"reduce_partitions": p, "seconds": round(time.time() - t0, 2)})
results["partition_experiment"] = exp
print(exp)

t0 = time.time()
g = rdd.map(lambda x: ((x[1], x[3]), x[4])).groupByKey().mapValues(lambda it: sum(it) / len(it)).count()
results["groupByKey_seconds"] = round(time.time() - t0, 2)
print("groupByKey version:", results["groupByKey_seconds"], "s vs reduceByKey:", exp[1]["seconds"], "s")

# COMMAND ----------
# MAGIC %md ## 6. Same Job A with the DataFrame API (Catalyst/Photon, runs on serverless)

# COMMAND ----------
t0 = time.time()
df_a = (load_table(T_SILVER)
        .groupBy("hour", "member_casual")
        .agg(F.count("*").alias("trips"), F.avg("trip_minutes").alias("avg_min"))
        .orderBy("hour", "member_casual"))
rows = df_a.collect()
results["job_a_seconds_df"] = round(time.time() - t0, 2)
mismatch = sum(1 for (k, v), r in zip(job_a, rows) if (r["hour"], r["member_casual"]) != k or r["trips"] != v[0])
results["rdd_vs_df_mismatches"] = mismatch
print(f"DataFrame Job A: {results['job_a_seconds_df']}s | key/count mismatches vs RDD: {mismatch}")
df_a.show(6)

rdd.unpersist()
print("RESULTS_JSON:" + json.dumps(results))
