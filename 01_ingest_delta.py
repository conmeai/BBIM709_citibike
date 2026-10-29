# Databricks notebook source
# MAGIC %md
# MAGIC # BBIM709 Assessment 2 - Notebook 01: Ingestion and Delta Lake (Task 1)
# MAGIC Medallion pipeline: raw CSV -> Bronze (typed) -> Silver (cleaned + features) -> Gold (hourly demand).
# MAGIC Works on serverless **and** classic compute. Tables are created as Unity Catalog managed Delta tables:
# MAGIC `citibike_bronze`, `citibike_silver` (partitioned by month) and `citibike_gold_hourly`.

# COMMAND ----------
import time, json
from pyspark.sql import functions as F, types as T

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
RAW_FILE = "citibike_2021_full_year_simulated.csv"   # use ".csv.gz" if you uploaded the gzip version
RAW_PATH = f"{VOLUME}/raw/{RAW_FILE}" if ON_DATABRICKS else "/mnt/user-data/outputs/citibike_2021_full_year_simulated.csv"
results = {}

# COMMAND ----------
# MAGIC %md ## 1. Extract: read raw CSV with an explicit schema (no inferSchema = one pass, no type surprises)

# COMMAND ----------
raw_schema = T.StructType([T.StructField(c, T.StringType(), True) for c in [
    "ride_id", "rideable_type", "started_at", "ended_at", "start_station_name", "start_station_id",
    "end_station_name", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng",
    "member_casual", "orig_started_at", "orig_ended_at", "is_simulated_date"]])

t0 = time.time()
raw = spark.read.option("header", True).schema(raw_schema).csv(RAW_PATH)

bronze = (raw
    .withColumn("started_at", F.to_timestamp("started_at"))
    .withColumn("ended_at", F.to_timestamp("ended_at"))
    .withColumn("orig_started_at", F.to_timestamp("orig_started_at"))
    .withColumn("orig_ended_at", F.to_timestamp("orig_ended_at"))
    .withColumn("is_simulated_date", F.col("is_simulated_date").cast("boolean"))
    .select("*", *[F.col(c).cast("double").alias(f"_{c}") for c in ["start_lat", "start_lng", "end_lat", "end_lng"]])
    .drop("start_lat", "start_lng", "end_lat", "end_lng")
    .withColumnRenamed("_start_lat", "start_lat").withColumnRenamed("_start_lng", "start_lng")
    .withColumnRenamed("_end_lat", "end_lat").withColumnRenamed("_end_lng", "end_lng")
    .withColumn("_source_file", F.col("_metadata.file_path"))
    .withColumn("_ingested_at", F.current_timestamp()))

save_table(bronze, T_BRONZE)
bronze = load_table(T_BRONZE)
n_bronze = bronze.count()
results["bronze_rows"] = n_bronze
results["bronze_seconds"] = round(time.time() - t0, 1)
print("Bronze rows:", n_bronze)
bronze.printSchema()

# COMMAND ----------
# MAGIC %md ## 2. Data-quality profile (drives the cleaning rules)

# COMMAND ----------
dq = bronze.select(
    F.count("*").alias("rows"),
    F.countDistinct("ride_id").alias("distinct_ride_id"),
    F.sum(F.col("end_station_name").isNull().cast("int")).alias("null_end_station"),
    F.sum(F.col("end_lat").isNull().cast("int")).alias("null_end_coord"),
    F.sum(((F.unix_timestamp("ended_at") - F.unix_timestamp("started_at")) / 60 > 1440).cast("int")).alias("trips_over_24h"),
    F.sum(((F.unix_timestamp("ended_at") - F.unix_timestamp("started_at")) / 60 < 1).cast("int")).alias("trips_under_1min"),
    F.sum((F.year("started_at") != 2021).cast("int")).alias("outside_2021"),
    F.min("started_at").alias("min_start"), F.max("started_at").alias("max_start"),
    F.min("start_lat").alias("min_lat"), F.max("start_lat").alias("max_lat"),
    F.min("start_lng").alias("min_lng"), F.max("start_lng").alias("max_lng"),
).collect()[0].asDict()
dq = {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in dq.items()}
results["data_quality"] = dq
print(json.dumps(dq, indent=2))

# COMMAND ----------
# MAGIC %md ## 3. Transform: Silver layer (cleaning rules + engineered features)

# COMMAND ----------
t0 = time.time()
R = 6371.0088  # Earth radius (km)
lat1, lat2 = F.radians("start_lat"), F.radians("end_lat")
dlat, dlon = lat2 - lat1, F.radians(F.col("end_lng") - F.col("start_lng"))
hav = F.sin(dlat / 2) ** 2 + F.cos(lat1) * F.cos(lat2) * F.sin(dlon / 2) ** 2

silver = (bronze
    .dropDuplicates(["ride_id"])                                           # R1 unique trips
    .filter(F.year("started_at") == 2021)                                  # R2 inside study year
    .withColumn("trip_minutes", (F.unix_timestamp("ended_at") - F.unix_timestamp("started_at")) / 60)
    .filter((F.col("trip_minutes") >= 1) & (F.col("trip_minutes") <= 1440))  # R3 plausible duration
    .filter(F.col("end_lat").isNotNull() & F.col("end_lng").isNotNull())   # R4 complete destination
    .filter(F.col("start_lat").between(40.5, 41.0) & F.col("start_lng").between(-74.3, -73.6))  # R5 NYC bounds
    .withColumn("distance_km", 2 * R * F.asin(F.sqrt(hav)))
    .withColumn("hour", F.hour("started_at"))
    .withColumn("dow", F.dayofweek("started_at"))                          # 1 = Sunday
    .withColumn("month", F.month("started_at"))
    .withColumn("is_weekend", F.col("dow").isin(1, 7).cast("int"))
    .withColumn("hour_ts", F.date_trunc("hour", "started_at"))
    .withColumn("label_casual", (F.col("member_casual") == "casual").cast("double"))
    .select("ride_id", "rideable_type", "started_at", "ended_at", "start_station_name", "start_station_id",
            "end_station_name", "end_station_id", "start_lat", "start_lng", "end_lat", "end_lng",
            "member_casual", "label_casual", "trip_minutes", "distance_km", "hour", "dow", "month",
            "is_weekend", "hour_ts"))

save_table(silver, T_SILVER, partition_by="month")
silver = load_table(T_SILVER)
n_silver = silver.count()
results["silver_rows"] = n_silver
results["rows_removed"] = n_bronze - n_silver
results["silver_seconds"] = round(time.time() - t0, 1)
print(f"Silver rows: {n_silver:,} (removed {n_bronze - n_silver:,})")

# COMMAND ----------
# MAGIC %md ## 4. Gold layer: complete hourly demand series (8,760 hours) for time-series modelling

# COMMAND ----------
grid = (spark.range(1).select(F.explode(F.expr(
    "sequence(to_timestamp('2021-01-01 00:00:00'), to_timestamp('2021-12-31 23:00:00'), interval 1 hour)")).alias("hour_ts")))
counts = silver.groupBy("hour_ts").agg(F.count("*").alias("trips"),
                                       F.avg("trip_minutes").alias("avg_minutes"),
                                       F.avg("label_casual").alias("casual_share"))
gold = grid.join(counts, "hour_ts", "left").fillna({"trips": 0}).orderBy("hour_ts")
save_table(gold, T_GOLD)
gold = load_table(T_GOLD)
results["gold_rows"] = gold.count()
print("Gold hourly rows:", results["gold_rows"])
gold.show(5, truncate=False)

# COMMAND ----------
# MAGIC %md ## 5. Delta Lake features (Databricks only): OPTIMIZE / Z-ORDER, history, time travel

# COMMAND ----------
if ON_DATABRICKS:
    spark.sql(f"OPTIMIZE {T_SILVER} ZORDER BY (start_station_id)")
    spark.sql(f"DESCRIBE HISTORY {T_SILVER}").select("version", "timestamp", "operation").show(truncate=False)
    spark.sql(f"DESCRIBE DETAIL {T_SILVER}").select("format", "numFiles", "sizeInBytes", "partitionColumns").show(truncate=False)
    spark.sql(f"SELECT COUNT(*) AS rows_at_version_0 FROM {T_SILVER} VERSION AS OF 0").show()

print("RESULTS_JSON:" + json.dumps(results))
