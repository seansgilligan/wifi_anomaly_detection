from config import Config
from joblib import dump
from pyspark.sql.functions import to_timestamp, col, avg, stddev
from aggregation import aggregate_data
import pandas as pd
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import re

def load_and_aggregate_last_n_days(spark, hdfs_path, days):
    jvm = spark._jvm
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    path = jvm.org.apache.hadoop.fs.Path(hdfs_path)
    cutoff = datetime.now() - timedelta(days=days)

    files = []
    for file_status in fs.listStatus(path):
        file_path = file_status.getPath().toString()
        if file_path.endswith(".csv"):
            match = re.search(r"thy_wifi_edr_(\d{6})_", file_path)
            if match:
                file_date = datetime.strptime(match.group(1), "%y%m%d")
                if file_date >= cutoff:
                    files.append(file_path)

    if not files:
        raise ValueError("❌ No training files found in the last {} days.".format(days))

    df = spark.read.option("header", True).option("inferSchema", True).csv(files)
    df = df.withColumn("timestamp", to_timestamp(col("edr_date")))
    df = aggregate_data(df, Config.WINDOW_MINUTES)
    print(f"✅ Aggregated training data contains {df.count()} rows.")
    return df

def train_zscore_stats(spark):
    print("🔧 Training Z-score thresholds...")
    df = load_and_aggregate_last_n_days(spark, Config.HDFS_PATH, Config.TRAINING_DAYS)
    stats = df.groupBy("fault_key").agg(
        avg("fail_rate").alias("mean_rate"),
        stddev("fail_rate").alias("std_rate")
    )
    pdf = stats.toPandas()
    Config.MODEL_DIR.mkdir(exist_ok=True, parents=True)
    pdf.to_csv(Config.ZSCORE_STATS_PATH, index=False)
    print(f"✅ Z-score stats saved to {Config.ZSCORE_STATS_PATH}")

def train_isolation_forest(spark):
    print("🔧 Training Isolation Forest model...")
    df = load_and_aggregate_last_n_days(spark, Config.HDFS_PATH, Config.TRAINING_DAYS)
    pdf = df.select("fail_rate", "fault_key", "start_time").toPandas()

    if pdf.empty:
        raise ValueError("❌ Not enough data to train Isolation Forest.")

    model = IsolationForest(**Config.IFOREST_PARAMS)
    model.fit(pdf[["fail_rate"]])
    Config.MODEL_DIR.mkdir(exist_ok=True, parents=True)
    dump(model, Config.IFOREST_MODEL_PATH)
    print(f"✅ Isolation Forest model saved to {Config.IFOREST_MODEL_PATH}")

def main():
    print("🚀 Starting THY WiFi Anomaly Trainer...")
    from pyspark.sql import SparkSession
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("❌ Spark session not found. Ensure you're running inside Airflow with Spark running.")

    train_zscore_stats(spark)
    train_isolation_forest(spark)
    print("🏁 Trainer finished.")

if __name__ == "__main__":
    main()
