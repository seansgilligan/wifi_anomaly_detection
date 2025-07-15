from pyspark.sql.functions import (
    col, count, split, concat_ws, window, round as spark_round, lit
)

def aggregate_data(df, window_minutes):
    w = f"{window_minutes} minutes"

    fail_df = df.filter(col("state") == "FAIL") \
               .withColumn("fault_key", concat_ws(" | ", col("edr_point"), col("description")))

    fail_agg = fail_df.groupBy(window("timestamp", w), "fault_key") \
                       .agg(count("*").alias("fail")) \
                       .withColumn("edr_point", split(col("fault_key"), " \\| ")[0])

    success_df = df.filter(col("state") == "SUCCESS")
    success_agg = success_df.groupBy(window("timestamp", w), col("edr_point")) \
                             .agg(count("*").alias("success"))

    joined = fail_agg.join(success_agg,
                           (fail_agg["window"] == success_agg["window"]) &
                           (fail_agg["edr_point"] == success_agg["edr_point"]),
                           how="left") \
                     .drop(success_agg["window"]) \
                     .drop(success_agg["edr_point"])

    return joined.fillna({"success": 0}) \
                 .withColumn("total", col("fail") + col("success")) \
                 .withColumn("fail_rate", spark_round((col("fail") / col("total")) * 100, 2)) \
                 .withColumn("anomaly_iforest", lit(0)) \
                 .select(
                     col("window.start").alias("start_time"),
                     col("window.end").alias("end_time"),
                     "fault_key", "fail", "success", "total", "fail_rate", "anomaly_iforest"
                 )
