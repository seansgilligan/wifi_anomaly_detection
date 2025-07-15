# runner.py

from config import Config
from joblib import load
from pyspark.sql import functions as F
from pyspark.sql.functions import to_timestamp, col, when
from aggregation import aggregate_data
import pandas as pd
import re
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate, COMMASPACE


def load_recent_data(spark, hdfs_path, minutes, fallback_limit=10):
    jvm = spark._jvm
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
    path = jvm.org.apache.hadoop.fs.Path(hdfs_path)
    cutoff = datetime.now() - timedelta(minutes=minutes)

    file_info = []
    for file_status in fs.listStatus(path):
        file_path = file_status.getPath().toString()
        if file_path.endswith(".csv"):
            match = re.search(r"thy_wifi_edr_(\d{6})_(\d{6})", file_path)
            if match:
                try:
                    timestamp = datetime.strptime(match.group(1) + match.group(2), "%y%m%d%H%M%S")
                    file_info.append((file_path, timestamp))
                except ValueError:
                    continue

    recent_files = [fp for fp, ts in file_info if ts >= cutoff]
    if not recent_files:
        print(f"⚠️ No files in last {minutes} minutes. Falling back to latest {fallback_limit} files.")
        file_info.sort(key=lambda x: x[1], reverse=True)
        recent_files = [fp for fp, _ in file_info[:fallback_limit]]

    if not recent_files:
        raise ValueError("❌ No recent HDFS files found for inference.")

    print(f"📂 Selected {len(recent_files)} recent files:")
    for f in recent_files:
        print(" -", f)

    df = spark.read.option("header", True).option("inferSchema", True).csv(recent_files)
    df = df.withColumn("timestamp", to_timestamp(col("edr_date")))
    df = aggregate_data(df, Config.WINDOW_MINUTES)

    if df.rdd.isEmpty():
        print("⚠️ Aggregated DataFrame is empty. Possibly no valid FAIL/SUCCESS rows.")
    else:
        print(f"✅ Aggregated DataFrame contains {df.count()} rows.")

    return df


def detect_zscore(spark, spark_df):
    print("🔍 Running Z-score anomaly detection...")
    stats_pdf = pd.read_csv(Config.ZSCORE_STATS_PATH)
    stats_df = spark.createDataFrame(stats_pdf)
    df = spark_df.join(stats_df, on="fault_key", how="left")
    df = df.withColumn("z_score", (col("fail_rate") - col("mean_rate")) / col("std_rate"))
    df = df.withColumn("anomaly_z", when(col("z_score") > 3, 1).otherwise(0))
    print("✅ Z-score detection completed.")
    return df


def detect_iforest(spark, spark_df):
    print("🔍 Running Isolation Forest anomaly detection...")
    pdf = spark_df.select("fail_rate", "fault_key", "start_time").toPandas()
    if pdf.empty:
        print("⚠️ No data to run Isolation Forest. Skipping.")
        spark_df = spark_df.withColumn("anomaly_iforest_if", F.lit(0))
        return spark_df

    model = load(Config.IFOREST_MODEL_PATH)
    pdf["anomaly_iforest_if"] = pd.Series(model.predict(pdf[["fail_rate"]])).map({1: 0, -1: 1})
    df_pred = spark_df.sparkSession.createDataFrame(pdf[["fault_key", "start_time", "anomaly_iforest_if"]])
    print("✅ Isolation Forest detection completed.")
    return spark_df.join(df_pred, on=["fault_key", "start_time"], how="left")


def write_anomalies_to_cassandra(df, table):
    print(f"📤 Writing {df.count()} anomalies to Cassandra ({table})...")
    df.write \
        .format("org.apache.spark.sql.cassandra") \
        .mode("append") \
        .options(table=table, keyspace=Config.CASSANDRA_KEYSPACE) \
        .save()
    print("✅ Write complete.")


def send_email_report(df):
    ts = datetime.now().strftime('%Y%m%d')
    subject = f"THY.WIFI Tespit Edilen Anomali Raporu"
    filename = f"thy_wifi_anomali_raporu_{ts}.csv"
    pdf = df.toPandas()
    pdf.to_csv(filename, index=False)

    if pdf.empty:
        html_main = "<p><b>Anomali bulunmadı.</b></p>"
    else:
        html_main = pdf.to_html(index=False, escape=False)

    body_html = f"""
    <html><body>
    <p>Merhaba,<br><br>
    Son çalıştırmada <b>{len(pdf)}</b> adet anomali tespit edilmiştir. Detaylar aşağıda yer almaktadır:</p>
    <h3>Anomali Kayıtları</h3>{html_main}
    <br>İyi çalışmalar.<br></body></html>
    """

    msg = MIMEMultipart()
    msg["From"] = Config.MAIL_FROM
    msg["To"] = COMMASPACE.join(Config.MAIL_TO)
    if Config.MAIL_CC:
        msg["Cc"] = COMMASPACE.join(Config.MAIL_CC)
    if Config.MAIL_BCC:
        msg["Bcc"] = COMMASPACE.join(Config.MAIL_BCC)
    msg["Date"] = formatdate(localtime=True)
    msg["Subject"] = subject
    msg.attach(MIMEText(body_html, "html"))

    try:
        host, port = Config.MAIL_SERVER.split(":")
        server = smtplib.SMTP(host, int(port))
        server.starttls()
        server.login(Config.EMAIL_USERNAME, Config.EMAIL_PASSWORD)
        server.sendmail(Config.MAIL_FROM, Config.MAIL_TO + Config.MAIL_CC + Config.MAIL_BCC, msg.as_string())
        server.quit()
        print("📧 Anomali raporu e-posta ile gönderildi.")
    except Exception as e:
        print(f"[ERROR] E-posta gönderimi başarısız: {e}")


def main():
    print("Starting WiFi Anomaly Runner...")
    spark = SparkSession.getActiveSession()
    if spark is None:
        raise RuntimeError("❌ Spark session not found. Ensure you're running inside Airflow with Spark running.")

    df = load_recent_data(spark, Config.HDFS_PATH, Config.WINDOW_MINUTES)
    df_z = detect_zscore(spark, df)
    df_if = detect_iforest(spark, df)

    df_combined = df_z.join(df_if.select("fault_key", "start_time", "anomaly_iforest_if"),
                            on=["fault_key", "start_time"], how="left")
    df_combined = df_combined.withColumn("anomaly_iforest", col("anomaly_iforest_if"))

    df_agreed = df_combined.filter((col("anomaly_z") == 1) & (col("anomaly_iforest") == 1))
    df_unrepeated = df_combined.withColumn("anomaly_combined", ((col("anomaly_z") + col("anomaly_iforest")) >= 1).cast("int"))

    df_z_all = df_combined.filter(col("anomaly_z") == 1)
    df_z_repeated = df_z_all.filter(col("anomaly_iforest") == 1)

    df_if_all = df_combined.filter(col("anomaly_iforest") == 1)
    df_if_repeated = df_if_all.filter(col("anomaly_z") == 1)

    if not df_z_all.rdd.isEmpty():
        write_anomalies_to_cassandra(df_z_all, Config.TABLE_ZSCORE_ALL)
    if not df_z_repeated.rdd.isEmpty():
        write_anomalies_to_cassandra(df_z_repeated, Config.TABLE_ZSCORE_REPEATED)

    if not df_if_all.rdd.isEmpty():
        write_anomalies_to_cassandra(df_if_all, Config.TABLE_IFOREST_ALL)
    if not df_if_repeated.rdd.isEmpty():
        write_anomalies_to_cassandra(df_if_repeated, Config.TABLE_IFOREST_REPEATED)

    if df_unrepeated.rdd.isEmpty():
        print("⚠️ No unrepeated anomalies detected.")
    else:
        print("📊 Unrepeated anomalies (zscore or iforest):")
        df_unrepeated.show(truncate=False)
        write_anomalies_to_cassandra(df_unrepeated, Config.TABLE_COMBINED_ALL)

    if df_agreed.rdd.isEmpty():
        print("✅ No agreed anomalies detected in this window.")
    else:
        print("🚨 AGREED ANOMALIES DETECTED:")
        df_agreed.show(truncate=False)
        write_anomalies_to_cassandra(df_agreed, Config.TABLE_COMBINED_REPEATED)

    send_email_report(df_agreed)
    print("🏎️ Runner finished.")


if __name__ == "__main__":
    main()
