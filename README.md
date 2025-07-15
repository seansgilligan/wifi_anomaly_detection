# 📡 WiFi Anomaly Detection Pipeline

This repository implements a real-time anomaly detection system for analyzing in-flight WiFi event logs. It processes HDFS data, detects unusual failure patterns using both statistical (Z-score) and machine learning (Isolation Forest) techniques, and exports the results to Cassandra and email reports.

## 📁 Project Structure
wifi_anomaly/
├── config.py         # Centralized configuration for paths, parameters, Cassandra, email, etc.

├── train.py          # Trainer script for model and statistical profile generation

├── runner.py         # Inference and alerting script

├── aggregation.py    # Data aggregation utilities (e.g., fail/success rate by time window)

├── model_store/

│   ├── iforest_model.pkl   # Trained Isolation Forest model

│   └── zscore_stats.csv    # Per-fault key mean and std for Z-score detection

## 🧠 Core Features

-  **Z-score Anomaly Detection**: Statistical method based on standard deviation
-  **Isolation Forest**: Machine learning model trained on failure rates
-  **Aggregation**: Customizable time window (default: 5 minutes)
-  **Repeated/Unrepeated Anomaly Separation**
-  **Cassandra Integration**: For storing anomalies by method
-  **Email Notification**: Sends HTML reports with anomaly details (even if empty)
-  **Airflow Support**: Optimized for use in DAGs with KubernetesPodOperator

---
- Loads last 5 minutes of data (with fallback to last N files)
- Aggregates and applies both anomaly detectors
- Stores results in 6 separate Cassandra tables
- Sends summary email

---

##  ⚙️Configuration

Modify `config.py` to set:

- HDFS path to log files
- Cassandra keyspace and table names
- Output paths for model and statistics
- Spark aggregation window duration
- Model hyperparameters
- Email SMTP settings and recipients

---
| Table Name         | Description                                      |
|--------------------|--------------------------------------------------|
| `combined_all`     | All anomalies (Z-score ∪ Isolation Forest)       |
| `combined_repeated`| Confirmed anomalies (Z-score ∩ Isolation Forest) |
| `zscore_all`       | All Z-score anomalies                            |
| `zscore_repeated`  | Z-score anomalies confirmed by IForest           |
| `iforest_all`      | All Isolation Forest anomalies                   |
| `iforest_repeated` | IForest anomalies confirmed by Z-score           |

---

### 🔧 Training

Generates Z-score stats and trains Isolation Forest on recent logs.

---
### 🧱Requirements
-Python 3.7+
-PySpark
-pandas
-scikit-learn
-joblib
-smtplib (built-in)
-Apache Airflow (KubernetesPodOperator)
-Apache Cassandra (via Spark connector)
-HDFS
---

```bash
python train.py

