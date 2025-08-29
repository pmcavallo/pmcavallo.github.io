---
title: "SignalGraph 5G: Anomaly Detection & Forecasts"
date: 2025-08-29
categories: [Projects, AI, Networks]
tags: [Spark, PySpark, Streamlit, Verizon, Anomaly Detection, Forecasting, Graph Analytics]
---

## Overview

**SignalGraph 5G** is a Verizon-aligned demo system that ingests synthetic 4G/5G KPI data, processes it through a Spark-based lakehouse pipeline, and exposes an analyst-friendly UI in Streamlit. The project was designed to **demonstrate the exact skills listed in Verizon’s Senior Data Scientist role**: anomaly detection, large-scale data engineering, data warehouse/lakehouse integration, and applied ML/forecasting in the network domain.

---

## Architecture at a Glance

- **Data pipeline:** Spark jobs implementing **Bronze → Silver → Gold** partitions (Parquet, hive-style).
- **Anomaly detection:** Initial rule-based seed (`latency_ms > 60 OR prb_util_pct > 85`), extending to supervised models (XGBoost, Prophet).
- **Storage & Lakehouse:** Hive partitions for scale-out processing; DuckDB/Postgres mirrors for BI/ops integration.
- **UI:** Streamlit analyst view with partition filters, anomaly tables, and alerts.
- **Forecasting:** Prophet-based forecasts on latency and PRB utilization.
- **Planned extensions:** Graph analytics with Neo4j (cell neighbors, centrality), warehouse DDL for Teradata/Postgres, SHAP/feature attribution.

---

## App Screenshots

**SignalGraph Analyst UI — Anomaly Monitoring**  
*Shows filtered Silver data with anomaly flags and region/date controls.*

![UI Screenshot Placeholder](./assets/signalgraph-ui-1.png)

**SignalGraph Alerts & Model Risk Scores**  
*Highlights top cells by anomaly rate and risk scores, with placeholder logic for alert explanations.*

![Alerts Screenshot Placeholder](./assets/signalgraph-ui-2.png)

---

## Key Features

### 1. Spark Bronze → Silver ETL
We start with synthetic KPI data (`cell_kpi_minute.parquet`) and run a Spark job to enforce schema, add partitions, and flag anomalies:

```python
df2 = (df
    .withColumn("date", F.to_date("ts"))
    .withColumn("hour", F.hour("ts"))
    .withColumn("region", F.coalesce(F.col("region"), F.lit("unknown"))))

df2 = (df2
    .filter(F.col("rsrp_dbm").isNotNull() & F.col("sinr_db").isNotNull())
    .withColumn("latency_poor", (F.col("latency_ms") > 60))
    .withColumn("prb_high", (F.col("prb_util_pct") > 85))
    .withColumn("anomaly_flag", (F.col("latency_poor") | F.col("prb_high")).cast("int")))
```

This produces **hive-partitioned Silver tables** by `date` and `region`.

---

### 2. Silver → Gold Aggregates
The **Gold layer** summarizes KPIs hourly per cell and generates a **next-hour anomaly label** for supervised training:

```python
agg = df.groupBy("cell_id", "region", "ts_hour").agg(
    F.count("*").alias("n_samples"),
    F.avg("latency_ms").alias("latency_ms_avg"),
    F.expr("percentile_approx(latency_ms, 0.95, 100)").alias("latency_ms_p95"),
    F.avg("prb_util_pct").alias("prb_util_pct_avg"),
    ...
).withColumn("y_next_anomaly", F.lead("anomaly_any").over(w).cast("int"))
```

---

### 3. Analyst UI (Streamlit)
Streamlit powers an **interactive analyst view**. It filters regions/dates, shows anomaly tables, and surfaces top cells by anomaly rate:

```python
st.subheader("Top cells by anomaly rate")
top = (
    df.groupby("cell_id")[["anomaly_flag", "latency_ms", "prb_util_pct"]]
      .mean(numeric_only=True)
      .sort_values("anomaly_flag", ascending=False)
      .reset_index()
      .head(10)
)
st.dataframe(top)
```

Alerts are also generated when model risk scores exceed thresholds:

```python
alerts = scores[scores["risk_score"] >= thr].copy()
alerts["reason"] = alerts.apply(reason_from_row, axis=1)
```

---

### 4. Forecasting (Prophet & Visualization)
SignalGraph includes **time-series forecasting** for key KPIs like latency and PRB utilization. Forecasts include prediction intervals and network engineering guardrails:

```python
ax.plot(obs["ds"], obs["y"], label="Observed p95 latency (ms)")
ax.plot(fc["ds"], fc["yhat"], label="Forecast (yhat)")
ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], alpha=0.2)
```

---

## Why This Matters

- **Direct tie to Verizon’s job requirements**: anomaly detection, forecasting, large data sets, Spark/Hadoop-style pipelines, Teradata/warehouse mirroring, and 4G/5G KPI expertise.  
- **Production discipline**: schema contracts, timestamp precision guardrails, partitioning strategies, and model monitoring artifacts.  
- **Scalable & extensible**: Designed to drop into **Dataproc/EMR** clusters and extend into graph/network analysis.  

---

## Next Steps

- Add **baseline XGBoost classifier** for anomaly prediction (using Gold labels).  
- Implement **SHAP feature attribution** for explainability in alerts.  
- Integrate **Neo4j** for neighbor/centrality analysis.  
- Mirror DuckDB marts into **Postgres/Teradata** with clean DDL.  

---

## Tech Stack

- **Languages & Libraries:** Python 3.10, PySpark 3.5.1, pandas, scikit-learn, XGBoost, Prophet, matplotlib, DuckDB.  
- **Frameworks:** Streamlit UI, Spark ETL, PyArrow.  
- **Data Stores:** Hive-partitioned Parquet, DuckDB, Postgres/Teradata schema.  
- **Domain:** 4G/5G KPIs (RSRP, RSRQ, SINR, PRB utilization, latency, jitter, packet loss).  

---

## Repository

👉 [View SignalGraph on GitHub](https://github.com/pmcavallo/signalgraph)
