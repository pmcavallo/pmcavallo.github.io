---
layout: post
title: SignalGraph 5G - Anomaly Detection & Forecasts (PySpark + Postgres/Teradata + Prophet)
date: 2025-08-29
---

**SignalGraph 5G** is a demo system that ingests synthetic 4G/5G KPI data, processes it through a Spark-based lakehouse pipeline, and exposes an analyst-friendly UI in Streamlit. The project was designed for anomaly detection, large-scale data engineering, data warehouse/lakehouse integration, and applied ML/forecasting in the network domain.

**Live App:** [Render Deployment](https://signalgraph.onrender.com/) 

This demo uses a minimal Silver slice committed to the repo and a Neon Postgres view to mirror the warehouse mart. Expect a short cold start on the Free plan.

---

## Architecture at a Glance

- **Data pipeline:** Spark jobs implementing **Bronze → Silver → Gold** partitions (Parquet, hive-style).
- **Anomaly detection:** Initial rule-based seed (`latency_ms > 60 OR prb_util_pct > 85`), extending to supervised models (XGBoost, Prophet).
- **Storage & Lakehouse:** Hive partitions for scale-out processing; DuckDB/Postgres mirrors for BI/ops integration.
- **UI:** Streamlit analyst view with partition filters, anomaly tables, and alerts. It will also be deployed on Render, making it accessible without setup.
- **Forecasting:** Prophet-based forecasts on latency and PRB utilization.
- **Planned extensions:** Graph analytics with Neo4j (cell neighbors, centrality), warehouse DDL for Teradata/Postgres, SHAP/feature attribution.

---

## Key Factors in the Data

The dataset used in SignalGraph is synthetic and was generated to reflect realistic 4G/5G KPIs. The synthetic generation used in the project was meant to create plausible correlations (e.g., high PRB utilization leading to higher latency and packet loss).

- **latency_ms** – average user-plane latency; higher values mean slower responsiveness.
- **prb_util_pct** – PRB utilization percentage; proxy for network load and congestion.
- **thrpt_mbps** – downlink throughput; what the end user experiences as speed.
- **sinr_db** – signal-to-interference-plus-noise ratio; higher values mean clearer signal.
- **rsrp_dbm** – reference signal received power; less negative is stronger signal.
- **pkt_loss_pct / drop_rate_pct** – packet or call drop percentage; reliability indicator.

---

## App Screenshots

**SignalGraph Analyst UI — Anomaly Monitoring**  
*Shows filtered Silver data with anomaly flags and region/date controls.*

![signalgraph](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/signalgraph1.png?raw=true)

**SignalGraph Alerts & Model Risk Scores**  
*Highlights top cells by anomaly rate and risk scores, with placeholder logic for alert explanations.*

![signalgraph](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/signalgraph2.png?raw=true)

**Forecast Graph — PRB Utilization**  
*Prophet-based logistic growth model applied to hourly PRB utilization (bounded 0–100). Forecast includes mean (orange) and 80% interval (blue band).*

**What is PRB Utilization?**

In 4G/5G networks, **Physical Resource Blocks (PRBs)** are the smallest unit of spectrum allocation that the base station scheduler can assign to users. Think of PRBs as the “lanes” on a highway:

- **Low PRB utilization** = free lanes → smooth traffic, high throughput, low latency.  
- **High PRB utilization (≥80–85%)** = congestion → packets queue up, throughput drops, latency and jitter spike, and call drops may increase.  

PRB utilization is therefore a direct **capacity and congestion indicator**, bridging RF (radio access) conditions with IP-level user experience.

![signalgraph](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/forecast_prb_dfw_CELL-012.png?raw=true)

**Forecast Graph: PRB Utilization**

The forecast graph shows **PRB utilization over time** at the cell level.  

- **Blue line (Observed):** historical hourly PRB usage, smoothed for noise.  
- **Orange line (Forecast):** Prophet’s logistic growth model prediction, bounded between 0–100% to reflect real physical limits.  
- **Shaded band:** 80% confidence interval, widening further out in time to capture growing uncertainty.  

**Interpretation:**  
- Sustained upward trends approaching 80–90% signal **impending congestion**, guiding proactive actions (e.g., cell splitting, carrier aggregation, or load balancing).  
- Downward stabilization near 50–60% suggests **healthy utilization**, with enough headroom for bursts.  
- The widening confidence band reflects realistic modeling: while we cannot predict exact usage, we can bound the risk of overload.

## Feature Update — Explanations, Interpretations & Polishing

Features I added to the app:

### Alerts & Thresholds
- **Interactive Threshold Slider**: Users can override the model’s default operating threshold (0.09) to surface more or fewer alerts.  
  - Lowering the threshold increases recall (more cells flagged) but also raises false positives.  
  - Default comes from `metrics.json` and is tuned for F1 balance.
 
![signalgraph](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/signalgraph3.png?raw=true)

### Cell Drill-In — Neighbors & Centrality
- **Metrics**:  
  - **Neighbor degree** counts direct peers (higher → bigger local blast-radius).  
  - **PageRank** weights peers by their influence (higher → wider impact).  
  - **Betweenness** shows if a cell is a bridge on shortest paths (higher → issues can fragment clusters).

![signalgraph](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/signalgraph4.png?raw=true)

- **Interpretation Block**: Centrality explains **network influence**. These metrics help triage whether an issue is isolated or may propagate via topology.  

### Model Metrics
- **AUC-ROC (0.745)**: Model separates risky vs. normal cells fairly well (75% chance a risky cell scores higher than a normal one).  
- **AUC-PR (0.241)**: Performance above random baseline in imbalanced data.  
- **Operating Threshold (0.09)**: Balances recall and precision; configurable in-app.  
- **F1@thr (0.437)**: Trade-off between catching risks vs. tolerating false alarms.  

### PRB Utilization Forecast
- **How to Read**:  
  - Blue line = median forecast.  
  - Shaded band = uncertainty.  
  - Threshold (85%) = capacity pressure.  

![signalgraph](https://github.com/pmcavallo/pmcavallo.github.io/blob/master/images/signalgraph5.png?raw=true)

- **Per-Cell Interpretation**: Each forecast panel automatically reports the chance of breaching the 85% PRB threshold within the next 12 hours.  

### Network SLO Summary
- **Capacity SLO (99.6%)**: Network is comfortably under PRB pressure.  
- **Latency SLO (93.7%)**: Most traffic meets the 60ms p95 target, though some cells drift.  
- **Reliability SLO (28.2%)**: Packet drops are frequent — weakest dimension.  

- **Note**: In full deployment, SLOs recompute dynamically per region and date window, so numbers change with the user’s filters.  

### Warehouse View (Postgres)
- **Purpose**: Exposes the raw `last_hour_risk` table mirrored from Postgres.  
- **Why**: Lets ops verify the same records that models and dashboards consume — ensuring transparency and auditability.  
- **User Action**: Can filter, sort, or export rows for validation.  

---

✅ These additions make the app not just a **scoreboard**, but an **interpretable triage tool** — linking model predictions, forecasts, graph centrality, and warehouse verification into one workflow.

---

## Key Features

### 1. Spark Bronze → Silver ETL
I start with synthetic KPI data (`cell_kpi_minute.parquet`) and run a Spark job to enforce schema, add partitions, and flag anomalies:

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

- **App Utility**: anomaly detection, forecasting, large data sets, Spark/Hadoop-style pipelines, Teradata/warehouse mirroring, **SLO tracking**, SHAP explainability, and 4G/5G KPI expertise.  
- **Production discipline**: schema contracts, timestamp precision guardrails, partitioning strategies, **model triage thresholds**, and monitoring artifacts.  
- **Scalable & extensible**: Designed to drop into **Dataproc/EMR** clusters and extend into graph/network analysis with Neo4j centrality and influence metrics.    

---

## Next Steps

- Deploy the Streamlit UI as a live web app on **Render** so users can interact with SignalGraph directly.  
- Mirror DuckDB marts into **Postgres/Teradata** with clean DDL.  
- Prototype a lightweight **agent layer**:  
  - Monitoring Agent: track ETL freshness and anomalies in real time.  
  - Forecasting Agent: run Prophet in parallel and compare with observed KPIs.  
  - Orchestrator Agent: combine monitoring + forecasting into a single dashboard summary.

---

## Tech Stack

- **Languages & Libraries:** Python 3.10, PySpark 3.5.1, pandas, scikit-learn, XGBoost, Prophet, matplotlib, DuckDB, SHAP, Altair.  
- **Frameworks:** Streamlit UI, Spark ETL, PyArrow.  
- **Data Stores:** Hive-partitioned Parquet, DuckDB, Postgres/Teradata schema (warehouse view).
- **Graph & Network Analysis:** Neo4j integration, centrality metrics (degree, PageRank, betweenness), neighbor drill-in.
- **Explainability & Monitoring:** SHAP local/global feature attribution, threshold tuning with triage slider, SLO summaries (capacity, latency, reliability).  
- **Domain:** 4G/5G KPIs (RSRP, RSRQ, SINR, PRB utilization, latency, jitter, packet loss).  

