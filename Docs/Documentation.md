# AI-Driven Port Logistics Management & Forecasting
### Technical Documentation — EG Maritime AI Dashboard

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Workflow & Development Stages](#3-project-workflow--development-stages)
4. [System Architecture](#4-system-architecture)
5. [File Structure](#5-file-structure)
6. [Installation & Setup](#6-installation--setup)
7. [Data Schema](#7-data-schema)
8. [Dashboard Pages](#8-dashboard-pages)
9. [ML Models](#9-ml-models)
10. [Port Reference Data](#10-port-reference-data)
11. [Configuration Constants](#11-configuration-constants)
12. [Feature Engineering](#12-feature-engineering)
13. [Known Limitations & Notes](#13-known-limitations--notes)

---

## 1. Project Overview

**EG Maritime AI** is an integrated intelligent system designed to minimize waste and inefficiencies in Egyptian maritime supply chains. It fuses three data sources — **AIS (Vessel Tracking)**, **Weather**, and **Customs** — into a unified relational dataset, then exposes four AI-powered analytical tools through an interactive Streamlit dashboard.

The system targets Egypt's six major export ports: **Alexandria, Dekheila, Damietta, Port Said, Suez,** and **Sokhna**.

**Expected Impact:**
- ~20% reduction in operational logistics costs
- Lower port storage fees and delay-related fines
- Reduced agricultural spoilage through faster, data-driven rerouting
- Balanced truck traffic distribution across ports based on real-time capacity

---

## 2. Problem Statement

Egyptian exports face three critical logistical challenges:

| Challenge | Description |
|---|---|
| **Sudden Congestion** | Unforeseen traffic at ports leading to high-cost delays and fines (demurrage) |
| **Weather Volatility** | Sea/weather fluctuations halt maritime traffic without sufficient early warning |
| **Operational Fragmentation** | No real-time synchronization between vessel arrivals and land-based truck scheduling |

---

## 3. Project Workflow & Development Stages

This project was built across **four specialised domains**, executed sequentially as a full data-to-deployment pipeline. Below is the end-to-end story of how the system was built.

---

### Stage 1 — Data Engineering

#### 1a. Problem Understanding & Data Source Research

The first step was a deep-dive into the Egyptian maritime logistics domain to understand what data variables drive port congestion and vessel delays. The team identified the following required data categories:

- **AIS (Automatic Identification System)** — vessel positions, speed, draft, vessel type
- **Weather** — wind speed, wave height, visibility
- **Customs** — processing times per cargo type and port
- **Operational** — berth occupancy, crane efficiency, truck queue lengths
- **Financial** — demurrage rates, fuel costs, perishability loss rates

After identifying the required sources, the team researched all available real-world data providers. **Every viable source** (port authorities, AIS aggregators, meteorological APIs, customs registries) required either **governmental permission** or **paid financial access** — neither of which was feasible for a prototype.

#### 1b. Synthetic Data Generation with Faker

Rather than stall the project, the team decided to **simulate the full database** using the Python package `faker`, combined with carefully researched real-world constraints.

**Seeds for reproducibility:** `Faker.seed(42)`, `np.random.seed(42)`, `random.seed(42)`

**Simulation parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Year | 2024 | Full leap year — 8,784 hourly slots |
| Number of ports | 6 | Egypt's 6 major export ports |
| Number of vessels | 100 | Realistic active fleet size for prototype |
| Target AIS records | 50,000 | Representative movement sample |

**Key simulation constraints applied:**

- **Port Status** derived from wind speed: `port_status = 1 if wind_speed_bft > 7 else 0` (Beaufort ≥ 8 = disruption)
- **Waiting time** conditioned on port status:
  ```python
  waiting_time = uniform(12, 48) hrs  # if port is disrupted
  waiting_time = uniform(0.5,  8) hrs  # if port is normal
  ```
- **Demurrage** conditioned on vessel type: Tanker/Container = $30K–$60K/day, others = $10K–$25K/day
- **Cargo categories**: Container, Bulk, Oil (port-level, not vessel-level)
- **Shift assignment**: Night = hour < 8, Morning = 8–15, Day = 16+
- **Holiday rate**: 5% of hours randomly flagged as holidays

> [Kaggle Notebook for Generating the Marine Database](https://www.kaggle.com/code/malakmohamed777/generating-the-marine-database)

#### 1c. Star Schema & Fact Table Assembly

Once the synthetic source tables were generated, they were integrated using a **star schema** data model:

```
             dim_vessels (100 rows)
                  |
  fact_port_conditions (52,704 rows)
         |                   |
dim_time (8,784 rows) -- fact_movements -- fact_port_operations (52,704 rows)
  (8,784 hrs × 6 ports)         |
                           df_costs (50,000 rows)
                                |
                          df_ports (6 rows)
```

**Source tables generated:**

| Table | Rows | Description |
|---|---|---|
| `dim_time` | 8,784 | One row per hour in 2024 (366 days × 24 hrs) |
| `df_ports` | 6 | Port IDs, names, coordinates |
| `dim_vessels` | 100 | Vessels with IMO, type, and draft |
| `fact_port_conditions` | 52,704 | Weather & port status per port per hour |
| `fact_port_operations` | 52,704 | Berth, crane, customs, cargo per port per hour |
| `df_ais` | 50,000 | Vessel movement events (randomly sampled) |
| `df_costs` | 50,000 | Financial record per movement |

All tables were merged on `time_key` + `port_key` + `movement_id` into the single master fact table.

**Final output — `fact_movments.csv`:**
- **50,000 rows** × **40 columns**, **0 missing values**
- All dimension and fact table features fully denormalised into one flat table
- Ready for direct consumption by analysis and modelling stages

**Saved source files:** `master_date.csv`, `dim_vessels.csv`, `df_ports.csv`, `dim_time.csv`, `fact_port_conditions.csv`, `fact_port_operations.csv`, `df_costs.csv`

---

### Stage 2 — Data Analysis

With the fact table ready, the team moved to exploratory and analytical work using the **Databricks** platform to build an interactive analytical dashboard.

Key insights extracted:
- Identification of peak congestion periods per port (seasonal and weekly patterns)
- Bottleneck analysis by cargo type — **Agricultural** cargo consistently showed higher waiting times due to customs sensitivity
- Port-level comparison of berth utilisation vs. average delay
- Correlation between weather severity (wind × wave height) and waiting time spikes
- Shift-based operational performance differences (Morning vs. Night shifts)

> [Databricks Analytical Dashboard](https://app.thebricks.com/file/34b8bb66-766c-4bd6-b71b-bd76e736255b)

---

### Stage 3 — Data Science

Three distinct model types were developed and trained on the fact table:

| Model | Type | Purpose | Output |
|---|---|---|---|
| **Random Forest** | Regression | Predict vessel waiting time from live conditions | Waiting hours (continuous) |
| **Prophet x6** | Time-Series | Forecast congestion per port over a future horizon | Daily waiting time forecast |
| **MaritimeGNN_v3** | Neural Network | Estimate waiting time with neighbour-port context | Normalised waiting time |

#### Random Forest Results

The data was first aggregated into `port_hourly` (32,321 rows — one record per port per hour), then split 80/20 by time index.

| Metric | Train | Test |
|---|---|---|
| R² Score | 0.9200 | 0.9039 |

#### GNN Results

Time-based split at the 80th percentile of the date range.

| Metric | Train | Test |
|---|---|---|
| R² Score | 0.8294 | 0.8245 |
| MAE | 3.75 hrs | 3.80 hrs |
| RMSE | 5.65 hrs | 5.75 hrs |
| Directional Accuracy (8 hr threshold) | 100% | 100% |
| R² Gap (overfitting check) | 0.0049 🟢 No overfitting | |

Early stopping triggered at **epoch 40** (patience=30, best test loss=0.1768).

#### Prophet Results (per port, evaluated on last 30 days)

| Port | MAE (hrs) | Within 80% CI | Trend Direction |
|---|---|---|---|
| Alexandria | 3.77 | 63.3% |
| Damietta | 2.46 | 83.3% |
| Dekheila | 3.72 | 60.0% |
| Port Said | 3.18 | 73.3% |
| Sokhna | 3.18 | 80.0% |
| Suez | 3.63 | 56.7% |

All models were serialised to files for the deployment stage:
- `prophet_model_{port}.json` — one per port, 6 total
- `rf_forecasting_model.joblib` — single global Random Forest
- `gnn_model_weights.pth` — GNN state dict
- `preprocessing_config.json` — scaler params, port mapping, feature lists

> [Kaggle Notebook for Training Models](https://www.kaggle.com/code/malakmohamed777/marine-models)

---

### Stage 4 — AI Engineering (Deployment)

Two parallel deployments were built from the trained models:

#### Deployment A — React / Node.js (Primary)

| Property | Detail |
|---|---|
| **Frontend** | React.js |
| **Backend** | Node.js |
| **Complexity** | High — full-stack, production-grade architecture |
| **Stability** | More robust and scalable |
| **Availability** | Local only — not yet publicly hosted |
| **Status** | Functional internally; publication pending |

This is the main intended production interface. It provides a richer, more stable user experience but requires a local environment to run.

#### Deployment B — Streamlit / Python on Hugging Face (Prototype)

| Property | Detail |
|---|---|
| **Framework** | Streamlit (Python) |
| **Host** | Hugging Face Spaces |
| **Complexity** | Lower — single-file Python app |
| **Availability** | Publicly accessible online |
| **Status** | Live and deployed |

This is the publicly accessible prototype documented in detail throughout this file. It is suitable for demonstration and testing purposes.

> [Live Hugging Face Space (Streamlit App)](https://huggingface.co/spaces/MalakMohamed21/Marine_Recommendation_System)


---

### Stage 5 — Presentation & Collaboration

#### Canva Presentation
A professional slide deck was created in **Canva** covering:
- The problem and its impact on Egyptian exports
- The proposed solution and data pipeline
- Model architectures and key results
- The full workflow from data generation to deployment

#### GitHub Repository
All project files (notebooks, models, data, and app code) are centralised in a shared **GitHub repository** used for collaboration and version control across the team.

---

### Workflow Summary

```
[Stage 1] Data Engineering
  Problem Research --> Faker Simulation --> Star Schema --> fact_movments.csv (50k rows)
         |
[Stage 2] Data Analysis
  Databricks Dashboard --> Key Insights (bottlenecks, seasonality, cargo patterns)
         |
[Stage 3] Data Science
  Random Forest + 6x Prophet + RoutingGNN --> Serialised model files
         |
[Stage 4] AI Engineering
  React/Node.js (local) + Streamlit/HuggingFace (online) --> Live Deployments
         |
[Stage 5] Presentation
  Canva Deck + GitHub Repo
```

---

## 4. System Architecture

```
Data Sources
  AIS Vessel Tracking | Weather API | Customs Records
           |
           v
Unified Fact Table (fact_movments.csv)
  Vessels . Weather . Customs . Trucking — joined schema
           |
     +-----+-----+
     |     |     |
  Prophet  RF   GNN
     |     |     |
     +-----+-----+
           |
  Streamlit Dashboard (4 Pages)
  Historical | Forecast | Predictor | Optimizer
```

**Full Technology Stack:**

| Layer | Technology |
|---|---|
| Dashboard UI | Streamlit + custom CSS (dark theme, Syne & JetBrains Mono fonts) |
| Visualizations | Plotly Express + Plotly Graph Objects |
| Time-Series Forecasting | Meta Prophet |
| Delay Prediction | Scikit-learn Random Forest |
| Spatial Routing | PyTorch (custom RoutingGNN) |
| Data Processing | Pandas, NumPy |
| Model Persistence | Prophet JSON, Joblib (RF), PyTorch .pth (GNN) |

---

## 5. File Structure

```
project/
|
+-- streamlit_app.py              # Main application — all 4 pages
|
+-- fact_movments.csv             # Primary dataset (AIS + Weather + Customs joined)
|
+-- prophet_model_Alexandria.json # Prophet model — Alexandria
+-- prophet_model_Damietta.json   # Prophet model — Damietta
+-- prophet_model_Dekheila.json   # Prophet model — Dekheila
+-- prophet_model_Port Said.json  # Prophet model — Port Said
+-- prophet_model_Sokhna.json     # Prophet model — Sokhna
+-- prophet_model_Suez.json       # Prophet model — Suez
|
+-- rf_forecasting_model.joblib   # Trained Random Forest regressor
+-- gnn_model_weights.pth         # Trained RoutingGNN state dict
|
+-- preprocessing_config.json     # Preprocessing pipeline config
+-- requirements.txt              # Python dependencies
```

> All model and data files must reside in the **same directory** as `streamlit_app.py`.

---

## 6. Installation & Setup

### Prerequisites
- Python 3.9+

### Install dependencies

```bash
pip install -r requirements.txt
```

Minimum required packages:

```
streamlit
pandas
numpy
plotly
prophet
joblib
scikit-learn
torch
```

### Run locally

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`.

### Deploy to Hugging Face Spaces

1. Push all files to a Hugging Face Space repository.
2. Set the Space SDK to **Streamlit**.
3. Ensure `requirements.txt` is present.

> `fact_movments.csv` (13.3 MB) and `rf_forecasting_model.joblib` (10.8 MB) are tracked with Git LFS via the `xet` storage backend.

---

## 7. Data Schema

The full column-level schema for all tables (`fact_movments.csv` and all source dimension tables) is covered in the project's **Data Catalog**. Refer to that document for data types, descriptions, value ranges, and nullability for every field.

For high-level context: the final fact table contains **50,000 rows × 40 columns** spanning vessel movement, weather, port operations, customs, and financial data — all for the year 2024 across Egypt's 6 major ports.

---

## 8. Dashboard Pages

### Page 1 — Historical Logistics Dashboard

**Purpose:** Full-dataset retrospective overview.

**KPI Cards:**

| Metric | Calculation |
|---|---|
| Avg Waiting Time | mean(waiting_time_hours) |
| Total Demurrage Cost | sum(daily_demurrage_cost) in $M |
| Avg Berth Occupancy | mean(berth_occupancy_rate) |
| Unique Vessels | nunique(vessel_imo) |

**Charts:**

| Chart | Description |
|---|---|
| Vessel Positions Map | Up to 2,000 sampled positions, coloured by waiting time (Plasma scale) |
| Avg Waiting Time by Port | Horizontal bar, grouped by port_id, mapped via PORT_ID_MAP |
| Monthly Trend | Line chart of average wait per calendar month+year |
| By Cargo Category | Bar chart of average wait per cargo type |
| Raw Data Browser | Expandable table, first 500 rows, heat-mapped on waiting_time_hours |

---

### Page 2 — Congestion Forecast (Prophet)

**Purpose:** Forward-looking hourly congestion forecast for a selected port.

**User Controls:**

| Control | Range | Default |
|---|---|---|
| Port selector | 6 ports | Alexandria |
| Forecast Horizon | 7–90 days | 30 days |

**Output:** Plotly chart with three traces:
1. Forecast line (yhat) — cyan
2. Confidence interval band (yhat_lower / yhat_upper) — semi-transparent fill
3. Historical actuals — amber scatter, filtered by **all** port IDs via PORT_ID_MAP

**Summary Metrics:** Peak, Average, and Minimum predicted wait over the forecast window.

---

### Page 3 — Operational Delay Predictor (Random Forest)

**Purpose:** Real-time estimate of vessel waiting time given current conditions.

**User Inputs:** 9 sliders + 4 dropdowns (port, cargo, vessel type, shift).

**Output:**
- Predicted waiting time in hours
- Port status badge (🟢 CLEAR / 🟡 BUSY / 🔴 CRITICAL)
- Perishability Loss Rate estimate
- Weather Risk Score
- Gauge chart (0–48 hrs range)

**Status thresholds:**

| Status | Threshold |
|---|---|
| 🟢 CLEAR | prediction < 4 hrs |
| 🟡 BUSY | 4 hrs <= prediction <= 8 hrs |
| 🔴 CRITICAL | prediction > 8 hrs |

---

### Page 4 — Alternative Routing Optimizer (GNN)

**Purpose:** Rank alternative ports by predicted waiting time and compute net economic benefit of diversion.

**User Controls:** Congested port, cargo type, vessel speed, urgency level.

**For each of the top 3 ranked alternatives:**

| Field | Calculation |
|---|---|
| Predicted Wait | GNN output (or historical mean fallback) |
| Distance (km) | Haversine great-circle distance from origin |
| Travel Time | distance_km / (speed_knots x 1.852) |
| Fuel Cost | distance_km x avg_fuel_cost x urgency_multiplier |
| Net Saving | demurrage_saved - fuel_cost |

**Urgency multipliers:** Low=0.8x, Medium=1.0x, High=1.3x, Critical=1.6x

---

## 9. ML Models

### 9.1 Prophet Time-Series Model

- **One model per port — 6 total**
- **Training data granularity:** Daily averages (aggregated from hourly `port_hourly` table)
- **Training data:** 366 days per port (full year 2024)
- **Evaluation:** Last 30 days held out
- **Forecast horizon in app:** Up to 90 days ahead (user-controlled, hourly frequency)
- **Serialization:** JSON string via `prophet.serialize.model_to_json()` / `model_from_json()`

**Prophet Configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| `yearly_seasonality` | True | Captures annual export cycles |
| `weekly_seasonality` | True | Captures day-of-week traffic patterns |
| `daily_seasonality` | False | Data already aggregated to daily |
| `seasonality_mode` | `'multiplicative'` | Better for shipping spikes (not additive shifts) |
| `changepoint_prior_scale` | 0.05 | Conservative — prevents overfitting trend changes |
| `interval_width` | 0.80 | 80% confidence interval |

**Evaluation Results (last 30 days per port):**

| Port | MAE (hrs) | Within 80% CI | Trend Direction |
|---|---|---|---|
| Alexandria | 3.77 | 63.3% |
| Damietta | 2.46 | 83.3% |
| Dekheila | 3.72 | 60.0% |
| Port Said | 3.18 | 73.3% |
| Sokhna | 3.18 | 80.0% |
| Suez | 3.63 | 56.7% |

> **Critical API note:** `model_from_json()` expects a raw JSON **string** (`f.read()`), not a Python dict (`json.load(f)`). Using `json.load()` causes a TypeError.

**Seasonal patterns captured:**
- Weekly: vessel traffic patterns by day of week
- Annual: agricultural export seasons (citrus peaks in winter)

---

### 9.2 Random Forest Delay Predictor

- **File:** `rf_forecasting_model.joblib`
- **Task:** Regression — predict `waiting_time_hours`
- **Training data:** `port_hourly` aggregated table — **32,321 rows** (one per port per hour)
- **Train/test split:** First 80% of rows by time index (time-based, not random shuffle)
- **Features:** 31 total (see Feature Engineering section)

**Hyperparameters:**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 10 |
| `min_samples_leaf` | 15 |
| `random_state` | 42 |
| `n_jobs` | -1 (all CPU cores) |

**Evaluation Results:**

| Split | R² Score |
|---|---|
| Train | 0.9200 |
| Test | 0.9039 |

**Label Encoding order (alphabetical — must match training):**

| Encoder | Sorted Categories | Values |
|---|---|---|
| vessel_type | Bulk Carrier, Container Ship, General Cargo, Tanker | 0, 1, 2, 3 |
| cargo_category | Agricultural, Bulk, Container, General, Oil | 0, 1, 2, 3, 4 |
| shift_name | Day, Morning, Night | 0, 1, 2 |

---

### 9.3 Graph Neural Network Router (MaritimeGNN_v3)

- **Class name (training):** `MaritimeGNN_v3` — saved as `gnn_model_weights.pth`
- **Task:** Regression — predict normalised vessel waiting time
- **Input:** 54 features per sample = 27 own-port features + 27 neighbour-averaged features
- **Output:** 1 scalar (normalised), denormalised at inference

**Architecture (as trained — Dropout values are inactive during eval()):**

```
net.0  Linear(54 -> 64)
net.1  BatchNorm1d(64)
net.2  ReLU
net.3  Dropout(p=0.4)      <- increased from 0.2 to reduce overfitting
net.4  Linear(64 -> 32)
net.5  BatchNorm1d(32)
net.6  ReLU
net.7  Dropout(p=0.3)      <- second dropout added
net.8  Linear(32 -> 1)     <- direct output (32→16→1 collapsed to 32→1)
```

**Training Configuration:**

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 1e-3 |
| Batch size | 256 |
| Max epochs | 500 |
| Early stopping patience | 30 epochs |
| Stopped at epoch | 40 |
| Best test loss | 0.1768 |

**Train/Test Split (time-based):**

| Split | Rows | Period |
|---|---|---|
| Train | 25,828 | 2024-01-01 → 2024-10-19 |
| Test | 6,457 | 2024-10-19 → 2024-12-31 |

**Graph Construction:** Adjacency matrix built from Gaussian RBF kernel on geodesic distances between port coordinates (`sigma = dist_matrix.std()`). Edges with weight < 0.1 are pruned.

**Evaluation Results:**

| Metric | Train | Test |
|---|---|---|
| R² Score | 0.8294 | 0.8245 |
| MAE | 3.75 hrs | 3.80 hrs |
| RMSE | 5.65 hrs | 5.75 hrs |
| Directional Accuracy (8 hr threshold) | 100% | 100% |
| R² Gap | 0.0049 — 🟢 No significant overfitting | |

**Denormalisation formula:**
```python
# Statistics computed from TRAIN split only
y_mean_train = 13.65  # hrs
y_std_train  = 13.68  # hrs
predicted_hours = max(0.2, (model_output * y_std) + y_mean)
```

**Fallback:** If the model fails to load, the optimizer uses historical mean waiting_time_hours per port from the CSV.

---

## 10. Port Reference Data

### Port IDs in the Dashboard (`PORTS` dict)

| Port Name | Primary ID | Latitude | Longitude |
|---|---|---|---|
| Alexandria | EG_ALX | 31.2001 | 29.9187 |
| Dekheila | EG_DKH | 31.1333 | 29.8000 |
| Damietta | EG_DMT | 31.4175 | 31.8144 |
| Port Said | EG_PSD | 31.2653 | 32.3019 |
| Suez | EG_SUZ | 29.9667 | 32.5500 |
| Sokhna | EG_SKH | 29.6667 | 32.3333 |

### Port IDs as generated in `fact_movments.csv`

The data generation notebook assigned these IDs, which differ from the dashboard primary IDs:

| Port ID in CSV | GNN id_map resolves to | Dashboard canonical name |
|---|---|---|
| `EG_ALY` | Dekheila | Alexandria / Dekheila area |
| `EG_PSD` | Port Said | Port Said |
| `EG_DMT` | Damietta | Damietta |
| `EG_SOK` | Sokhna | Sokhna |
| `EG_SAF` | Alexandria | Alexandria |
| `EG_SUZ` | Suez | Suez |

### `PORT_ID_MAP` — Full alternate ID resolution

| Raw ID in CSV | Resolved Name |
|---|---|
| EG_ALX, EG_ALY | Alexandria |
| EG_DKH | Dekheila |
| EG_DMT | Damietta |
| EG_PSD, EG_SAF | Port Said |
| EG_SUZ | Suez |
| EG_SKH, EG_SOK | Sokhna |

> Always filter using `df["port_id"].isin(all_ids)` with all IDs for a given port name — not just the primary one — to avoid silently dropping data.

---

## 11. Configuration Constants

| Constant | App Value | Actual Training Value | Description |
|---|---|---|---|
| `GNN_Y_MEAN` | 13.74 | 13.65 | Mean waiting time used to normalise the GNN target |
| `GNN_Y_STD` | 11.32 | 13.68 | Std deviation used to normalise the GNN target |

> ⚠️ **Discrepancy:** The constants in `streamlit_app.py` (`GNN_Y_MEAN=13.74`, `GNN_Y_STD=11.32`) differ slightly from the values computed during training (`y_mean=13.65`, `y_std=13.68`). The `y_std` difference (11.32 vs 13.68) will scale denormalised predictions by a factor of ~0.83. If you retrain the GNN, update both constants from `preprocessing_config.json` → `gnn_normalization` → `y_mean` / `y_std`.

Update both if the GNN is retrained.

---

## 12. Feature Engineering

### Three composite features fed into the Random Forest:

**Weather Severity**
```python
weather_severity = wind_speed_bft x wave_height
```
Combined physical stress on port operations.

**Weather Visibility Index**
```python
weather_visibility = weather_severity / (visibility_km + 0.1)
```
Weights severity against visibility. +0.1 prevents division by zero.

**Operational Efficiency**
```python
ops_efficiency = crane_efficiency / (berth_occupancy_rate + 1)
```
Throughput-to-demand ratio. Low values indicate bottlenecks. +1 prevents division by zero.

### Cyclical Time Encoding
```python
hour_sin = sin(2*pi * hour / 24)
hour_cos = cos(2*pi * hour / 24)
month_sin = sin(2*pi * month / 12)
month_cos = cos(2*pi * month / 12)
```
At live inference, these are fixed to noon (hour=12) and June (month=6) as neutral defaults.

---
