# SkyGuard AI

**SIH26073 — AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)**

SkyGuard AI is an intelligent real-time quality-control and anomaly-detection system for Automatic Weather Station observations. It analyzes **temperature (°C), atmospheric pressure (hPa), and relative humidity (%)** to identify abnormal, inconsistent, or potentially faulty observations while reducing false alarms and distinguishing sensor/data problems from genuine meteorological events.

## Problem

AWS networks are essential for weather forecasting, climate monitoring, disaster management, aviation, agriculture, and scientific research. Their observations can be affected by sensor malfunction, calibration drift, communication failures, power fluctuations, harsh environmental conditions, and data corruption. Simple fixed-threshold quality checks are often insufficient because atmospheric variables are time-dependent and interrelated.

## Proposed Solution

SkyGuard AI combines data preprocessing, time-series analysis, statistical quality checks, machine-learning anomaly detection, **contextual analysis**, and explainability into a real-time pipeline.

### Core inputs
- Temperature
- Atmospheric pressure
- Relative humidity
- Timestamp / temporal ordering

### Core capabilities
- Real-time anomaly detection
- Univariate and multivariate anomaly analysis
- Detection of spikes, drops, stuck sensors, drift, missing/corrupt observations, and inconsistent combinations of variables
- **Context-aware analysis to distinguish genuine meteorological events from sensor/data anomalies**
- Anomaly severity/confidence scoring
- Explainable alerts
- Visualization and monitoring dashboard
- Scalable architecture for multiple AWS stations

## Planned Architecture

```text
AWS Data
   ↓
Ingestion / Validation
   ↓
Preprocessing & Cleaning
   ↓
Feature Engineering
   ↓
Anomaly Detection Engine
   ├── Statistical / Rule-based QC
   ├── Time-series methods
   └── ML-based detection
   ↓
Contextual Analysis
   ├── Temporal context
   ├── Cross-variable relationships
   ├── Rate-of-change & persistence
   ├── Meteorological plausibility
   └── Historical/local patterns
   ↓
Context-Aware Decision Layer
   ├── Genuine meteorological event
   ├── Sensor fault
   ├── Data/communication issue
   └── Uncertain
   ↓
Anomaly Score + Classification + Explanation
   ↓
Dashboard / Alerts / API
```

### Why contextual analysis matters

An unusual observation is not automatically a faulty observation. For example, a sudden temperature change may be a genuine meteorological event or a sensor problem. Contextual analysis examines the observation in relation to its surrounding time window, the behavior of the other AWS variables, persistence, rate of change, and meteorological plausibility before the system determines how an anomaly should be classified.

## Project Structure

```text
SkyGuard-AI/
├── data/
│   ├── raw/              # Original datasets (do not commit sensitive/large data)
│   ├── processed/        # Cleaned/feature-engineered data
│   └── sample/           # Small sample datasets for development
├── notebooks/            # Exploration and experiments
├── src/
│   ├── data/             # Ingestion and preprocessing
│   ├── features/         # Feature engineering
│   ├── detection/        # Anomaly detection algorithms
│   ├── contextual/       # Contextual analysis and event/fault classification
│   ├── evaluation/       # Metrics and evaluation utilities
│   └── explainability/   # SHAP/LIME or other explanation utilities
├── app/                  # Prototype dashboard/API
├── tests/                # Automated tests
├── docs/                 # Technical and project documentation
├── requirements.txt
├── .gitignore
└── README.md
```

## Development Principles

1. Start with a transparent baseline before introducing complex ML.
2. Avoid training/evaluating models on contaminated data without a clear methodology.
3. Evaluate false positives as carefully as anomaly recall.
4. Preserve temporal ordering when validating time-series models.
5. **Do not classify an unusual observation as a sensor fault without considering its context.**
6. Treat genuine extreme weather events differently from sensor/data faults.
7. Make every alert explainable enough for an operator to investigate.

## Team Goal

Build a working, demonstrable prototype for Smart India Hackathon 2026 that clearly shows the complete pipeline from AWS observations to anomaly detection, contextual interpretation, explanation, and operator-facing visualization.

## SIH Reference

- **Problem ID:** SIH26073
- **Project:** SkyGuard AI
- **Domain:** Artificial Intelligence / Machine Learning / Meteorological Data Quality Control
