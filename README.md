<div align="center">

# 🚕 NYC Taxi Fare AI Predictor

**An end-to-end Deep Learning & Geospatial Web Application for real-time New York City taxi fare estimation.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Leaflet.js](https://img.shields.io/badge/Leaflet.js-1.9.4-199900?style=for-the-badge&logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

[Features](#-key-features) • [Dashboard Preview](#-dashboard-preview) • [System Architecture](#-system-architecture) • [ML Pipeline](#-machine-learning-pipeline) • [Installation](#-installation--setup) • [API Reference](#-api-reference)

</div>

---

## 📖 Overview

**NYC Taxi Fare AI Predictor** bridges deep neural regression modeling with an interactive geospatial user interface. Trained on New York City Taxi & Limousine Commission (TLC) trip records, the system engineers spatial and temporal features—including geodesic Haversine distances, airport proximity zones, and rush-hour indicators—to predict accurate trip fares in real time via a lightweight Flask REST API.

---

## 📸 Dashboard Preview

<div align="center">

### 🗺️ Interactive Geospatial Map & Trip Configuration
<img src="assets/dashboard-preview.png" alt="NYC Taxi Fare Predictor Dashboard" width="90%"/>

<br/>

### 💵 Real-Time Fare Breakdown & Surcharge Analysis
<img src="assets/route-selection.png" alt="NYC Taxi Fare Calculation Output" width="90%"/>

</div>

---

## ✨ Key Features

- **📍 Interactive Pin-Drop & Drag Map:** Powered by **Leaflet.js** and OpenStreetMap. Click anywhere on the NYC map or drag the **Pickup** and **Dropoff** markers to dynamically recalculate coordinates and flight-path polylines in real time.
- **🌐 Geodesic Feature Engineering:** Computes exact great-circle trip distances (in both miles and kilometers) using the **Haversine Formula**, alongside proximity detection for major NYC hubs (**JFK**, **LaGuardia**, and **Newark**).
- **⏱️ Dynamic Temporal & Surcharge Engine:** Automatically evaluates pickup timestamps to factor in NYC TLC pricing rules, including weekday peak rush-hour surcharges (`4:00 PM – 8:00 PM`), overnight rates (`8:00 PM – 6:00 AM`), and JFK flat-rate zones.
- **🧠 Deep Neural Network (DNN) Pipeline:** Includes the complete Jupyter Notebook covering Exploratory Data Analysis (EDA), coordinate anomaly filtering, feature scaling, and Keras Sequential regression training.
- **⚡ Responsive Dark-Mode UI:** Built with **Tailwind CSS** and **Lucide Icons** for a clean, low-latency user experience across desktop and mobile viewports.

---

## 🏗️ System Architecture

```text
┌───────────────────────────┐       POST /predict (JSON)       ┌──────────────────────────────────┐
│   Interactive Frontend    │ ───────────────────────────────► │      Flask Inference Server      │
│  (Leaflet.js + Tailwind)  │                                  │            (app.py)              │
└───────────────────────────┘                                  └────────────────┬─────────────────┘
              ▲                                                                 │
              │                                                                 ▼
              │                                                ┌──────────────────────────────────┐
              │                                                │  Geospatial & Temporal Engine    │
              │                                                │  • Haversine Distance (km / mi)  │
              │                                                │  • JFK / LGA / EWR Proximity     │
              │                                                │  • Rush Hour & Night Surcharges  │
              │                                                └────────────────┬─────────────────┘
              │                                                                 │
              │                    JSON Prediction Response                     ▼
              └─────────────────────────────────────────────── ┌──────────────────────────────────┐
                                                               │   Keras DNN Regression Pipeline  │
                                                               │  (Dense + BatchNorm + Dropout)   │
                                                               └──────────────────────────────────┘
