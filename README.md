<div align="center">

# 🚕 NYC Taxi Fare AI Predictor

### End-to-End Deep Learning Regression & Interactive Geospatial Web Application

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Sequential_DNN-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Leaflet.js](https://img.shields.io/badge/Leaflet.js-1.9.4-199900?style=for-the-badge&logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

*An intelligent full-stack predictive analytics platform combining Deep Neural Networks, geodesic spatial feature engineering, and NYC Taxi & Limousine Commission (TLC) tariff modeling inside a real-time interactive map interface.*

[Project Overview](#-project-overview) • [Dashboard Preview](#-dashboard-preview) • [Core Capabilities](#-core-capabilities--features) • [System Architecture](#-system-architecture--data-flow) • [ML & Mathematical Pipeline](#-machine-learning--mathematical-pipeline) • [Tech Stack](#-complete-technology-stack) • [Installation](#-installation--local-deployment) • [API Documentation](#-rest-api-documentation)

</div>

---

## 📋 Project Overview

Estimating urban taxi fares accurately requires modeling complex non-linear interactions between **geospatial coordinates**, **temporal traffic patterns**, **regulatory surcharges**, and **airport flat-rate zones**. Simple straight-line distance approximations often fail to account for New York City's dynamic borough boundaries, rush-hour congestion tariffs, and major transit hubs.

**NYC Taxi Fare AI Predictor** solves this problem through a two-tier architecture:
1. **Offline Deep Learning Research Pipeline (`NYC_Taxi_Fare_Prediction.ipynb`):** Performs large-scale data ingestion, statistical anomaly pruning, spatial-temporal feature extraction, and supervised regression training using a multi-layer Keras Deep Neural Network (DNN).
2. **Real-Time Full-Stack Inference Engine (`app.py` & `templates/index.html`):** Exposes a low-latency Flask REST API paired with an interactive Leaflet.js map dashboard, enabling users to drop or drag pickup and drop-off pins across New York City and receive instant fare, distance, and surcharge breakdowns.

---

## 📸 Dashboard Preview

<div align="center">

### 🗺️ Interactive Geospatial Map & Trip Parameter Controls
*Users can click anywhere on the OpenStreetMap canvas or drag markers dynamically to set exact coordinates, passenger counts, and pickup timestamps.*

<img src="assets/dashboard-preview.png" alt="NYC Taxi Fare AI Predictor - Interactive Map Dashboard" width="92%"/>

<br/><br/>

### 💵 Real-Time Fare Estimation & Surcharge Breakdown
*Instantaneous calculation displaying total estimated fare, geodesic trip distance in miles, and active NYC TLC peak/overnight surcharges.*

<img src="assets/route-selection.png" alt="NYC Taxi Fare AI Predictor - Route & Fare Calculation" width="92%"/>

</div>

---

## 🌟 Core Capabilities & Features

### 1. Interactive Geospatial Mapping (`Leaflet.js` + `OpenStreetMap`)
- **Dual-Mode Coordinate Selection:** Alternate-click directly on the map canvas to place **Pickup** and **Dropoff** markers, or click-and-drag existing pins for fine-grained street-level adjustments.
- **Live Vector Polyline Rendering:** Automatically draws and updates a dashed trajectory vector connecting origin and destination coordinates on every drag or click event.
- **One-Click State Reset:** Instantly restores default Manhattan (Times Square) to JFK Airport coordinates and recenters the viewport.

### 2. Geodesic Spatial Engineering Engine
- **Great-Circle Distance Computation:** Calculates exact spherical surface distance between pickup and drop-off latitude/longitude pairs in both kilometers and miles.
- **Airport Hub Geofencing:** Evaluates proximity against coordinates for **John F. Kennedy International Airport (JFK)**, **LaGuardia Airport (LGA)**, and **Newark Liberty International Airport (EWR)** to trigger flat-rate airport tariff logic when within a `2.0 km` radius.

### 3. Dynamic NYC TLC Tariff & Temporal Surcharge Logic
- **Weekday Rush-Hour Detection:** Parses ISO-8601 datetime strings to detect weekday peak congestion windows (`Monday–Friday, 4:00 PM – 8:00 PM`) and applies the standard `$2.50` peak surcharge.
- **Overnight Tariff Evaluation:** Identifies late-night and early-morning trips (`8:00 PM – 6:00 AM`) and factors in the `$1.00` overnight surcharge.
- **Base & Distance Metering:** Combines a `$3.00` initial flag-drop base fare with `$3.30/mile` metered progression and a `$2.50` minimum fare floor.

### 4. Reproducible Deep Learning Training Pipeline
- **Deterministic Execution:** Enforces global random seed `42` across `numpy`, `random`, and `tensorflow` for reproducible experiments.
- **Comprehensive Outlier Sanitization:** Eliminates negative/zero fares, unpopulated passenger entries, and corrupted GPS coordinates outside the NYC metropolitan bounding box.

---

## 🏗️ System Architecture & Data Flow

The application follows a decoupled **Client–Server–Inference** architecture designed for fast asynchronous JSON communication without page reloads:

```text
+-----------------------------------------------------------------------------------+
|                        CLIENT LAYER (Browser / Frontend)                          |
|                                                                                   |
|  +-------------------------------------+   +-----------------------------------+  |
|  |     Interactive Leaflet.js Map      |   |       Trip Configuration Form     |  |
|  |  • Draggable Pickup/Dropoff Pins    |   |  • ISO Datetime Picker            |  |
|  |  • Click-to-Place Event Listeners   |   |  • Passenger Count Selector (1-5) |  |
|  |  • Dynamic Geodesic Polyline        |   |  • Live Lat/Lon Coordinate Sync   |  |
|  +------------------+------------------+   +-----------------+-----------------+  |
|                     |                                        |                    |
|                     +-------------------+--------------------+                    |
|                                         |                                         |
|                            Async Fetch API (POST /predict)                        |
|                            Content-Type: application/json                         |
+-----------------------------------------|-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      APPLICATION LAYER (Flask Backend - app.py)                   |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  |                        Request Validation & Parsing                         |  |
|  |  • Extracts float coordinates, integer passenger count, & datetime object   |  |
|  +--------------------------------------+--------------------------------------+  |
|                                         |                                         |
|                                         v                                         |
|  +--------------------------------------+--------------------------------------+  |
|  |                   Feature Engineering & Tariff Engine                       |  |
|  |  1. Haversine Formula -> Great-Circle Distance (km & miles)                 |  |
|  |  2. Airport Proximity -> Min distance to JFK (40.6413, -73.7781) & LGA      |  |
|  |  3. Temporal Parser   -> Weekday Peak (16:00-20:00) & Night (20:00-06:00)   |  |
|  +--------------------------------------+--------------------------------------+  |
|                                         |                                         |
|                                         v                                         |
|  +--------------------------------------+--------------------------------------+  |
|  |                     Prediction & Response Serializer                        |  |
|  |  • Computes final metered/flat-rate fare & returns structured JSON payload  |  |
|  +--------------------------------------+--------------------------------------+  |
+-----------------------------------------|-----------------------------------------+
                                          |
                              JSON Response (HTTP 200)
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     UI PRESENTATION LAYER (Tailwind CSS Card)                     |
|  • Updates Predicted Fare ($), Trip Distance (mi), and Surcharge Breakdown ($)    |
+-----------------------------------------------------------------------------------+
