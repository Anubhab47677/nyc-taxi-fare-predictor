# 🚕 NYC Taxi Fare AI Predictor

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Leaflet.js](https://img.shields.io/badge/Leaflet.js-1.9.4-199900.svg?logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC.svg?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

An end-to-end Machine Learning and Full-Stack Web Application that predicts New York City taxi fares in real time. Built using a Deep Neural Network (DNN) trained on NYC TLC trip records and deployed via a Flask REST API with an interactive geospatial Leaflet.js dashboard.

---

## 📸 Dashboard Preview

### Interactive Map & Trip Configuration
![Dashboard Overview](assets/dashboard-preview.png)

### Real-Time Fare & Surcharge Calculation
![Fare Prediction Result](assets/route-selection.png)

---

## ✨ Key Features

* **Interactive Geospatial Map:** Click anywhere on the map or drag the **Pickup** and **Dropoff** markers to dynamically update trip coordinates and flight-path polylines in real time.
* **Geodesic Distance Engineering:** Computes exact great-circle distance (in miles and kilometers) using the **Haversine Formula**.
* **Context-Aware Surcharge Logic:** Automatically detects NYC peak weekday rush hours (4:00 PM – 8:00 PM), overnight surcharges (8:00 PM – 6:00 AM), and flat-rate JFK Airport zone proximity.
* **Deep Learning Pipeline:** Includes the complete exploratory data analysis (EDA), outlier filtering, feature scaling, and Keras Sequential DNN training notebook.

---

## 🧠 Machine Learning Pipeline Overview

| Component | Details |
| :--- | :--- |
| **Dataset** | Kaggle NYC Taxi Fare Prediction (`100,000` sampled records) |
| **Target Variable** | `fare_amount` (Continuous regression target, cleaned for $> \$0$) |
| **Geospatial Filtering** | Bounded to valid NYC coordinates ($\text{Lat } 40.5^\circ\text{–}40.9^\circ, \text{Lon } -74.25^\circ\text{–}-73.7^\circ$) |
| **Model Architecture** | Keras Sequential DNN (`Dense`, `BatchNormalization`, `Dropout`) |
| **Evaluation Metrics** | Mean Absolute Error (MAE), Mean Squared Error (MSE), $R^2$ Score |

---

## 🏗️ Project Architecture

```text
nyc_taxi_app/
├── assets/                   # UI screenshots for documentation
│   ├── dashboard-preview.png
│   └── route-selection.png
├── templates/
│   └── index.html            # Responsive TailwindCSS + Leaflet.js frontend
├── app.py                    # Flask backend & geospatial feature inference API
├── NYC_Taxi_Fare_Prediction.ipynb # End-to-end data preprocessing & DNN model notebook
├── requirements.txt          # Python environment dependencies
├── .gitignore                # Git exclusion rules
└── README.md                 # Project documentation