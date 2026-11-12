<div align="center">

# 🚕 NYC Taxi Fare AI Predictor

### Deep Learning Regression Research & Interactive Geospatial Fare Estimation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Sequential_DNN-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Leaflet](https://img.shields.io/badge/Leaflet.js-1.9.4-199900?style=for-the-badge&logo=leaflet&logoColor=white)](https://leafletjs.com/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](#license)

**An end-to-end NYC taxi fare estimation project** combining geospatial feature engineering, a TensorFlow/Keras regression workflow, and an interactive Flask web application.

[Overview](#-project-overview) · [Features](#-features) · [Screenshots](#-screenshots) · [Architecture](#-system-architecture) · [Machine Learning](#-machine-learning-workflow) · [API](#-api-documentation) · [Installation](#-installation) · [Roadmap](#-roadmap)

</div>

---

## 📋 Project Overview

Estimating taxi fares involves more than measuring the distance between two points. Pickup and drop-off locations, trip timing, passenger information, airport zones, and local fare rules can all affect the final amount.

**NYC Taxi Fare AI Predictor** explores this problem through two connected parts:

1. **Machine Learning research pipeline — `NYC_Taxi_Fare_Prediction.ipynb`**  
   A Jupyter Notebook for dataset inspection, cleaning, exploratory analysis, geospatial and temporal feature engineering, and training/evaluating a Keras Deep Neural Network (DNN) regression model.

2. **Interactive web application — `app.py`**  
   A Flask application with a Leaflet.js map interface. Users can select pickup and drop-off coordinates, specify trip details, and request a fare estimate through a JSON API.

> **Implementation note:** The notebook contains the DNN training workflow. The documented Flask prediction flow currently performs geodesic distance and tariff/surcharge calculations. Exporting and integrating the trained model and fitted scaler into the live application is listed as a future enhancement.

## ✨ Features

### 1. Interactive map

- Select pickup and drop-off locations directly on an OpenStreetMap-powered map.
- Drag existing markers to adjust trip coordinates.
- Display a route line between the selected points.
- View and synchronize pickup/drop-off latitude and longitude values.
- Reset the map to the default Times Square-to-JFK example route.

### 2. Geospatial calculations

- Calculate great-circle distance using the Haversine formula.
- Display trip distance in both kilometres and miles.
- Check proximity to JFK, LaGuardia (LGA), and Newark Liberty (EWR) airport coordinates.
- Apply the documented airport proximity rule when a trip endpoint is within the configured 2 km radius.

### 3. Fare and surcharge estimation

The application includes the following fare components in its documented tariff calculation:

| Component | Amount / Rule |
|---|---:|
| Base fare | $3.00 |
| Distance rate | $3.30 per mile |
| Minimum fare | $2.50 |
| Weekday peak surcharge | $2.50 |
| Overnight surcharge | $1.00 |
| Airport proximity radius | 2.0 km |

- **Weekday peak period:** Monday–Friday, 4:00 PM–8:00 PM.
- **Overnight period:** 8:00 PM–6:00 AM.
- The API response provides the estimated fare, trip distance, pickup time, passenger count, and surcharge details.

*These values describe the tariff rules implemented/documented in this project; verify current official NYC TLC rates before using the estimate for real-world pricing.*

### 4. Machine Learning workflow

- Reproducible experiment setup using random seed `42`.
- Dataset cleaning and invalid-record filtering.
- Geospatial and time-based feature engineering.
- Keras Sequential DNN regression model.
- Evaluation using MAE, MSE, and R².
- Training callbacks including EarlyStopping and ReduceLROnPlateau.

## 📸 Screenshots

<div align="center">

### Interactive dashboard and map

<img src="assets/dashboard-preview.png" alt="NYC Taxi Fare AI Predictor dashboard showing the interactive map" width="92%">

### Route selection and fare estimate

<img src="assets/route-selection.png" alt="Selected taxi route and fare estimation result" width="92%">

</div>

## 🏗️ System Architecture

The project separates model experimentation from web-based fare estimation.

```text
                         MACHINE LEARNING WORKFLOW
                 NYC_Taxi_Fare_Prediction.ipynb
                               |
                 Dataset loading and exploration
                               |
                    Cleaning and preprocessing
                               |
              Geospatial and temporal feature work
                               |
                    DNN training and evaluation
                               |
                   Model export (planned integration)


                           WEB APPLICATION
                               |
                 Browser: HTML / Tailwind / Leaflet
                               |
                  User selects trip coordinates
                               |
                   Fetch API: POST /predict
                               |
                        Flask backend
                             app.py
                               |
                 Validate and parse request data
                               |
                  Haversine distance calculation
                               |
              Airport proximity and time surcharge rules
                               |
                    Fare estimate and JSON response
                               |
                  Dashboard displays trip results
```

### Request flow

1. The user selects pickup and drop-off locations on the map.
2. The frontend sends the coordinates, passenger count, and pickup datetime to `POST /predict`.
3. Flask validates and parses the submitted values.
4. The backend calculates the geodesic distance and evaluates the configured tariff rules.
5. The API returns a JSON response for the dashboard to display.

## 🧠 Machine Learning Workflow

### Dataset

The project notebook uses the **Kaggle New York City Taxi Fare Prediction** dataset. The documented experiment uses a sample of approximately 100,000 records.

The raw dataset fields include:

| Column | Description |
|---|---|
| `key` | Record identifier |
| `fare_amount` | Recorded taxi fare; prediction target |
| `pickup_datetime` | Pickup date and time |
| `pickup_longitude` | Pickup longitude |
| `pickup_latitude` | Pickup latitude |
| `dropoff_longitude` | Drop-off longitude |
| `dropoff_latitude` | Drop-off latitude |
| `passenger_count` | Number of passengers |

### Data cleaning

The notebook applies domain-based filtering to remove invalid records, including:

- Non-positive fare values.
- Missing or invalid passenger-count values.
- Coordinates outside the defined NYC metropolitan bounding box.

The documented geographic bounds are:

\[
40.50 \leq \text{latitude} \leq 40.90
\]

\[
-74.25 \leq \text{longitude} \leq -73.70
\]

The notebook also examines the target fare distribution and outliers before model training.

### Geospatial feature engineering: Haversine formula

The Haversine formula estimates the great-circle distance between two latitude/longitude coordinates on a spherical Earth.

For pickup \((\phi_1,\lambda_1)\) and drop-off \((\phi_2,\lambda_2)\), with angles expressed in radians:

\[
a = \sin^2\left(\frac{\Delta\phi}{2}\right)
+ \cos(\phi_1)\cos(\phi_2)
\sin^2\left(\frac{\Delta\lambda}{2}\right)
\]

\[
d_{\text{km}} =
2R \cdot \operatorname{atan2}
\left(\sqrt{a},\sqrt{1-a}\right)
\]

Where:

- \(R = 6371.0\) km is the Earth's approximate radius.
- \(\Delta\phi = \phi_2-\phi_1\).
- \(\Delta\lambda = \lambda_2-\lambda_1\).

Convert kilometres to miles using:

\[
d_{\text{miles}} = d_{\text{km}} \times 0.621371
\]

### Deep Neural Network

The notebook builds a regression model using the TensorFlow/Keras Sequential API.

| Stage | Description |
|---|---|
| Input | Normalized coordinates, engineered distance, passenger count, and temporal features |
| Hidden layers | Dense layers with ReLU activation |
| Regularization | Batch Normalization and Dropout |
| Output | One linear neuron for continuous fare prediction |
| Optimizers explored | Adam, RMSprop, and SGD |
| Training callbacks | EarlyStopping and ReduceLROnPlateau |
| Evaluation | MAE, MSE, and R² |

The exact layer sizes and experiment-specific results should be taken from the notebook rather than inferred from this overview.

## 🛠️ Technology Stack

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Notebook experiments and backend |
| Deep learning | TensorFlow 2.20 / Keras | DNN regression workflow |
| Data processing | NumPy, Pandas | Data manipulation and numerical operations |
| ML utilities | Scikit-learn, Joblib | Preprocessing, splitting, scaling, and evaluation |
| Backend | Flask 3.0+ | Web server and REST API |
| Frontend | HTML, JavaScript | Dashboard interface and client-side interactions |
| Map | Leaflet.js 1.9.4 | Interactive map and markers |
| Map tiles | OpenStreetMap | Map display |
| Styling | Tailwind CSS 3.4 | Responsive UI styling |
| Icons | Lucide Icons | Interface icons |

## 🔌 API Documentation

### `POST /predict`

Calculates an estimated taxi fare and returns trip distance and surcharge details.

**Request headers**

```http
Content-Type: application/json
```

**Request body**

```json
{
  "pickup_latitude": 40.7580,
  "pickup_longitude": -73.9855,
  "dropoff_latitude": 40.6413,
  "dropoff_longitude": -73.7781,
  "passenger_count": 2,
  "pickup_datetime": "2026-11-12T17:30"
}
```

### Request parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `pickup_latitude` | Float | Yes | Pickup latitude in decimal degrees |
| `pickup_longitude` | Float | Yes | Pickup longitude in decimal degrees |
| `dropoff_latitude` | Float | Yes | Drop-off latitude in decimal degrees |
| `dropoff_longitude` | Float | Yes | Drop-off longitude in decimal degrees |
| `passenger_count` | Integer | Yes | Number of passengers |
| `pickup_datetime` | String | Yes | Pickup time in `YYYY-MM-DDTHH:MM` format |

### Successful response — `200 OK`

```json
{
  "status": "success",
  "predicted_fare": "$72.50",
  "distance_miles": "13.48 mi",
  "distance_km": "21.69 km",
  "pickup_time": "Thursday, Nov 12 2026 at 05:30 PM",
  "details": {
    "peak_surcharge": "$2.50",
    "night_surcharge": "$0.00",
    "passengers": 2
  }
}
```

*The values above illustrate the documented response structure; they are not a guaranteed result for every run.*

### Error response — `400 Bad Request`

```json
{
  "status": "error",
  "message": "Invalid coordinate or datetime format provided."
}
```

## 📂 Project Structure

```text
nyc-taxi-fare-predictor/
│
├── assets/
│   ├── dashboard-preview.png       # Dashboard screenshot
│   └── route-selection.png         # Route and fare result screenshot
│
├── static/
│   ├── css/
│   │   └── style.css               # Custom stylesheet
│   └── js/
│       └── main.js                 # Frontend interactions
│
├── templates/
│   └── index.html                  # Main dashboard template
│
├── app.py                          # Flask server and /predict API
├── NYC_Taxi_Fare_Prediction.ipynb  # EDA, preprocessing, and DNN workflow
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                        # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12.
- `pip`.
- Git.
- A modern web browser.

### 1. Clone the repository

```bash
git clone https://github.com/Anubhab47677/nyc-taxi-fare-predictor.git
cd nyc-taxi-fare-predictor
```

### 2. Create a virtual environment

**Windows — PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

### 5. Open the dashboard

Visit:

```text
http://127.0.0.1:5000
```

The Flask development server should now serve the interactive taxi fare estimation dashboard.

## 🗺️ Roadmap

- [ ] **Integrate the trained DNN into live inference:** Export the trained Keras model and fitted preprocessing scaler, then connect them to the Flask prediction flow.
- [ ] **Road-network routing:** Integrate OSRM or another routing service to estimate road distance and travel duration instead of relying only on straight-line geodesic distance.
- [ ] **NYC borough overlays:** Add GeoJSON boundaries for Manhattan, Brooklyn, Queens, The Bronx, and Staten Island.
- [ ] **Fare visualizations:** Add historical fare-density and trip-analysis heatmaps.
- [ ] **Production deployment:** Add Docker and a production WSGI server such as Gunicorn, then configure deployment for a hosting platform.

## ⚠️ Notes and Limitations

- Haversine distance is a straight-line surface distance; it is not the actual road distance travelled by a taxi.
- Fare estimates depend on the tariff values and rules configured in the application. They should not be treated as official quotes.
- The notebook's trained DNN is a research/training component until model serialization and integration into the Flask inference path are completed.
- Keep large datasets, virtual environments, generated model files, and other local artifacts out of Git unless they are intentionally part of the repository.

## 📄 License

This project is available under the **MIT License**. See the `LICENSE` file if included in the repository.

---

<div align="center">

### Developed by Anubhab

If you find this project useful, consider giving the repository a ⭐ on GitHub.

[GitHub Repository](https://github.com/Anubhab47677/nyc-taxi-fare-predictor)

</div>
