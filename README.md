# 🏙️ NYC Airbnb Room Type Predictor

An end-to-end machine learning project that predicts whether an Airbnb listing in New York City is an **Entire home/apt**, a **Private room**, or a **Shared room**, based on its price, location, availability and review activity.

The project covers the full lifecycle: data exploration, preprocessing, model comparison, hyperparameter tuning, model serialization, a **FastAPI** REST backend, and a simple **HTML/CSS/JavaScript** frontend where anyone can try the model.

<!-- Add a screenshot or GIF of your web app here. This is the single most useful thing for recruiters. -->
<!-- ![App Demo](screenshots/demo.png) -->

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Workflow](#-project-workflow)
- [Model Results](#-model-results)
- [Project Structure](#-project-structure)
- [How to Run Locally](#-how-to-run-locally)
- [API Documentation](#-api-documentation)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🎯 Problem Statement

Given the attributes of an Airbnb listing (location, price, minimum nights, reviews, availability, host activity), can we predict its **room type**?

This is a **multi-class classification** problem with **imbalanced classes**: Shared rooms make up only a small fraction of listings. Because of that, I used **macro F1-score** (not just accuracy) to evaluate models fairly across all three classes.

**Possible real-world uses:** detecting mislabeled listings, validating listing data quality, and supporting pricing or recommendation systems.

---

## 📊 Dataset

- **Source:** [New York City Airbnb Open Data (Kaggle)](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
- **File used:** `AB_NYC_2019.csv`
- **Target variable:** `room_type` (`Entire home/apt`, `Private room`, `Shared room`)

**Features used for prediction:**

| Feature | Type | Description |
|---|---|---|
| `latitude`, `longitude` | Numeric | Location of the listing |
| `price` | Numeric | Price per night (USD) |
| `minimum_nights` | Numeric | Minimum nights required to book |
| `number_of_reviews` | Numeric | Total reviews |
| `reviews_per_month` | Numeric | Average reviews per month |
| `calculated_host_listings_count` | Numeric | Number of listings owned by the host |
| `availability_365` | Numeric | Days available in a year |
| `neighbourhood_group` | Categorical | Borough (Manhattan, Brooklyn, etc.) |
| `neighbourhood` | Categorical | Specific neighbourhood |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Data analysis & ML** | Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn |
| **Model persistence** | Joblib |
| **Backend** | FastAPI, Pydantic, Uvicorn |
| **Frontend** | HTML, CSS, JavaScript |

---

## 🔄 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked missing values, class distribution and feature distributions (skewness)
- Bivariate analysis (e.g., price vs. room type using boxplots)
- Correlation heatmap and a geographic scatter plot of listings colored by room type

### 2. Data Cleaning & Feature Engineering
- Dropped identifier / free-text columns with no generalizable signal: `id`, `name`, `host_id`, `host_name`, `last_review`
- Filled missing `reviews_per_month` with `0` (a listing with no reviews has no monthly review rate)
- **Capped outliers** at the 99th percentile for `price` and `minimum_nights` instead of deleting rows, so data-entry errors do not distort the model

### 3. Preprocessing with `Pipeline` + `ColumnTransformer`
- **Numeric features:** median imputation → standard scaling
- **Categorical features:** most-frequent imputation → one-hot encoding (`handle_unknown='ignore'`)
- Preprocessing lives *inside* the pipeline, so it is fit only on training data (**no data leakage**) and is applied identically at inference time

### 4. Handling Class Imbalance
- Used a **stratified** train/test split to preserve class proportions
- Used `class_weight="balanced"` for Logistic Regression, Decision Tree and Random Forest
- Chose class weighting over SMOTE to avoid the overfitting and leakage risks of synthetic oversampling

### 5. Model Comparison (3-fold Cross-Validation)
Compared four algorithms on both **accuracy** and **macro F1**:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

### 6. Hyperparameter Tuning
- Tuned the **Random Forest** with `RandomizedSearchCV` (10 iterations, 3-fold CV), optimizing **macro F1**
- Parameters searched: `n_estimators`, `max_depth`, `min_samples_split`

### 7. Final Evaluation & Deployment
- Evaluated once on the held-out test set (accuracy, macro F1, confusion matrix)
- Saved the full pipeline (preprocessing + model) as `Model_Pipeline.pkl`
- Served it through a FastAPI endpoint and connected a web frontend to it

---

## 📈 Model Results

> ⚠️ Fill in these numbers from your notebook output before publishing.

| Model | CV Accuracy | CV Macro F1 |
|---|---|---|
| Logistic Regression | _xx.x%_ | _0.xxx_ |
| Decision Tree | _xx.x%_ | _0.xxx_ |
| Random Forest | _xx.x%_ | _0.xxx_ |
| Gradient Boosting | _xx.x%_ | _0.xxx_ |

**Final tuned Random Forest on the test set:**

- Accuracy: **_xx.x%_**
- Macro F1: **_0.xxx_**
- Best hyperparameters: `{ ... }`

<!-- Optionally add your confusion matrix image here -->
<!-- ![Confusion Matrix](screenshots/confusion_matrix.png) -->

---

## 📁 Project Structure

```
NYC-Airbnb-Room-Type-Predicton/
│
├── main.py                                     # FastAPI backend (prediction API)
├── Model_Pipeline.pkl                          # Trained pipeline (preprocessing + model)
├── nyc_airbnb_room_type_classification.ipynb   # EDA, training and evaluation notebook
├── index.html                                  # Frontend UI
├── style.css                                   # Frontend styling
├── script.js                                   # Frontend logic (calls the API)
├── requirements.txt                            # Python dependencies
└── README.md
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/dheeraj1306/NYC-Airbnb-Room-Type-Predicton.git
cd NYC-Airbnb-Room-Type-Predicton
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI backend
```bash
uvicorn main:app --reload
```
The API runs at `http://127.0.0.1:8000`, and interactive docs are available at `http://127.0.0.1:8000/docs`.

### 5. Open the frontend
Open `index.html` in your browser (or use the VS Code *Live Server* extension), fill in the listing details, and click **Predict**.

> Note: `script.js` must point to your backend URL (e.g. `http://127.0.0.1:8000/predict`).

---

## 🔌 API Documentation

### `GET /`
Health check. Returns a greeting message.

### `POST /predict`
Predicts the room type for a listing.

**Request body:**
```json
{
  "latitude": 40.7128,
  "longitude": -74.0060,
  "price": 150,
  "minimum_nights": 2,
  "number_of_reviews": 25,
  "reviews_per_month": 1.2,
  "calculated_host_listings_count": 1,
  "availability_365": 180,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Harlem"
}
```

**Response:**
```json
{
  "Predicted_room_type": "Entire home/apt",
  "Probability": [0.82, 0.16, 0.02]
}
```

> The probabilities follow the model's class order: `Entire home/apt`, `Private room`, `Shared room`.

**Input validation (Pydantic):** latitude must be within -90 to 90, longitude within -180 to 180, `price` must be positive, `minimum_nights` between 1 and 365, and `availability_365` between 0 and 365. Invalid input returns a `422` error with a clear message.

---

## 💡 Key Learnings

- Building leakage-free ML workflows with `Pipeline` and `ColumnTransformer`
- Why **macro F1** is a better metric than accuracy on imbalanced data
- Capping outliers instead of deleting data
- Serializing a full pipeline so inference needs no manual preprocessing
- Exposing an ML model as a validated REST API and connecting it to a frontend

---

## 🔮 Future Improvements

- Deploy the backend (Render / Railway / AWS) and the frontend (GitHub Pages / Netlify) with a live demo link
- Try XGBoost / LightGBM and compare against Random Forest
- Add feature importance and SHAP explanations to the UI
- Dockerize the application
- Add unit tests for the API endpoints
- Replace manual coordinate entry with a map picker in the frontend

---

## 👤 Author

**Dheeraj**

- GitHub: [@dheeraj1306](https://github.com/dheeraj1306)
- LinkedIn: _add your LinkedIn URL_
- Email: _add your email_

⭐ If you found this project useful, consider giving it a star!
