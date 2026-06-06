# 🎬 Movie Rating Predictor

## Overview

This project presents a machine learning web application that predicts IMDb movie ratings based on movie characteristics available before release.

The system was developed as part of a Machine Learning course project and includes:

- A trained Elastic Net prediction model
- Data preprocessing pipeline
- Flask backend API
- Interactive web interface
- Actor selection with automatic ID conversion
- Real-time rating prediction

The model predicts ratings on a scale of **1–10** using only information available before a movie is released, while carefully preventing **Data Leakage**.

---

## Project Structure

```text
movie_rating_app/
│
├── api.py
├── assets_data_prep.py
├── trained_model.pkl
├── top_100_actor_mapping.xlsx
│
├── templates/
│   └── index.html
│
├── static/
│   └── movie_bg.jpg
│
├── requirements.txt
└── README.md
```

---

## Features Used by the Model

The prediction model uses the following information:

### Movie Information

- Release Year
- Runtime (Minutes)
- Known Budget (Yes / No)

### Genres

- Drama
- Comedy
- Romance
- Action
- Documentary
- Crime
- Thriller
- Horror
- Adventure
- Mystery
- Family
- Fantasy
- Biography
- History
- Sci-Fi
- Music
- Unknown
- Other

### Languages

- English
- French
- Hindi
- Spanish
- Unknown
- Other

### Countries

- United States
- India
- United Kingdom
- France
- Italy
- Japan
- Canada
- Unknown
- Other

### Actors

The application supports selecting up to 5 actors from the Top-100 actors used during training.

Actor names are automatically converted to IMDb actor IDs behind the scenes before prediction.

---

## Data Preparation

The preprocessing pipeline performs:

### Data Leakage Prevention

The following features were removed because they are unavailable before a movie is released:

- plot
- numVotes
- BoxOffice

### Feature Engineering

- Creation of decade feature from release year
- Genre encoding
- Language encoding
- Country encoding
- Actor encoding using Top-100 actor indicators
- Runtime missing-value indicator
- Runtime squared feature
- Budget availability indicator

All preprocessing is automatically handled by:

```python
data_prepare()
```

---

## Machine Learning Model

### Model Type

Elastic Net Regression

### Prediction Target

```text
averageRating
```

### Rating Scale

```text
1.0 – 10.0
```

Predictions are automatically clipped to remain within the valid IMDb rating range.

---

## Running the Application

### Step 1 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 – Start the Flask Server

```bash
python api.py
```

### Step 3 – Open the Application

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## Example Workflow

1. Select movie release year.
2. Select runtime.
3. Select one or more genres.
4. Select one or more languages.
5. Select one or more countries.
6. Select up to 5 actors.
7. Indicate whether budget information is known.
8. Click **Predict Rating**.
9. Receive a predicted IMDb rating.

---

## Data Leakage Prevention

This project was specifically designed to prevent Data Leakage.

The following variables were removed during preprocessing:

- numVotes
- BoxOffice
- plot

These variables are not available before a movie is released and therefore cannot be used for prediction.

All preprocessing steps are applied consistently through the preprocessing pipeline before generating predictions.

---

## Authors

**יהודית אלמיהו**  
ID: 206131468

**מאי אפוטה**  
ID: 315098707

---

## Course

Machine Learning – Part 3

Movie Rating Prediction System

Ariel University