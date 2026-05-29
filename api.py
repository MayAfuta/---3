import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

from assets_data_prep import data_prepare

app = Flask(__name__)

# טעינת המודל המאומן מחלק 2
model = joblib.load("trained_model.pkl")


@app.route("/", methods=["GET"])
def home():
    """מחזיר את דף הטופס הראשי"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """מקבל נתוני סרט, מפעיל data_prepare ומחזיר תחזית דירוג"""
    try:
        # 1. קריאת הנתונים מהטופס (JSON)
        data = request.get_json()

        # בדיקת שדות חסרים
        required_fields = ['runtime', 'year', 'genre', 'language', 'country', 'has_budget']
        for field in required_fields:
            if field not in data or data[field] in (None, ''):
                return jsonify({"error": f"חסר שדה: {field}"}), 400

        # בדיקת ערכים מספריים תקינים
        try:
            runtime = float(data['runtime'])
            year = int(data['year'])
        except (ValueError, TypeError):
            return jsonify({"error": "שנה ואורך חייבים להיות מספרים תקינים"}), 400

        # המרת has_budget: אם המשתמש בחר "Yes" נשים ערך תקציב כלשהו, אחרת Unknown
        budget_value = "1000000" if data['has_budget'] == "Yes" else "Unknown"

        # 2. בניית DataFrame עם שורה אחת
        new_movie = pd.DataFrame([{
            "runtimeMinutes": runtime,
            "startYear": year,
            "genres": data['genre'],
            "lead_actors_ids": "Unknown",
            "Language": data['language'],
            "Country": data['country'],
            "budget": budget_value
        }])

        # 3. הפעלת data_prepare
        X_new = data_prepare(new_movie)
        X_new = X_new.select_dtypes(include=["number"])

        # 4. הפעלת המודל
        predicted_rating = model.predict(X_new)[0]
        predicted_rating = round(float(predicted_rating), 2)

        # הגבלת הטווח ל-1-10
        predicted_rating = max(1.0, min(10.0, predicted_rating))

        # 5. החזרת התוצאה כ-JSON
        return jsonify({"predicted_rating": predicted_rating})

    except Exception as e:
        # שגיאה פנימית לא צפויה
        return jsonify({"error": f"שגיאה פנימית: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)