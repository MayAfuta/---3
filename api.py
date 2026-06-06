import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

from assets_data_prep import data_prepare

app = Flask(__name__)

# טעינת המודל עם error handling
try:
    model = joblib.load("trained_model.pkl")
except FileNotFoundError:
    print("ERROR: trained_model.pkl not found!")
    print("Please ensure trained_model.pkl is in the same directory as api.py")
    exit(1)
except Exception as e:
    print(f"ERROR loading model: {str(e)}")
    exit(1)

# טעינת קובץ המרה: שם שחקן -> מזהה שחקן
try:
    actor_mapping = pd.read_excel("top_100_actor_mapping.xlsx")
    actor_mapping = actor_mapping.dropna(subset=["actor_id", "actor_name"])
    name_to_id = dict(zip(actor_mapping["actor_name"], actor_mapping["actor_id"]))
except FileNotFoundError:
    print("ERROR: top_100_actor_mapping.xlsx not found!")
    print("Please ensure top_100_actor_mapping.xlsx is in the same directory as api.py")
    exit(1)
except Exception as e:
    print(f"ERROR loading actor mapping: {str(e)}")
    exit(1)


@app.route("/", methods=["GET"])
def home():
    """
    מחזיר את דף index.html
    """
    return render_template("index.html")


@app.route("/actors", methods=["GET"])
def get_actors():
    """
    מחזיר רשימה של שחקנים מהדאטאסט
    """
    try:
        actors = actor_mapping["actor_name"].dropna().sort_values().tolist()
        return jsonify({"actors": actors})
    except Exception as e:
        return jsonify({"error": f"Error loading actors: {str(e)}"}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    מקבל JSON עם נתוני סרט ומחזיר חיזוי דירוג
    
    Input JSON:
    {
        "year": int (1904-2024),
        "runtime": int or "unknown",
        "genres": list[str],
        "languages": list[str],
        "countries": list[str],
        "actors": list[str],
        "has_budget": bool
    }
    
    Output JSON:
    {
        "predicted_rating": float (1.0-10.0)
    }
    """
    try:
        # קריאת הנתונים מהטופס
        data = request.get_json()

        # בדיקת שדות חסרים
        required_fields = [
            "runtime",
            "year",
            "genres",
            "languages",
            "countries",
            "has_budget",
            "actors"
        ]

        for field in required_fields:
            if field not in data or data[field] in (None, ""):
                return jsonify({"error": f"Missing field: {field}"}), 400

        # בדיקת arrays (יש להיות list עם לפחות element אחד)
        if not isinstance(data["genres"], list) or len(data["genres"]) == 0:
            return jsonify({"error": "Please select at least one genre"}), 400

        if not isinstance(data["languages"], list) or len(data["languages"]) == 0:
            return jsonify({"error": "Please select at least one language"}), 400

        if not isinstance(data["countries"], list) or len(data["countries"]) == 0:
            return jsonify({"error": "Please select at least one country"}), 400

        if not isinstance(data["actors"], list) or len(data["actors"]) == 0:
            return jsonify({"error": "Please select at least one actor"}), 400

        # בדיקת מגבלה: עד 5 שחקנים
        if len(data["actors"]) > 5:
            return jsonify({"error": "You can select up to 5 actors only"}), 400

        # בדיקת ערכים מספריים - טיפול ב-'unknown' runtime
        if data["runtime"] == 'unknown':
            runtime = None
        else:
            try:
                runtime = float(data["runtime"])
            except (ValueError, TypeError):
                return jsonify({"error": "Runtime must be a valid number"}), 400

        # בדיקת שנה
        try:
            year = int(data["year"])
        except (ValueError, TypeError):
            return jsonify({"error": "Year must be a valid number"}), 400

        # המרת arrays לסטרינגים (קומה מופרדים)
        genres_str = ",".join(data["genres"])
        languages_str = ",".join(data["languages"])
        countries_str = ",".join(data["countries"])

        # המרת שמות שחקנים ל-IDs
        selected_actor_names = data.get("actors", [])

        actor_ids = []
        for actor_name in selected_actor_names:
            if actor_name == "Other":
                # אם בחרו "Other", שמור את המילה כמו שהיא
                actor_ids.append("Other")
            else:
                # חפש את ה-ID בדיקשונרי
                actor_id = name_to_id.get(actor_name)
                if actor_id:
                    actor_ids.append(str(actor_id))

        lead_actors_value = ",".join(actor_ids) if actor_ids else "Unknown"

        # המרת has_budget: True/False -> ערך תקציב
        budget_value = "1000000" if data["has_budget"] is True else "Unknown"

        # בניית DataFrame עם שורה אחת
        # חשוב: כל העמודות הנדרשות ל-data_prepare()
        new_movie = pd.DataFrame([{
            "tconst": "tt_prediction",
            "primaryTitle": "Movie to Predict",
            "startYear": year,
            "runtimeMinutes": runtime,
            "genres": genres_str,
            "lead_actors_ids": lead_actors_value,
            "Language": languages_str,
            "Country": countries_str,
            "budget": budget_value,
            "numVotes": 0,
            "BoxOffice": 0,
            "plot": ""
        }])

        # הפעלת data_prepare() - עיבוד הנתונים
        X_new = data_prepare(new_movie)
        
        # בחר רק עמודות מספריות
        X_new = X_new.select_dtypes(include=["number"])

        # הפעלת המודל לחיזוי
        predicted_rating = model.predict(X_new)[0]
        
        # עיגול לשתי ספרות אחרי הנקודה
        predicted_rating = round(float(predicted_rating), 2)
        
        # הגבלה לטווח תקין 1-10
        predicted_rating = max(1.0, min(10.0, predicted_rating))

        # החזרת התוצאה כ-JSON
        return jsonify({"predicted_rating": predicted_rating})

    except Exception as e:
        # שגיאה פנימית לא צפויה
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == "__main__":
    # הרצת Flask development server
    # debug=True מאפשר reload אוטומטי בזמן פיתוח
    app.run(debug=True, host="127.0.0.1", port=5000)