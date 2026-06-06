import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify

from assets_data_prep import data_prepare

app = Flask(__name__)

model = joblib.load("trained_model.pkl")

# טעינת קובץ המרה: שם שחקן -> מזהה שחקן
actor_mapping = pd.read_excel("top_100_actor_mapping.xlsx")
actor_mapping = actor_mapping.dropna(subset=["actor_id", "actor_name"])

name_to_id = dict(zip(actor_mapping["actor_name"], actor_mapping["actor_id"]))


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/actors", methods=["GET"])
def get_actors():
    actors = actor_mapping["actor_name"].dropna().sort_values().tolist()
    return jsonify({"actors": actors})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

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

        if not isinstance(data["genres"], list) or len(data["genres"]) == 0:
            return jsonify({"error": "Please select at least one genre"}), 400

        if not isinstance(data["languages"], list) or len(data["languages"]) == 0:
            return jsonify({"error": "Please select at least one language"}), 400

        if not isinstance(data["countries"], list) or len(data["countries"]) == 0:
            return jsonify({"error": "Please select at least one country"}), 400

        if not isinstance(data["actors"], list) or len(data["actors"]) == 0:
            return jsonify({"error": "Please select at least one actor"}), 400

        if len(data["actors"]) > 5:
            return jsonify({"error": "You can select up to 5 actors only"}), 400

        # בדיקת ערכים מספריים - טיפול ב-'unknown'
        if data["runtime"] == 'unknown':
            runtime = None
        else:
            try:
                runtime = float(data["runtime"])
            except (ValueError, TypeError):
                return jsonify({"error": "Runtime must be a valid number"}), 400

        try:
            year = int(data["year"])
        except (ValueError, TypeError):
            return jsonify({"error": "Year must be a valid number"}), 400

        genres_str = ",".join(data["genres"])
        languages_str = ",".join(data["languages"])
        countries_str = ",".join(data["countries"])

        selected_actor_names = data.get("actors", [])

        actor_ids = []
        for actor_name in selected_actor_names:
            if actor_name == "Other":
                actor_ids.append("Other")
            else:
                actor_id = name_to_id.get(actor_name)
                if actor_id:
                    actor_ids.append(str(actor_id))

        lead_actors_value = ",".join(actor_ids) if actor_ids else "Unknown"

        budget_value = "1000000" if data["has_budget"] is True else "Unknown"

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

        X_new = data_prepare(new_movie)
        X_new = X_new.select_dtypes(include=["number"])

        predicted_rating = model.predict(X_new)[0]
        predicted_rating = round(float(predicted_rating), 2)
        predicted_rating = max(1.0, min(10.0, predicted_rating))

        return jsonify({"predicted_rating": predicted_rating})

    except Exception as e:
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)