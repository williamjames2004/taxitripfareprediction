from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import math

app = Flask(__name__)

MODEL_PATH = os.path.join(app.root_path, "trip_price_model.pkl")
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Trip_Distance_km",
    "Time_of_Day",
    "Day_of_Week",
    "Passenger_Count",
    "Traffic_Conditions",
    "Weather",
    "Base_Fare",
    "Per_Km_Rate",
    "Per_Minute_Rate",
    "Trip_Duration_Minutes"
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()

        try:
            distance = float(request.form["distance"])
            time_of_day = request.form["time_of_day"].strip()
            day_of_week = request.form["day_of_week"].strip()
            passengers = int(request.form["passengers"])
            traffic = request.form["traffic"].strip()
            weather = request.form["weather"].strip()
            base_fare = float(request.form["base_fare"])
            per_km_rate = float(request.form["per_km_rate"])
            per_minute_rate = float(request.form["per_minute_rate"])
            duration = float(request.form["duration"])

            numeric_values = [
                distance, passengers, base_fare,
                per_km_rate, per_minute_rate, duration
            ]

            if not all(math.isfinite(value) for value in numeric_values):
                raise ValueError("Please enter valid finite numeric values.")

            if distance <= 0:
                raise ValueError("Trip distance must be greater than zero.")

            if passengers < 1:
                raise ValueError("Passenger count must be at least one.")

            if duration <= 0:
                raise ValueError("Trip duration must be greater than zero.")

            if min(base_fare, per_km_rate, per_minute_rate) < 0:
                raise ValueError("Fare and rate values cannot be negative.")

            if time_of_day not in [
                "Morning", "Afternoon", "Evening", "Night"
            ]:
                raise ValueError("Please select a valid time of day.")

            if day_of_week not in ["Weekday", "Weekend"]:
                raise ValueError("Please select a valid day type.")

            if traffic not in ["Low", "Medium", "High"]:
                raise ValueError("Please select valid traffic conditions.")

            if weather not in ["Clear", "Rainy", "Cloudy"]:
                raise ValueError("Please select valid weather conditions.")

            trip = pd.DataFrame([{
                "Trip_Distance_km": distance,
                "Time_of_Day": time_of_day,
                "Day_of_Week": day_of_week,
                "Passenger_Count": passengers,
                "Traffic_Conditions": traffic,
                "Weather": weather,
                "Base_Fare": base_fare,
                "Per_Km_Rate": per_km_rate,
                "Per_Minute_Rate": per_minute_rate,
                "Trip_Duration_Minutes": duration
            }], columns=FEATURES)

            result = float(model.predict(trip)[0])

            if not math.isfinite(result):
                raise ValueError("The model returned an invalid prediction.")

            prediction = round(max(0, result), 2)

        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc) or "Please enter valid values in all fields."

        except Exception:
            app.logger.exception("Trip price prediction failed")
            error = (
                "Prediction failed. Please check that the trained model "
                "matches the current dataset and feature names."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data
    )


if __name__ == "__main__":
    app.run(debug=True)