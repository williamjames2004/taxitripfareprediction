import pandas as pd
import joblib

model = joblib.load("trip_price_model.pkl")

trip = pd.DataFrame([{
    "Trip_Distance_km": float(input("Trip Distance (km): ")),
    "Time_of_Day": input("Time of Day (Morning/Afternoon/Evening/Night): "),
    "Day_of_Week": input("Day of Week (Weekday/Weekend): "),
    "Passenger_Count": int(input("Passenger Count: ")),
    "Traffic_Conditions": input("Traffic (Low/Medium/High): "),
    "Weather": input("Weather (Clear/Rainy/Cloudy): "),
    "Base_Fare": float(input("Base Fare: ")),
    "Per_Km_Rate": float(input("Per Km Rate: ")),
    "Per_Minute_Rate": float(input("Per Minute Rate: ")),
    "Trip_Duration_Minutes": float(input("Trip Duration (minutes): "))
}])

prediction = model.predict(trip)[0]

print("\nPredicted Trip Price:", round(prediction, 2))