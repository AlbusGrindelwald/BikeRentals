from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np
import pandas as pd

# Load trained model and feature list
model = joblib.load('bike_rental_regressor.pkl')
training_features = pickle.load(open("training_features.pkl", "rb"))

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        # Collect user inputs
        hour = int(request.form['hour'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        wind_speed = float(request.form['wind_speed'])
        visibility = float(request.form['visibility'])
        dew_point = float(request.form['dew_point'])
        solar_radiation = float(request.form['solar_radiation'])
        rainfall = float(request.form['rainfall'])
        snowfall = float(request.form['snowfall'])

        season = request.form['season']
        holiday = request.form['holiday']
        functioning_day = request.form['functioning_day']

        # One-hot encoding
        feature_dict = {
            'Hour': hour,
            'Temperature(°C)': temperature,
            'Humidity(%)': humidity,
            'Wind speed (m/s)': wind_speed,
            'Visibility (10m)': visibility,
            'Dew point temperature(°C)': dew_point,
            'Solar Radiation (MJ/m2)': solar_radiation,
            'Rainfall(mm)': rainfall,
            'Snowfall (cm)': snowfall,
            'Year': 2018,
            'Month': 6,
            'Day': 15,
            'Seasons_Spring': 0,
            'Seasons_Summer': 0,
            'Seasons_Winter': 0,
            'Holiday_No Holiday': 0,
            'Functioning Day_Yes': 0
        }

        # Update categorical values
        feature_dict[f"Seasons_{season.capitalize()}"] = 1
        feature_dict["Holiday_No Holiday"] = 1 if holiday == "no" else 0
        feature_dict["Functioning Day_Yes"] = 1 if functioning_day == "yes" else 0

        # Create DataFrame
        input_df = pd.DataFrame([feature_dict])

        # Ensure all features match training
        for col in training_features:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[training_features]
        print(feature_dict)

        # Predict
        prediction = model.predict(input_df)[0]
        print(prediction)
        result = "High Rentals" if prediction >0 else "Low Rentals"

        return render_template('result.html', prediction=result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
