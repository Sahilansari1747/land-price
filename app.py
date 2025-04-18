from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd  # ✅ ADD THIS

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        area = float(request.form["area"])
        location = request.form["location"]
        land_type = request.form["land_type"]
        road_distance = float(request.form["road_distance"])

        # ✅ Convert input to DataFrame with column names
        input_data = pd.DataFrame([[area, location, land_type, road_distance]],
                                  columns=["area", "location", "land_type", "road_distance"])

        # ✅ Model predict
        prediction = model.predict(input_data)[0]

        return render_template("result.html", price=round(prediction, 2))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)