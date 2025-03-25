from flask import Flask, render_template, request, jsonify
import pandas as pd
from flask import jsonify
from src.data_loader import load_dataset, preprocess_data
from src.stress_detection import train_stress_detector, detect_stress
from src.visualization import generate_plot
import joblib
app = Flask(__name__)

# Load and preprocess the dataset
DATASET_PATH = "data/healthcare_iot_target_dataset.csv"
df = load_dataset(DATASET_PATH)
df = preprocess_data(df)

# Train the stress detection model (or load a pre-trained model)
try:
    clf = joblib.load("stress_detection_model.pkl")
    print("Loaded pre-trained model.")
except FileNotFoundError:
    print("Training new model...")
    clf = train_stress_detector(df)

@app.route("/")
def home():
    """Render the home page."""
    return render_template("index.html")

@app.route("/detect_stress", methods=["POST"])
def detect_stress_route():
    """Detect stress based on input data."""
    try:
        # Get input data from the request
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate input data
        required_fields = ["temperature", "systolic_bp", "diastolic_bp", "heart_rate"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        temperature = float(data["temperature"])
        systolic_bp = float(data["systolic_bp"])
        diastolic_bp = float(data["diastolic_bp"])
        heart_rate = float(data["heart_rate"])

        # Detect stress
        stress_level = detect_stress(clf, [temperature, systolic_bp, diastolic_bp, heart_rate])

        # Return the result as JSON
        return jsonify({"stress_level": int(stress_level)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/plot")
def plot():
    """Generate and return a plot of the dataset."""
    try:
        plot_url = generate_plot(df)
        return jsonify({"plot_url": plot_url})  # Return as JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)