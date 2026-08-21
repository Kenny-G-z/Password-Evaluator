import os
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

file = "passwords.pkl"

if os.path.exists(file):
    model = joblib.load(file)
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        if classifier is not None:
            classifier.set_params(device="cpu")
else:
    model = None


def estimate_crack_time(password):
    if not password:
        return "Instant"

    charset_size = 0

    if any(c.islower() for c in password):
        charset_size += 26
    if any(c.isupper() for c in password):
        charset_size += 26
    if any(c.isdigit() for c in password):
        charset_size += 10
    if any(not c.isalnum() for c in password):
        charset_size += 32
    if charset_size == 0:
        return "Instant"

    combinations = charset_size ** len(password)
    guesses_per_second = 5_000_000_000_000
    seconds = combinations / guesses_per_second

    if seconds < 1:
        return "Instant"
    if seconds < 60:
        return f"{int(seconds)} seconds"
    if seconds < 3600:
        return f"{int(seconds / 60)} minutes"
    if seconds < 86400:
        return f"{int(seconds / 3600)} hours"
    if seconds < 31536000:
        return f"{int(seconds / 86400)} days"
    years = int(seconds / 31536000)

    return f"{years:,} years"


def get_strength_label(tier):
    if tier == 0:
        return "Weak"
    if tier == 1:
        return "Average"
    return "Strong"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"Error": "Model not found."}), 500

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"Error": "No JSON data received."}), 400

    password = data.get("password", "")

    if not isinstance(password, str):
        return jsonify({"Error": "Password must be a string."}), 400
    if not password:
        return jsonify({"label": "Empty", "tier": -1, "crack_time": ""})

    try:
        pred_tier = int(model.predict([password])[0])
        label = get_strength_label(pred_tier)
        crack_time = estimate_crack_time(password)

        return jsonify({"label": label, "tier": pred_tier, "crack_time": crack_time})
    except Exception as Error:
        return jsonify({ "Error": "Unable to analyze password."}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)