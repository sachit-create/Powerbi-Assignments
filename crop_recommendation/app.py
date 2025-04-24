from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and label encoder
model = joblib.load("crop_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", error_message="Something went wrong, please try again later.")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/send_contact", methods=["POST"])
def send_contact():
    # Handle contact form submission here, save to a file or send an email.
    return "Thank you for contacting us!" 


@app.route("/predict", methods=["POST"])
def predict():
    # Extract input values from form
    try:
        features = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"]),
        ]
        features_np = np.array([features])
        prediction = model.predict(features_np)
        crop = label_encoder.inverse_transform(prediction)[0]
    except Exception as e:
        crop = f"Error: {str(e)}"

    return render_template("index.html", crop=crop)

if __name__ == "__main__":
    app.run(debug=True)
