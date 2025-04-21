from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('bike_price_predictor.pkl')


# Route for the Home Page (index.html)
@app.route('/')
def index():
    return render_template('index.html')

# Route for the History Page (history.html)
@app.route('/history')
def history():
    return render_template('history.html')

# Route for the History 1 Page (history1.html)
@app.route('/history1')
def history1():
    return render_template('history1.html')

# Route for the Contact Page (contact.html)
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Here you can process the form submission
        # For now, let's assume we just redirect to a thank you page
        return render_template('thankyou.html')
    return render_template('contact.html')

# Route for the Thank You Page (thankyou.html)
@app.route('/about')
def thankyou():
    return render_template('about.html')

@app.route('/project')
def home():
    return render_template('project.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    bike_name = request.form['bike_name']
    city = request.form['city']
    owner = request.form['owner']
    brand = request.form['brand']
    kms_driven = int(request.form['kms_driven'])
    age = int(request.form['age'])
    power = float(request.form['power'])

    # Create input dataframe
    input_df = pd.DataFrame([[bike_name, city, owner, brand, kms_driven, age, power]],
                            columns=['bike_name', 'city', 'owner', 'brand', 'kms_driven', 'age', 'power'])

    # Predict
    prediction = model.predict(input_df)[0]
    prediction = round(prediction, 2)

    return render_template('project.html', prediction=f"₹ {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
