from flask import Flask, render_template,request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('polynomial_regression_model.joblib')

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    reviews = float(request.form['reviews'])
    size = float(request.form['size'])
    installs = float(request.form['installs'])
    price = float(request.form['price'])

    # Create a dictionary to store the input values
    input_values = {'Reviews': reviews, 'Size': size, 'Installs': installs, 'Price': price}

    # Use the model to make a prediction
    prediction = model.predict([input_values])

    # Return the prediction as a string
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)