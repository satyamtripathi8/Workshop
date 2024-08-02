from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model_filename = 'multi_linear_regression_model.joblib'
model = joblib.load(model_filename)

# Load training data to get column names
df_train = pd.read_csv('advertising.csv')
feature_names = ['TV', 'Radio', 'Newspaper']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input
    input_data = {
        'TV': [float(request.form['TV'])],
        'Radio': [float(request.form['Radio'])],
        'Newspaper': [float(request.form['Newspaper'])]
    }

    # Create DataFrame from input data
    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Make prediction
    predicted_sales = model.predict(input_df)
    prediction = round(predicted_sales[0], 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
