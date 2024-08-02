from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model_filename = 'C:/Users/HP/Desktop/Workshop/Regression/Linear Regression/linear_regression_model.joblib'
model = joblib.load(model_filename)

# Load training data to get column names
df_train = pd.read_csv('housing.csv')
df_train = pd.get_dummies(df_train, drop_first=True)
feature_names = [col for col in df_train.columns if col != 'price']

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Collect user input
    input_data = {
        'area': [float(request.form['area'])],
        'bedrooms': [int(request.form['bedrooms'])],
        'bathrooms': [int(request.form['bathrooms'])],
        'stories': [int(request.form['stories'])],
        'parking': [int(request.form['parking'])],
        'mainroad_yes': [int(request.form['mainroad'])],
        'guestroom_yes': [int(request.form['guestroom'])],
        'basement_yes': [int(request.form['basement'])],
        'hotwaterheating_yes': [int(request.form['hotwaterheating'])],
        'airconditioning_yes': [int(request.form['airconditioning'])],
        'prefarea_yes': [int(request.form['prefarea'])],
        'furnishingstatus_semi-furnished': [1 if request.form['furnishingstatus'] == 'semi-furnished' else 0],
        'furnishingstatus_unfurnished': [1 if request.form['furnishingstatus'] == 'unfurnished' else 0]
    }

    # Create DataFrame from input data
    input_df = pd.DataFrame(input_data)

    # Ensure that the input data has the same columns as the model
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Make prediction
    predicted_price = model.predict(input_df)
    prediction = round(predicted_price[0], 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
