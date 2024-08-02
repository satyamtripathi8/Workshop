from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]

        # Convert input features to DataFrame and scale
        input_df = pd.DataFrame([input_features], columns=[
            'sepal_length', 'sepal_width', 'petal_length', 'petal_width'
        ])
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        classes = ['Setosa', 'Versicolour', 'Virginica']
        predicted_class = classes[prediction[0]]
        return render_template('index.html', prediction_text=f'Predicted Class: {predicted_class}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
