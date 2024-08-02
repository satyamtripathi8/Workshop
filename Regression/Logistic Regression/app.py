from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('Logistic_regression.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = int(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])
    embarked = int(request.form['embarked'])

    user_input = [[pclass, sex, age, sibsp, parch, fare, embarked]]
    prediction = model.predict_proba(user_input)
    survival_probability = prediction[0][1]

    return render_template('result.html', survival_probability=survival_probability)

if __name__ == '__main__':
    app.run(debug=True)