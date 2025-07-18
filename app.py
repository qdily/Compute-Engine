from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model/credit_risk_model.pkl')  # adjust path if needed

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    try:
        person_income = float(data['person_income'])
        loan_amnt = float(data['loan_amnt'])
        loan_percent_income = loan_amnt / person_income if person_income != 0 else 0

        input_data = [[
            int(data['person_age']),
            person_income,
            float(data['person_emp_length']),
            data['loan_grade'],
            data['person_home_ownership'],
            data['loan_intent'],
            loan_amnt,
            float(data['loan_int_rate']),
            loan_percent_income,
            int(data['cb_person_default_on_file']),
            int(data['cb_person_cred_hist_length'])
        ]]

        cols = ['person_age', 'person_income', 'person_emp_length', 'loan_grade',
                'person_home_ownership', 'loan_intent', 'loan_amnt', 'loan_int_rate',
                'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']

        input_df = pd.DataFrame(input_data, columns=cols)

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        return render_template('index.html', prediction=pred, proba=round(proba, 3))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)