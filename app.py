from flask import Flask, render_template_string, request
import numpy as np
from catboost import CatBoostClassifier, Pool
import pandas as pd

app = Flask(__name__)
model = CatBoostClassifier()
model.load_model('catboost_last_canvas_2.cbm')

categorical_columns_inference = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None
    form_data = {}

    if request.method == 'POST':
        form_data = {
            'age': request.form.get('age'),
            'sex': request.form.get('sex'),
            'cp': request.form.get('cp'),
            'trestbps': request.form.get('trestbps'),
            'chol': request.form.get('chol'),
            'fbs': request.form.get('fbs'),
            'restecg': request.form.get('restecg'),
            'thalach': request.form.get('thalach'),
            'exang': request.form.get('exang'),
            'oldpeak': request.form.get('oldpeak'),
            'slope': request.form.get('slope'),
            'ca': request.form.get('ca'),
            'thal': request.form.get('thal'),
        }

        input_df = pd.DataFrame([form_data], columns=[
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ])

        numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        for col in numerical_columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        for col in categorical_columns_inference:
            input_df[col] = input_df[col].astype('object')

        cat_features_idx = [input_df.columns.get_loc(col) for col in categorical_columns_inference]
        test_pool = Pool(input_df, cat_features=cat_features_idx)

        prediction = model.predict(test_pool)
        probability = model.predict_proba(test_pool)[:, 1][0] * 100

    return render_template_string(HTML_TEMPLATE, 
                                prediction=prediction, 
                                probability=probability,
                                form_data=form_data)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Predictor</title>
    <style>
        :root {
            --primary-color: #1a73e8;
            --secondary-color: #34a853;
            --accent-color: #4285f4;
            --light-color: #f8f9fa;
            --dark-color: #202124;
            --danger-color: #ea4335;
            --warning-color: #fbbc05;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #e8f5e9, #e3f2fd);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 15px 30px;
        }
        
        .header-content {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .logo {
            height: 120px;
            width: 120px;
            background: #fff url(/static/logo.jpeg) no-repeat center;
            background-size: contain;
            border-radius: 8px;
            flex-shrink: 0;
        }
        
        h1 {
            font-size: 2.2rem;
            margin-bottom: 5px;
        }
        
        .subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .content {
            display: flex;
            flex-wrap: wrap;
            padding: 20px;
        }
        
        .form-container {
            flex: 1;
            min-width: 300px;
            padding: 20px;
        }
        
        .result-container {
            flex: 1;
            min-width: 300px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
            color: var(--dark-color);
        }
        
        .form-control {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        
        .form-control:focus {
            border-color: var(--accent-color);
            outline: none;
            box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.2);
        }
        
        .form-row {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .form-col {
            flex: 1;
        }
        
        .btn {
            background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
            margin-top: 10px;
        }
        
        .btn:hover {
            background: linear-gradient(135deg, #0d62d1, #3367d6);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .result-card {
            background: linear-gradient(135deg, #e8f5e9, #e3f2fd);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        }
        
        .result-title {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: var(--dark-color);
        }
        
        .prediction {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 15px;
        }
        
        .prediction.positive {
            color: var(--danger-color);
        }
        
        .prediction.negative {
            color: var(--secondary-color);
        }
        
        .prediction.possible {
            color: var(--warning-color);
        }
        
        .probability {
            font-size: 1.2rem;
            margin-bottom: 20px;
            color: #555;
        }
        
        .progress-bar {
            height: 10px;
            background-color: #e0e0e0;
            border-radius: 5px;
            margin-bottom: 10px;
            overflow: hidden;
        }
        
        .progress {
            height: 100%;
            background: linear-gradient(to right, var(--secondary-color), var(--primary-color));
            border-radius: 5px;
            transition: width 0.5s ease-in-out;
        }
        
        .info-section {
            margin-top: 20px;
            padding: 20px;
            background-color: #f5f5f5;
            border-radius: 10px;
            font-size: 0.9rem;
            color: #555;
        }
        
        .info-title {
            font-weight: 600;
            margin-bottom: 10px;
            color: var(--dark-color);
        }
        
        .feature-info {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
        }
        
        .feature-badge {
            background-color: rgba(66, 133, 244, 0.1);
            color: var(--primary-color);
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
            margin-left: 5px;
            cursor: pointer;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 0.8rem;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.9rem;
        }
        
        .threshold-note {
            color: var(--warning-color);
            font-weight: 500;
            margin: 15px 0;
        }
        
        @media (max-width: 768px) {
            .content {
                flex-direction: column;
            }
            
            .form-row {
                flex-direction: column;
                gap: 0;
            }
            
            .header-content {
                flex-direction: column;
                text-align: center;
            }
            
            .logo {
                margin: 0 auto;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-content">
                <div class="logo"></div>
                <div>
                    <h1>Heart Disease Predictor</h1>
                    <p class="subtitle">Enter patient information to predict heart disease risk</p>
                </div>
            </div>
        </header>
        
        <div class="content">
            <div class="form-container">
                <form method="post">
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="age">Age <span class="tooltip">ⓘ<span class="tooltiptext">Patient's age in years</span></span></label>
                                <input type="number" class="form-control" id="age" name="age" required 
                                    value="{{ form_data.age if form_data else '55' }}">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="sex">Sex <span class="tooltip">ⓘ<span class="tooltiptext">0 = Female, 1 = Male</span></span></label>
                                <select class="form-control" id="sex" name="sex" required>
                                    <option value="1" {{ 'selected' if form_data.get('sex') == '1' else '' }}>Male</option>
                                    <option value="0" {{ 'selected' if form_data.get('sex') == '0' else '' }}>Female</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label for="cp">Chest Pain Type <span class="tooltip">ⓘ<span class="tooltiptext">0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic</span></span></label>
                        <select class="form-control" id="cp" name="cp" required>
                            {% for val, text in [('0', 'Typical angina'), ('1', 'Atypical angina'), 
                                                ('2', 'Non-anginal pain'), ('3', 'Asymptomatic')] %}
                                <option value="{{ val }}" {{ 'selected' if form_data.get('cp') == val else '' }}>
                                    {{ text }}
                                </option>
                            {% endfor %}
                        </select>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="trestbps">Resting Blood Pressure <span class="tooltip">ⓘ<span class="tooltiptext">Resting blood pressure in mm Hg</span></span></label>
                                <input type="number" class="form-control" id="trestbps" name="trestbps" required 
                                    value="{{ form_data.trestbps if form_data else '130' }}">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="chol">Cholesterol <span class="tooltip">ⓘ<span class="tooltiptext">Serum cholesterol in mg/dl</span></span></label>
                                <input type="number" class="form-control" id="chol" name="chol" required 
                                    value="{{ form_data.chol if form_data else '230' }}">
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="fbs">Fasting Blood Sugar <span class="tooltip">ⓘ<span class="tooltiptext">Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)</span></span></label>
                                <select class="form-control" id="fbs" name="fbs" required>
                                    <option value="0" {{ 'selected' if form_data.get('fbs') == '0' else '' }}>Less than 120 mg/dl</option>
                                    <option value="1" {{ 'selected' if form_data.get('fbs') == '1' else '' }}>Greater than 120 mg/dl</option>
                                </select>
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="restecg">Resting ECG <span class="tooltip">ⓘ<span class="tooltiptext">0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy</span></span></label>
                                <select class="form-control" id="restecg" name="restecg" required>
                                    {% for val, text in [('0', 'Normal'), ('1', 'ST-T wave abnormality'), ('2', 'Left ventricular hypertrophy')] %}
                                        <option value="{{ val }}" {{ 'selected' if form_data.get('restecg') == val else '' }}>
                                            {{ text }}
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="thalach">Max Heart Rate <span class="tooltip">ⓘ<span class="tooltiptext">Maximum heart rate achieved</span></span></label>
                                <input type="number" class="form-control" id="thalach" name="thalach" required 
                                    value="{{ form_data.thalach if form_data else '150' }}">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="exang">Exercise Induced Angina <span class="tooltip">ⓘ<span class="tooltiptext">Exercise induced angina (1 = yes; 0 = no)</span></span></label>
                                <select class="form-control" id="exang" name="exang" required>
                                    <option value="0" {{ 'selected' if form_data.get('exang') == '0' else '' }}>No</option>
                                    <option value="1" {{ 'selected' if form_data.get('exang') == '1' else '' }}>Yes</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="oldpeak">ST Depression <span class="tooltip">ⓘ<span class="tooltiptext">ST depression induced by exercise relative to rest</span></span></label>
                                <input type="number" class="form-control" id="oldpeak" name="oldpeak" required 
                                    value="{{ form_data.oldpeak if form_data else '1.0' }}" step="0.1">
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="slope">Slope of ST <span class="tooltip">ⓘ<span class="tooltiptext">The slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)</span></span></label>
                                <select class="form-control" id="slope" name="slope" required>
                                    {% for val, text in [('0', 'Upsloping'), ('1', 'Flat'), ('2', 'Downsloping')] %}
                                        <option value="{{ val }}" {{ 'selected' if form_data.get('slope') == val else '' }}>
                                            {{ text }}
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <div class="form-row">
                        <div class="form-col">
                            <div class="form-group">
                                <label for="ca">Number of Major Vessels <span class="tooltip">ⓘ<span class="tooltiptext">Number of major vessels (0-4) colored by fluoroscopy</span></span></label>
                                <select class="form-control" id="ca" name="ca" required>
                                    {% for val in ['0', '1', '2', '3', '4'] %}
                                        <option value="{{ val }}" {{ 'selected' if form_data.get('ca') == val else '' }}>
                                            {{ val }}
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>
                        </div>
                        <div class="form-col">
                            <div class="form-group">
                                <label for="thal">Thalassemia <span class="tooltip">ⓘ<span class="tooltiptext">0: Null, 1: Fixed defect, 2: Normal, 3: Reversible defect</span></span></label>
                                <select class="form-control" id="thal" name="thal" required>
                                    {% for val, text in [('0', 'Null'), ('1', 'Fixed defect'), 
                                                        ('2', 'Normal'), ('3', 'Reversible defect')] %}
                                        <option value="{{ val }}" {{ 'selected' if form_data.get('thal') == val else '' }}>
                                            {{ text }}
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn">Predict Heart Disease Risk</button>
                </form>
            </div>
            
            <div class="result-container">
                {% if prediction is not none %}
                <div class="result-card">
                    <h3 class="result-title">Prediction Result</h3>
                    
                    <div class="prediction {% if prediction == 1 %}positive{% elif prediction == 0 and probability < 60 %}possible{% else %}negative{% endif %}">
                        {% if prediction == 1 %}
                            Positive: Heart Disease Detected
                        {% elif prediction == 0 and probability < 60 %}
                            Possible Heart Disease Detected
                        {% else %}
                            Negative: No Heart Disease Detected
                        {% endif %}
                    </div>

                    {% if prediction == 0 and probability < 60 %}
                        <div class="threshold-note">
                            Note: Probability is below our 60% confidence threshold. Further clinical evaluation recommended.
                        </div>
                    {% endif %}

                    <div class="probability">
                        Confidence: {{ "%.1f"|format(probability) }}%
                    </div>
                    
                    <div class="progress-bar">
                        <div class="progress" style="width: {{ probability }}%;"></div>
                    </div>
                    
                    <div class="info-section">
                        <div class="info-title">What does this mean?</div>
                        {% if prediction == 1 %}
                            <p>The model predicts that this patient has a high risk of heart disease. Further clinical evaluation is recommended.</p>
                        {% elif prediction == 0 and probability < 60 %}
                            <p>The model predicts a possible risk of heart disease due to low confidence. Further clinical evaluation is strongly recommended.</p>
                        {% else %}
                            <p>The model predicts that this patient has a low risk of heart disease. However, regular check-ups are still recommended.</p>
                        {% endif %}
                        
                        <div class="info-title">Important factors in this prediction:</div>
                        <div class="feature-info">
                            <span class="feature-badge">Age</span>
                            <span class="feature-badge">Chest Pain Type</span>
                            <span class="feature-badge">Max Heart Rate</span>
                            <span class="feature-badge">ST Depression</span>
                            <span class="feature-badge">Major Vessels</span>
                        </div>
                    </div>
                </div>
                {% else %}
                <div class="result-card">
                    <h3 class="result-title">Heart Disease Prediction</h3>
                    <p>Fill in the patient information and click "Predict" to get a heart disease risk assessment.</p>
                    
                    <div class="info-section">
                        <div class="info-title">About this tool</div>
                        <p>This prediction tool uses AI trained on a custom Heart Disease dataset to assess the risk of heart disease based on various clinical parameters.</p>
                        <p>The model analyzes factors such as age, sex, chest pain type, blood pressure, cholesterol levels, and other clinical measurements to provide a risk assessment.</p>
                        <p><strong>Note:</strong> This tool is used to help doctors make decisions and should not be used in itself for clinical decision making.</p>
                    </div>
                </div>
                {% endif %}
            </div>
        </div>
        
        <footer>
            <p>Heart Disease Prediction Tool | Powered by AI</p>
            <p>Disclaimer: This tool is used to help doctors make decisions and should not be used in itself for clinical decision making.</p>
        </footer>
    </div>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)