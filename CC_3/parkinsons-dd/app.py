from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
import numpy as np
import joblib
import logging
import pandas as pd
import io
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages
model = joblib.load("model.pkl")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define feature names in the same order as they appear in the dataset
FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)", 
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]

# Feature descriptions for tooltips
FEATURE_DESCRIPTIONS = {
    "MDVP:Fo(Hz)": "Average vocal fundamental frequency",
    "MDVP:Fhi(Hz)": "Maximum vocal fundamental frequency",
    "MDVP:Flo(Hz)": "Minimum vocal fundamental frequency",
    "MDVP:Jitter(%)": "Jitter percentage - measure of frequency variation",
    "MDVP:Jitter(Abs)": "Absolute jitter value",
    "MDVP:RAP": "Relative average perturbation",
    "MDVP:PPQ": "Five-point period perturbation quotient",
    "Jitter:DDP": "Average absolute difference of differences between consecutive periods",
    "MDVP:Shimmer": "Shimmer - measure of amplitude variation",
    "MDVP:Shimmer(dB)": "Shimmer in decibels",
    "Shimmer:APQ3": "Three-point amplitude perturbation quotient",
    "Shimmer:APQ5": "Five-point amplitude perturbation quotient",
    "MDVP:APQ": "Amplitude perturbation quotient",
    "Shimmer:DDA": "Average absolute difference between consecutive differences between amplitudes",
    "NHR": "Noise-to-harmonics ratio",
    "HNR": "Harmonics-to-noise ratio",
    "RPDE": "Recurrence period density entropy",
    "DFA": "Detrended fluctuation analysis",
    "spread1": "Nonlinear measure of fundamental frequency variation",
    "spread2": "Nonlinear measure of fundamental frequency variation",
    "D2": "Correlation dimension",
    "PPE": "Pitch period entropy"
}

@app.route('/')
def home():
    return render_template('index.html', feature_descriptions=FEATURE_DESCRIPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate that all required fields are present
        missing_fields = []
        input_data = []
        
        for feature in FEATURE_NAMES:
            if feature not in request.form:
                missing_fields.append(feature)
            else:
                value = request.form[feature].strip()
                if not value:
                    missing_fields.append(feature)
                else:
                    try:
                        # Validate numeric input
                        float_value = float(value)
                        # Check for reasonable ranges (basic validation)
                        if float_value < 0 and feature not in ['spread1', 'spread2', 'D2']:
                            flash(f"Error: {feature} should be a positive number", "error")
                            return render_template('index.html', feature_descriptions=FEATURE_DESCRIPTIONS)
                        input_data.append(float_value)
                    except ValueError:
                        flash(f"Error: {feature} must be a valid number", "error")
                        return render_template('index.html', feature_descriptions=FEATURE_DESCRIPTIONS)
        
        if missing_fields:
            flash(f"Error: Please fill in all fields. Missing: {', '.join(missing_fields)}", "error")
            return render_template('index.html', feature_descriptions=FEATURE_DESCRIPTIONS)
        
        # Make prediction
        input_array = np.array([input_data])
        prediction = model.predict(input_array)[0]
        
        # Get prediction probability if available
        try:
            prediction_proba = model.predict_proba(input_array)[0]
            confidence = max(prediction_proba) * 100
        except:
            confidence = None
        
        # Log the prediction
        logger.info(f"Prediction made: {prediction}, Confidence: {confidence}, Features: {dict(zip(FEATURE_NAMES, input_data))}")
        
        # Determine result and explanation
        if prediction == 1:
            result = "Positive for Parkinson's Disease"
            explanation = "This result suggests the presence of Parkinson's disease symptoms. Please consult with a healthcare professional for proper diagnosis and treatment."
        else:
            result = "Negative for Parkinson's Disease"
            explanation = "This result suggests no significant Parkinson's disease symptoms detected. However, this is not a definitive diagnosis - consult a healthcare professional for any concerns."
        
        return render_template('index.html', 
                             prediction_text=result,
                             confidence=confidence,
                             explanation=explanation,
                             feature_descriptions=FEATURE_DESCRIPTIONS,
                             input_data=dict(zip(FEATURE_NAMES, input_data)))
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        flash("An error occurred during prediction. Please try again.", "error")
        return render_template('index.html', feature_descriptions=FEATURE_DESCRIPTIONS)

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('home'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('home'))
        
        if not file.filename.endswith('.csv'):
            flash('Please upload a CSV file', 'error')
            return redirect(url_for('home'))
        
        # Read CSV file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f'Error reading CSV file: {str(e)}', 'error')
            return redirect(url_for('home'))
        
        # Validate CSV structure
        missing_features = [feature for feature in FEATURE_NAMES if feature not in df.columns]
        if missing_features:
            flash(f'Missing required features in CSV: {", ".join(missing_features)}', 'error')
            return redirect(url_for('home'))
        
        # Process each row
        results = []
        for index, row in df.iterrows():
            try:
                # Extract features in correct order
                input_data = [float(row[feature]) for feature in FEATURE_NAMES]
                input_array = np.array([input_data])
                
                # Make prediction
                prediction = model.predict(input_array)[0]
                
                # Get confidence if available
                try:
                    prediction_proba = model.predict_proba(input_array)[0]
                    confidence = max(prediction_proba) * 100
                except:
                    confidence = None
                
                results.append({
                    'row': index + 1,
                    'prediction': 'Positive' if prediction == 1 else 'Negative',
                    'confidence': confidence,
                    'features': dict(zip(FEATURE_NAMES, input_data))
                })
                
            except Exception as e:
                results.append({
                    'row': index + 1,
                    'prediction': 'Error',
                    'confidence': None,
                    'error': str(e)
                })
        
        # Log batch prediction
        logger.info(f"Batch prediction completed: {len(results)} records processed")
        
        return render_template('batch_results.html', 
                             results=results, 
                             total_records=len(results),
                             feature_descriptions=FEATURE_DESCRIPTIONS)
    
    except Exception as e:
        logger.error(f"Error during CSV upload: {str(e)}")
        flash(f'Error processing CSV file: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        missing_features = [feature for feature in FEATURE_NAMES if feature not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        # Extract features in correct order
        input_data = []
        for feature in FEATURE_NAMES:
            try:
                value = float(data[feature])
                input_data.append(value)
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid value for {feature}'}), 400
        
        # Make prediction
        input_array = np.array([input_data])
        prediction = model.predict(input_array)[0]
        
        # Get confidence if available
        try:
            prediction_proba = model.predict_proba(input_array)[0]
            confidence = max(prediction_proba) * 100
        except:
            confidence = None
        
        # Log API call
        logger.info(f"API prediction: {prediction}, Confidence: {confidence}")
        
        return jsonify({
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'confidence': confidence,
            'features': dict(zip(FEATURE_NAMES, input_data))
        })
    
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
