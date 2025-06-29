from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os
from data_preprocessing import HeartDiseaseDataProcessor

app = Flask(__name__)
app.secret_key = 'heart_disease_assessment_secret_key'

# Global variable to store the processor
processor = None

def load_model():
    """Load the trained model"""
    global processor
    if processor is None:
        processor = HeartDiseaseDataProcessor()
        if os.path.exists('heart_disease_model.pkl'):
            processor.load_model()
        else:
            # Train model if it doesn't exist
            print("Training new model...")
            df = processor.create_sample_data()
            X_train, X_test, y_train, y_test = processor.preprocess_data(df)
            processor.train_model(X_train, y_train)
            processor.save_model()
            processor.create_visualizations(df)

@app.route('/')
def index():
    """Main page with heart disease risk assessment form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        # Get form data
        data = {
            'age': int(request.form['age']),
            'sex': int(request.form['sex']),
            'chest_pain_type': int(request.form['chest_pain_type']),
            'resting_bp': int(request.form['resting_bp']),
            'cholesterol': int(request.form['cholesterol']),
            'fasting_bs': int(request.form['fasting_bs']),
            'resting_ecg': int(request.form['resting_ecg']),
            'max_hr': int(request.form['max_hr']),
            'exercise_angina': int(request.form['exercise_angina']),
            'oldpeak': float(request.form['oldpeak']),
            'st_slope': int(request.form['st_slope'])
        }
        
        # Validate input ranges
        if not (25 <= data['age'] <= 85):
            flash('Age must be between 25 and 85 years', 'error')
            return redirect(url_for('index'))
        
        if not (90 <= data['resting_bp'] <= 200):
            flash('Resting blood pressure must be between 90 and 200 mmHg', 'error')
            return redirect(url_for('index'))
        
        if not (100 <= data['cholesterol'] <= 400):
            flash('Cholesterol must be between 100 and 400 mg/dL', 'error')
            return redirect(url_for('index'))
        
        if not (60 <= data['max_hr'] <= 202):
            flash('Maximum heart rate must be between 60 and 202 bpm', 'error')
            return redirect(url_for('index'))
        
        if not (-2.6 <= data['oldpeak'] <= 6.2):
            flash('ST depression must be between -2.6 and 6.2 mm', 'error')
            return redirect(url_for('index'))
        
        # Make prediction
        features = list(data.values())
        prediction, probability = processor.predict_risk(features)
        
        # Prepare result
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        confidence = probability[1] if prediction == 1 else probability[0]
        
        result = {
            'prediction': prediction,
            'risk_level': risk_level,
            'confidence': f"{confidence:.2%}",
            'data': data
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        flash(f'Error processing request: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/about')
def about():
    """About page with information about heart disease risk factors"""
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard with model statistics and visualizations"""
    if processor is None:
        load_model()
    
    # Get feature importance
    importance_df = processor.get_feature_importance()
    
    # Prepare dashboard data
    dashboard_data = {
        'feature_importance': importance_df.to_dict('records'),
        'total_features': len(importance_df),
        'model_accuracy': '85-90%'  # This would be calculated from actual test data
    }
    
    return render_template('dashboard.html', data=dashboard_data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
                          'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina', 
                          'oldpeak', 'st_slope']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Make prediction
        features = [data[field] for field in required_fields]
        prediction, probability = processor.predict_risk(features)
        
        return jsonify({
            'prediction': int(prediction),
            'risk_level': "High Risk" if prediction == 1 else "Low Risk",
            'confidence': float(probability[1] if prediction == 1 else probability[0]),
            'probabilities': {
                'low_risk': float(probability[0]),
                'high_risk': float(probability[1])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': processor is not None})

if __name__ == '__main__':
    # Load model on startup
    load_model()
    print("Heart Disease Risk Assessment App is running!")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 