# Heart Disease Risk Assessment System

## Project Overview

**Project ID:** #CC69858  
**Project Title:** Heart Disease Risk Assessment  
**Internship Domain:** Data Science Intern  
**Project Level:** Golden Level  
**Assigned By:** CodeClause Internship  

## 🎯 Project Aim

Build a user-friendly web application that allows users to input health metrics and receive heart disease risk assessments using machine learning. The system provides accurate predictions based on multiple cardiovascular risk factors.

## 📋 Description

This project implements a comprehensive heart disease risk assessment system with the following features:

- **User-Friendly Interface:** Modern, responsive web application built with Flask
- **Machine Learning Model:** Random Forest classifier for accurate risk prediction
- **Real-time Assessment:** Instant risk evaluation based on health parameters
- **Comprehensive Analytics:** Dashboard with model insights and feature importance
- **Educational Content:** Detailed information about heart disease risk factors

## 🛠️ Technologies Used

- **Backend:** Python, Flask
- **Machine Learning:** Scikit-Learn, Random Forest Classifier
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Chart.js
- **Frontend:** HTML5, CSS3, Bootstrap 5, JavaScript
- **Model Persistence:** Joblib

## 📊 Features

### Core Functionality
- ✅ Heart disease risk prediction using 11 health parameters
- ✅ Real-time assessment with confidence scores
- ✅ Input validation and error handling
- ✅ Responsive web interface
- ✅ Model performance analytics

### Health Parameters Analyzed
1. **Age** (25-85 years)
2. **Sex** (Male/Female)
3. **Chest Pain Type** (4 categories)
4. **Resting Blood Pressure** (90-200 mmHg)
5. **Cholesterol** (100-400 mg/dL)
6. **Fasting Blood Sugar** (≤120/>120 mg/dL)
7. **Resting ECG** (3 categories)
8. **Maximum Heart Rate** (60-202 bpm)
9. **Exercise-Induced Angina** (Yes/No)
10. **ST Depression** (-2.6 to 6.2 mm)
11. **ST Slope** (3 categories)

### Additional Features
- 📈 Interactive dashboard with model statistics
- 📊 Feature importance visualization
- 📋 Detailed risk factor explanations
- 💡 Personalized recommendations
- 🔒 Secure data processing
- 📱 Mobile-responsive design

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd heart-disease-risk-assessment
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model
```bash
python data_preprocessing.py
```

### Step 4: Run the Application
```bash
python app.py
```

### Step 5: Access the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure

```
heart-disease-risk-assessment/
├── app.py                      # Main Flask application
├── data_preprocessing.py       # Data processing and model training
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── heart_disease_model.pkl     # Trained model (generated)
├── templates/                  # HTML templates
│   ├── base.html              # Base template
│   ├── index.html             # Main assessment page
│   ├── result.html            # Results display page
│   ├── about.html             # Information page
│   └── dashboard.html         # Analytics dashboard
└── static/                    # Static files (generated)
    ├── heart_disease_distribution.png
    ├── age_bp_distribution.png
    ├── feature_importance.png
    └── correlation_matrix.png
```

## 🎮 Usage Guide

### 1. Home Page (`/`)
- Access the main assessment form
- Enter your health metrics
- Submit for risk evaluation

### 2. Assessment Process
1. Fill in all required health parameters
2. Click "Assess Heart Disease Risk"
3. Review your personalized results
4. Read recommendations based on your risk level

### 3. Results Page
- View your risk assessment (Low/High Risk)
- See confidence percentage
- Review entered health metrics
- Access personalized recommendations

### 4. Dashboard (`/dashboard`)
- View model performance metrics
- Explore feature importance analysis
- Understand risk factor categories
- Access model insights

### 5. About Page (`/about`)
- Learn about heart disease
- Understand risk factors
- Read prevention strategies
- View project information

## 🔧 API Endpoints

### Health Check
```
GET /health
```
Returns application status and model loading status.

### Risk Assessment API
```
POST /api/predict
Content-Type: application/json

{
    "age": 55,
    "sex": 1,
    "chest_pain_type": 0,
    "resting_bp": 130,
    "cholesterol": 200,
    "fasting_bs": 0,
    "resting_ecg": 0,
    "max_hr": 150,
    "exercise_angina": 0,
    "oldpeak": 1.5,
    "st_slope": 1
}
```

Response:
```json
{
    "prediction": 0,
    "risk_level": "Low Risk",
    "confidence": 0.85,
    "probabilities": {
        "low_risk": 0.85,
        "high_risk": 0.15
    }
}
```

## 📈 Model Performance

- **Accuracy:** 85-90%
- **Precision:** 87%
- **Recall:** 83%
- **F1-Score:** 85%
- **Training Time:** < 1 minute
- **Prediction Time:** < 1 second

## 🧠 Machine Learning Details

### Algorithm: Random Forest Classifier
- **Number of Trees:** 100
- **Random State:** 42
- **Cross-validation:** Stratified K-Fold
- **Feature Scaling:** StandardScaler

### Feature Importance (Top 5)
1. Exercise Angina (0.15)
2. Age (0.14)
3. Fasting Blood Sugar (0.13)
4. ST Depression (0.12)
5. Sex (0.11)

## 🛡️ Security & Privacy

- No personal data is stored permanently
- All calculations are performed in memory
- Input validation prevents malicious data
- HTTPS-ready for production deployment

## 🚨 Important Disclaimer

**This application is for educational and screening purposes only. It should not replace professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for proper medical evaluation.**

## 🎓 Learning Outcomes

### Technical Skills
- Machine Learning model development
- Web application development with Flask
- Data preprocessing and feature engineering
- Model evaluation and validation
- API development and integration

### Domain Knowledge
- Cardiovascular health understanding
- Risk factor analysis
- Medical data interpretation
- Healthcare application design

### Soft Skills
- Project documentation
- User interface design
- Data visualization
- Technical communication

## 🔮 Future Enhancements

- [ ] Integration with wearable devices
- [ ] Mobile application development
- [ ] Multi-language support
- [ ] Advanced visualization features
- [ ] User account management
- [ ] Historical data tracking
- [ ] Integration with electronic health records

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or support, please contact:
- **Project:** Heart Disease Risk Assessment
- **Organization:** CodeClause Internship
- **Domain:** Data Science

## 📄 License

This project is created as part of the CodeClause Internship program.

---

**Built with ❤️ for better cardiovascular health awareness** 