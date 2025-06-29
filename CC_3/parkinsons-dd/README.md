# Parkinson's Disease Detection System

AI-powered web application for early detection of Parkinson's disease using voice analysis.

## Features

- **Single Patient Analysis**: Input 22 voice parameters manually
- **Batch Processing**: Upload CSV files for multiple patients
- **REST API**: Programmatic access
- **Modern UI**: Responsive design with tooltips and validation
- **Confidence Scoring**: Shows prediction probability

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install flask numpy pandas scikit-learn joblib
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open browser**: Navigate to `http://localhost:5000`

## Usage

### Single Analysis
- Go to "Single Analysis" tab
- Enter the 22 voice parameters
- Click "Analyze Voice Data"
- View results with confidence score

### Batch Analysis
- Go to "Batch Analysis" tab
- Upload CSV file with required columns
- View results for all patients

### API Usage
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"MDVP:Fo(Hz)": 119.992, "MDVP:Fhi(Hz)": 157.302, ...}'
```

## Required CSV Columns
```
MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs), 
MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB), 
Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA, NHR, HNR, 
RPDE, DFA, spread1, spread2, D2, PPE
```

## Medical Disclaimer
This application is for educational purposes only. Consult healthcare professionals for medical decisions.

## File Structure
```
parkinsons-dd/
├── app.py                 # Main Flask application
├── model.pkl             # Trained ML model
├── parkinsons.csv        # Training dataset
├── templates/            # HTML templates
└── README.md            # This file
```

## 📊 Voice Analysis Parameters

The system analyzes 22 voice parameters:

### Fundamental Frequency Parameters
- **MDVP:Fo(Hz)**: Average vocal fundamental frequency
- **MDVP:Fhi(Hz)**: Maximum vocal fundamental frequency  
- **MDVP:Flo(Hz)**: Minimum vocal fundamental frequency

### Jitter Parameters (Frequency Variation)
- **MDVP:Jitter(%)**: Jitter percentage - measure of frequency variation
- **MDVP:Jitter(Abs)**: Absolute jitter value
- **MDVP:RAP**: Relative average perturbation
- **MDVP:PPQ**: Five-point period perturbation quotient
- **Jitter:DDP**: Average absolute difference of differences between consecutive periods

### Shimmer Parameters (Amplitude Variation)
- **MDVP:Shimmer**: Shimmer - measure of amplitude variation
- **MDVP:Shimmer(dB)**: Shimmer in decibels
- **Shimmer:APQ3**: Three-point amplitude perturbation quotient
- **Shimmer:APQ5**: Five-point amplitude perturbation quotient
- **MDVP:APQ**: Amplitude perturbation quotient
- **Shimmer:DDA**: Average absolute difference between consecutive differences between amplitudes

### Noise Parameters
- **NHR**: Noise-to-harmonics ratio
- **HNR**: Harmonics-to-noise ratio

### Nonlinear Dynamic Complexity Measures
- **RPDE**: Recurrence period density entropy
- **DFA**: Detrended fluctuation analysis
- **spread1**: Nonlinear measure of fundamental frequency variation
- **spread2**: Nonlinear measure of fundamental frequency variation
- **D2**: Correlation dimension
- **PPE**: Pitch period entropy

## 📁 File Structure

```
parkinsons-dd/
├── app.py                 # Main Flask application
├── model.pkl             # Trained machine learning model
├── parkinsons.csv        # Training dataset
├── model.py              # Model training script (if available)
├── templates/
│   ├── index.html        # Main application interface
│   └── batch_results.html # Batch processing results
├── static/
│   └── style.css         # Additional styles (if any)
└── README.md             # This file
```

## 🔧 Usage

### Single Patient Analysis

1. Navigate to the "Single Analysis" tab
2. Enter the 22 voice parameters
3. Click "Analyze Voice Data"
4. View results with confidence score and explanation

### Batch Analysis

1. Navigate to the "Batch Analysis" tab
2. Prepare a CSV file with the required columns
3. Drag & drop or browse to upload the file
4. Click "Process CSV File"
5. View comprehensive results for all patients

### API Usage

Make a POST request to `/api/predict` with JSON data:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    "MDVP:Flo(Hz)": 74.997,
    "MDVP:Jitter(%)": 0.00784,
    "MDVP:Jitter(Abs)": 0.00007,
    "MDVP:RAP": 0.0037,
    "MDVP:PPQ": 0.00554,
    "Jitter:DDP": 0.01109,
    "MDVP:Shimmer": 0.04374,
    "MDVP:Shimmer(dB)": 0.426,
    "Shimmer:APQ3": 0.02182,
    "Shimmer:APQ5": 0.0313,
    "MDVP:APQ": 0.02971,
    "Shimmer:DDA": 0.06545,
    "NHR": 0.02211,
    "HNR": 21.033,
    "RPDE": 0.414783,
    "DFA": 0.815285,
    "spread1": -4.813031,
    "spread2": 0.266482,
    "D2": 2.301442,
    "PPE": 0.284654
  }'
```

Response:
```json
{
  "prediction": "Positive",
  "confidence": 95.2,
  "features": {
    "MDVP:Fo(Hz)": 119.992,
    "MDVP:Fhi(Hz)": 157.302,
    ...
  }
}
```

## 📋 CSV Format for Batch Analysis

Your CSV file must include these exact column names:

```csv
MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE
119.992,157.302,74.997,0.00784,0.00007,0.0037,0.00554,0.01109,0.04374,0.426,0.02182,0.0313,0.02971,0.06545,0.02211,21.033,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654
```

## 🏥 Medical Disclaimer

**Important**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## 🔒 Privacy & Security

- All data processing is done locally
- No patient data is stored permanently
- Logs are kept for debugging purposes only
- API calls are logged for monitoring

## 🛠️ Development

### Adding New Features

1. **Model Improvements**: Update `model.py` to retrain with new data
2. **UI Enhancements**: Modify templates in the `templates/` directory
3. **API Extensions**: Add new routes in `app.py`

### Testing

Run the test script to verify feature mapping:
```bash
python test_features.py
```

### Deployment

For production deployment, consider:
- Using a production WSGI server (Gunicorn, uWSGI)
- Setting up proper logging
- Implementing user authentication
- Adding rate limiting for API endpoints
- Using environment variables for configuration

## 📈 Performance

- **Accuracy**: Model achieves high accuracy on the Parkinson's dataset
- **Speed**: Single predictions complete in milliseconds
- **Scalability**: Batch processing handles hundreds of records efficiently

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 📞 Support

For questions or issues:
1. Check the documentation above
2. Review the error logs
3. Open an issue in the repository

---

**Built with ❤️ for medical research and education** 