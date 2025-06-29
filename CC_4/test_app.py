#!/usr/bin/env python3
"""
Test script for Heart Disease Risk Assessment Application
"""

import requests
import json
import time
from data_preprocessing import HeartDiseaseDataProcessor

def test_model_training():
    """Test model training and saving"""
    print("🧪 Testing Model Training...")
    
    try:
        processor = HeartDiseaseDataProcessor()
        
        # Create sample data
        df = processor.create_sample_data()
        print(f"✅ Created dataset with {len(df)} samples")
        
        # Preprocess data
        X_train, X_test, y_train, y_test = processor.preprocess_data(df)
        print(f"✅ Data preprocessed - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train model
        processor.train_model(X_train, y_train)
        print("✅ Model trained successfully")
        
        # Evaluate model
        accuracy, y_pred = processor.evaluate_model(X_test, y_test)
        print(f"✅ Model accuracy: {accuracy:.4f}")
        
        # Save model
        processor.save_model()
        print("✅ Model saved successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {str(e)}")
        return False

def test_prediction():
    """Test prediction functionality"""
    print("\n🧪 Testing Prediction Functionality...")
    
    try:
        processor = HeartDiseaseDataProcessor()
        processor.load_model()
        
        # Test case 1: Low risk patient
        low_risk_features = [45, 0, 0, 120, 180, 0, 0, 160, 0, 0.0, 0]
        prediction, probability = processor.predict_risk(low_risk_features)
        print(f"✅ Low risk test - Prediction: {prediction}, Probability: {probability}")
        
        # Test case 2: High risk patient
        high_risk_features = [65, 1, 2, 160, 280, 1, 1, 140, 1, 2.5, 2]
        prediction, probability = processor.predict_risk(high_risk_features)
        print(f"✅ High risk test - Prediction: {prediction}, Probability: {probability}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {str(e)}")
        return False

def test_api_endpoints():
    """Test Flask API endpoints"""
    print("\n🧪 Testing API Endpoints...")
    
    base_url = "http://localhost:5000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test prediction API
        test_data = {
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
        
        response = requests.post(
            f"{base_url}/api/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API prediction successful: {result['risk_level']}")
        else:
            print(f"❌ API prediction failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Make sure it's running on localhost:5000")
        return False
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

def test_web_pages():
    """Test web page accessibility"""
    print("\n🧪 Testing Web Pages...")
    
    base_url = "http://localhost:5000"
    pages = ["/", "/about", "/dashboard"]
    
    try:
        for page in pages:
            response = requests.get(f"{base_url}{page}")
            if response.status_code == 200:
                print(f"✅ {page} page accessible")
            else:
                print(f"❌ {page} page failed: {response.status_code}")
                return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Make sure it's running on localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Web page test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Heart Disease Risk Assessment Tests\n")
    
    tests = [
        ("Model Training", test_model_training),
        ("Prediction Functionality", test_prediction),
        ("API Endpoints", test_api_endpoints),
        ("Web Pages", test_web_pages)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Application is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    print("\n" + "="*50)
    print("📝 NEXT STEPS")
    print("="*50)
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the application: python app.py")
    print("3. Open browser: http://localhost:5000")
    print("4. Start using the Heart Disease Risk Assessment!")

if __name__ == "__main__":
    main() 