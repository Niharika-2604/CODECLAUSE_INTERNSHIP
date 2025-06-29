#!/usr/bin/env python3
"""
Simple test script for the Parkinson's Disease Detection application
"""

import requests
import json
import sys

def test_api():
    """Test the API endpoint with sample data"""
    
    # Sample data from the dataset
    sample_data = {
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
    }
    
    try:
        # Test API endpoint
        response = requests.post(
            'http://localhost:5000/api/predict',
            json=sample_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test PASSED")
            print(f"Prediction: {result.get('prediction')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}%")
            return True
        else:
            print(f"❌ API Test FAILED - Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ API Test FAILED - Could not connect to server")
        print("Make sure the Flask app is running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ API Test FAILED - Error: {str(e)}")
        return False

def test_web_interface():
    """Test if the web interface is accessible"""
    
    try:
        response = requests.get('http://localhost:5000/', timeout=10)
        
        if response.status_code == 200:
            print("✅ Web Interface Test PASSED")
            return True
        else:
            print(f"❌ Web Interface Test FAILED - Status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Web Interface Test FAILED - Could not connect to server")
        print("Make sure the Flask app is running on http://localhost:5000")
        return False
    except Exception as e:
        print(f"❌ Web Interface Test FAILED - Error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Parkinson's Disease Detection Application")
    print("=" * 50)
    
    # Test web interface
    web_test = test_web_interface()
    
    # Test API
    api_test = test_api()
    
    print("=" * 50)
    if web_test and api_test:
        print("🎉 All tests PASSED! Application is working correctly.")
        sys.exit(0)
    else:
        print("💥 Some tests FAILED. Please check the application.")
        sys.exit(1)

if __name__ == "__main__":
    main() 