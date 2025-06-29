import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class HeartDiseaseDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = None
        
    def create_sample_data(self):
        """Create sample heart disease dataset with realistic values"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic heart disease data
        data = {
            'age': np.random.normal(55, 15, n_samples).astype(int),
            'sex': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
            'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples),  # 0-3 pain types
            'resting_bp': np.random.normal(130, 20, n_samples).astype(int),
            'cholesterol': np.random.normal(200, 50, n_samples).astype(int),
            'fasting_bs': np.random.choice([0, 1], n_samples),  # 0: <120, 1: >120
            'resting_ecg': np.random.choice([0, 1, 2], n_samples),  # ECG results
            'max_hr': np.random.normal(150, 25, n_samples).astype(int),
            'exercise_angina': np.random.choice([0, 1], n_samples),  # 0: No, 1: Yes
            'oldpeak': np.random.normal(1.5, 2, n_samples),  # ST depression
            'st_slope': np.random.choice([0, 1, 2], n_samples),  # ST slope
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic ranges
        df['age'] = df['age'].clip(25, 85)
        df['resting_bp'] = df['resting_bp'].clip(90, 200)
        df['cholesterol'] = df['cholesterol'].clip(100, 400)
        df['max_hr'] = df['max_hr'].clip(60, 202)
        df['oldpeak'] = df['oldpeak'].clip(-2.6, 6.2)
        
        # Create target variable based on risk factors
        risk_score = (
            (df['age'] - 25) / 60 * 0.3 +
            df['sex'] * 0.2 +
            (df['resting_bp'] - 90) / 110 * 0.2 +
            (df['cholesterol'] - 100) / 300 * 0.15 +
            df['fasting_bs'] * 0.3 +
            df['exercise_angina'] * 0.4 +
            (df['oldpeak'] + 2.6) / 8.8 * 0.25
        )
        
        df['heart_disease'] = (risk_score > 0.5).astype(int)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Separate features and target
        X = df.drop('heart_disease', axis=1)
        y = df['heart_disease']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        self.model.fit(X_train, y_train)
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model performance"""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy, y_pred
    
    def save_model(self, filename='heart_disease_model.pkl'):
        """Save the trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='heart_disease_model.pkl'):
        """Load the trained model and scaler"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filename}")
    
    def predict_risk(self, features):
        """Predict heart disease risk for a single patient"""
        # Ensure features are in the correct order
        feature_df = pd.DataFrame([features], columns=self.feature_names)
        
        # Scale the features
        features_scaled = self.scaler.transform(feature_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def create_visualizations(self, df, save_path='static/'):
        """Create and save visualizations"""
        # Create directory if it doesn't exist
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Heart Disease Distribution
        plt.figure(figsize=(10, 6))
        df['heart_disease'].value_counts().plot(kind='bar')
        plt.title('Heart Disease Distribution')
        plt.xlabel('Heart Disease (0: No, 1: Yes)')
        plt.ylabel('Count')
        plt.savefig(f'{save_path}heart_disease_distribution.png')
        plt.close()
        
        # 2. Age Distribution by Heart Disease
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        df[df['heart_disease'] == 0]['age'].hist(alpha=0.7, label='No Heart Disease')
        df[df['heart_disease'] == 1]['age'].hist(alpha=0.7, label='Heart Disease')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Age Distribution by Heart Disease')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        df[df['heart_disease'] == 0]['resting_bp'].hist(alpha=0.7, label='No Heart Disease')
        df[df['heart_disease'] == 1]['resting_bp'].hist(alpha=0.7, label='Heart Disease')
        plt.xlabel('Resting Blood Pressure')
        plt.ylabel('Frequency')
        plt.title('Blood Pressure Distribution by Heart Disease')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}age_bp_distribution.png')
        plt.close()
        
        # 3. Feature Importance
        importance_df = self.get_feature_importance()
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'{save_path}feature_importance.png')
        plt.close()
        
        # 4. Correlation Matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{save_path}correlation_matrix.png')
        plt.close()

def main():
    """Main function to train and save the model"""
    processor = HeartDiseaseDataProcessor()
    
    # Create and preprocess data
    print("Creating sample dataset...")
    df = processor.create_sample_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = processor.preprocess_data(df)
    
    # Train model
    print("Training Random Forest model...")
    processor.train_model(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    accuracy, y_pred = processor.evaluate_model(X_test, y_test)
    
    # Save model
    processor.save_model()
    
    # Create visualizations
    print("Creating visualizations...")
    processor.create_visualizations(df)
    
    print("Model training completed successfully!")
    print(f"Final Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main() 