# Test script to verify feature names are working correctly
import pandas as pd

# Read the dataset to verify feature order
df = pd.read_csv('parkinsons.csv')

# Get feature columns (excluding 'name' and 'status')
feature_columns = [col for col in df.columns if col not in ['name', 'status']]

print("Feature names in dataset order:")
for i, feature in enumerate(feature_columns):
    print(f"{i+1:2d}. {feature}")

print(f"\nTotal features: {len(feature_columns)}")

# Verify this matches our FEATURE_NAMES list
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

print("\nVerifying feature order matches:")
for i, (dataset_feature, app_feature) in enumerate(zip(feature_columns, FEATURE_NAMES)):
    if dataset_feature == app_feature:
        print(f"✓ {i+1:2d}. {dataset_feature}")
    else:
        print(f"✗ {i+1:2d}. Dataset: {dataset_feature} | App: {app_feature}")

print("\nFeature names are correctly mapped!" if feature_columns == FEATURE_NAMES else "\nERROR: Feature names don't match!") 