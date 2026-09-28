"""
Script to train the loan approval model and save it as pickle file
along with the label encoders and scaler for use in Django application.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("LOAN APPROVAL MODEL - TRAINING AND EXPORT")
print("=" * 60)

# Load the dataset
df = pd.read_csv('loan.csv')
print(f"\n✓ Loaded dataset with {len(df)} records")

# Create a copy for preprocessing
df_processed = df.copy()

# Encode categorical variables
label_encoders = {}
categorical_columns = ['gender', 'occupation', 'education_level', 'marital_status', 'loan_status']

for col in categorical_columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

print("\n✓ Encoded categorical variables")
print("\nLabel Encoders:")
for col, le in label_encoders.items():
    print(f"  {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Create additional features
df_processed['debt_to_income_ratio'] = (df_processed['credit_score'] * 100) / df_processed['income']
df_processed['age_income_interaction'] = df_processed['age'] * df_processed['income'] / 1000

# Normalize numeric features
scaler = StandardScaler()
numeric_columns = ['age', 'income', 'credit_score', 'debt_to_income_ratio', 'age_income_interaction']
df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])

print("\n✓ Created features and normalized data")

# Separate features and target
X = df_processed.drop('loan_status', axis=1)
y = df_processed['loan_status']

# Apply SMOTE to balance the classes
print("\nBefore SMOTE:")
print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("\nAfter SMOTE:")
print(f"  Class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")

# Split the balanced data
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train Logistic Regression
print("\n" + "=" * 60)
print("Training Logistic Regression...")
print("=" * 60)

log_reg = LogisticRegression(C=0.5, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42)
cv_scores_lr = cross_val_score(log_reg, X_balanced, y_balanced, cv=cv, scoring='accuracy')
cv_f1_lr = cross_val_score(log_reg, X_balanced, y_balanced, cv=cv, scoring='f1')
log_reg.fit(X_train, y_train)

print(f"  CV Accuracy: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std() * 2:.4f})")
print(f"  CV F1-Score: {cv_f1_lr.mean():.4f} (+/- {cv_f1_lr.std() * 2:.4f})")

# Train Random Forest
print("\n" + "=" * 60)
print("Training Random Forest...")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=50, 
    max_depth=5, 
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42, 
    n_jobs=-1
)
cv_scores_rf = cross_val_score(rf_model, X_balanced, y_balanced, cv=cv, scoring='accuracy')
cv_f1_rf = cross_val_score(rf_model, X_balanced, y_balanced, cv=cv, scoring='f1')
rf_model.fit(X_train, y_train)

print(f"  CV Accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std() * 2:.4f})")
print(f"  CV F1-Score: {cv_f1_rf.mean():.4f} (+/- {cv_f1_rf.std() * 2:.4f})")

# Select best model
best_model = log_reg if cv_f1_lr.mean() >= cv_f1_rf.mean() else rf_model
best_model_name = "Logistic Regression" if cv_f1_lr.mean() >= cv_f1_rf.mean() else "Random Forest"

print(f"\n✓ Best Model Selected: {best_model_name}")

# Store feature columns for prediction
feature_columns = list(X.columns)

# Create a dictionary with all required objects
model_data = {
    'model': best_model,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'feature_columns': feature_columns,
    'model_name': best_model_name,
    'numeric_columns': numeric_columns
}

# Save to pickle file
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n" + "=" * 60)
print("✓ Model saved to 'loan_model.pkl'")
print("=" * 60)

# Test the saved model
print("\nTesting saved model...")
with open('loan_model.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

print(f"  Model Type: {loaded_data['model_name']}")
print(f"  Features: {loaded_data['feature_columns']}")
print(f"  Label Encoders: {list(loaded_data['label_encoders'].keys())}")

# Test prediction
def test_prediction():
    """Test prediction with sample data"""
    model = loaded_data['model']
    le = loaded_data['label_encoders']
    sc = loaded_data['scaler']
    
    test_input = pd.DataFrame({
        'age': [35],
        'gender': [le['gender'].transform(['Male'])[0]],
        'occupation': [le['occupation'].transform(['Engineer'])[0]],
        'education_level': [le['education_level'].transform(["Bachelor's"])[0]],
        'marital_status': [le['marital_status'].transform(['Married'])[0]],
        'income': [85000],
        'credit_score': [740]
    })
    
    # Create features
    test_input['debt_to_income_ratio'] = (test_input['credit_score'] * 100) / test_input['income']
    test_input['age_income_interaction'] = test_input['age'] * test_input['income'] / 1000
    
    # Normalize
    numeric_cols = ['age', 'income', 'credit_score', 'debt_to_income_ratio', 'age_income_interaction']
    test_input[numeric_cols] = sc.transform(test_input[numeric_cols])
    
    # Ensure column order matches training
    test_input = test_input[loaded_data['feature_columns']]
    
    prediction = model.predict(test_input)[0]
    probability = model.predict_proba(test_input)[0]
    
    status = 'APPROVED' if prediction == 0 else 'DENIED'
    print(f"\n  Test Prediction: {status}")
    print(f"  Approval Probability: {probability[0]*100:.2f}%")
    print(f"  Denial Probability: {probability[1]*100:.2f}%")

test_prediction()

print("\n" + "=" * 60)
print("✓ Model is ready for Django deployment!")
print("=" * 60)
