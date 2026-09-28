from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .forms import LoanPredictionForm
from .models import LoanPrediction
import pickle
import pandas as pd
import os
from django.conf import settings

# Load the model once when the module is imported
MODEL_PATH = os.path.join(settings.BASE_DIR, 'loan_model.pkl')

def load_model():
    """Load the trained model and related objects"""
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Load model data at startup
try:
    MODEL_DATA = load_model()
except Exception as e:
    MODEL_DATA = None
    print(f"Warning: Could not load model: {e}")


def predict_loan(age, gender, occupation, education_level, marital_status, income, credit_score):
    """
    Make a loan approval prediction using the loaded model
    """
    if MODEL_DATA is None:
        raise Exception("Model not loaded")
    
    model = MODEL_DATA['model']
    label_encoders = MODEL_DATA['label_encoders']
    scaler = MODEL_DATA['scaler']
    feature_columns = MODEL_DATA['feature_columns']
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [label_encoders['gender'].transform([gender])[0]],
        'occupation': [label_encoders['occupation'].transform([occupation])[0]],
        'education_level': [label_encoders['education_level'].transform([education_level])[0]],
        'marital_status': [label_encoders['marital_status'].transform([marital_status])[0]],
        'income': [float(income)],
        'credit_score': [credit_score]
    })
    
    # Create additional features
    input_data['debt_to_income_ratio'] = (input_data['credit_score'] * 100) / input_data['income']
    input_data['age_income_interaction'] = input_data['age'] * input_data['income'] / 1000
    
    # Normalize numeric features
    numeric_cols = ['age', 'income', 'credit_score', 'debt_to_income_ratio', 'age_income_interaction']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    
    # Ensure column order matches training
    input_data = input_data[feature_columns]
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    
    # 0 = Approved, 1 = Denied
    status = 'APPROVED' if prediction == 0 else 'DENIED'
    confidence = max(probability) * 100
    
    return {
        'status': status,
        'confidence': confidence,
        'approval_probability': probability[0] * 100,
        'denial_probability': probability[1] * 100
    }


def home(request):
    """Landing page with information about the website"""
    return render(request, 'prediction/landing.html')


@login_required
def apply(request):
    """Loan application form page"""
    if request.method == 'POST':
        form = LoanPredictionForm(request.POST)
        if form.is_valid():
            try:
                # Get form data
                age = form.cleaned_data['age']
                gender = form.cleaned_data['gender']
                occupation = form.cleaned_data['occupation']
                education_level = form.cleaned_data['education_level']
                marital_status = form.cleaned_data['marital_status']
                income = form.cleaned_data['income']
                credit_score = form.cleaned_data['credit_score']
                
                # Make prediction
                result = predict_loan(
                    age=age,
                    gender=gender,
                    occupation=occupation,
                    education_level=education_level,
                    marital_status=marital_status,
                    income=income,
                    credit_score=credit_score
                )
                
                # Save prediction to database
                prediction_obj = LoanPrediction.objects.create(
                    user=request.user,
                    age=age,
                    gender=gender,
                    occupation=occupation,
                    education_level=education_level,
                    marital_status=marital_status,
                    income=income,
                    credit_score=credit_score,
                    prediction_result=result['status'],
                    approval_probability=result['approval_probability'],
                    denial_probability=result['denial_probability'],
                    confidence=result['confidence']
                )
                
                # Redirect to result page
                return redirect('prediction_result', pk=prediction_obj.pk)
                
            except Exception as e:
                messages.error(request, f"Error making prediction: {str(e)}")
    else:
        form = LoanPredictionForm()
    
    return render(request, 'prediction/home.html', {'form': form})


@login_required
def prediction_result(request, pk):
    """Display prediction result"""
    try:
        prediction = LoanPrediction.objects.get(pk=pk, user=request.user)
    except LoanPrediction.DoesNotExist:
        messages.error(request, "Prediction not found")
        return redirect('home')
    
    return render(request, 'prediction/result.html', {'prediction': prediction})


@login_required
def prediction_history(request):
    """Display prediction history"""
    all_predictions = LoanPrediction.objects.filter(user=request.user).order_by('-created_at')
    approved_count = all_predictions.filter(prediction_result='APPROVED').count()
    denied_count = all_predictions.filter(prediction_result='DENIED').count()
    predictions = all_predictions[:50]
    return render(request, 'prediction/history.html', {
        'predictions': predictions,
        'approved_count': approved_count,
        'denied_count': denied_count
    })
