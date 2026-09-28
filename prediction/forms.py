from django import forms
from .models import LoanPrediction

class LoanPredictionForm(forms.Form):
    """Form for loan prediction input"""
    
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]
    
    EDUCATION_CHOICES = [
        ('High School', 'High School'),
        ("Associate's", "Associate's"),
        ("Bachelor's", "Bachelor's"),
        ("Master's", "Master's"),
        ('Doctoral', 'Doctoral'),
    ]
    
    MARITAL_CHOICES = [
        ('Single', 'Single'),
        ('Married', 'Married'),
    ]
    
    OCCUPATION_CHOICES = [
        ('Accountant', 'Accountant'),
        ('Analyst', 'Analyst'),
        ('Architect', 'Architect'),
        ('Banker', 'Banker'),
        ('Chef', 'Chef'),
        ('Consultant', 'Consultant'),
        ('Designer', 'Designer'),
        ('Doctor', 'Doctor'),
        ('Engineer', 'Engineer'),
        ('IT', 'IT'),
        ('Lawyer', 'Lawyer'),
        ('Manager', 'Manager'),
        ('Marketing', 'Marketing'),
        ('Nurse', 'Nurse'),
        ('Pharmacist', 'Pharmacist'),
        ('Researcher', 'Researcher'),
        ('Sales', 'Sales'),
        ('Student', 'Student'),
        ('Teacher', 'Teacher'),
        ('Writer', 'Writer'),
    ]
    
    age = forms.IntegerField(
        min_value=18,
        max_value=100,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your age'
        })
    )
    
    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    occupation = forms.ChoiceField(
        choices=OCCUPATION_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    education_level = forms.ChoiceField(
        choices=EDUCATION_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    marital_status = forms.ChoiceField(
        choices=MARITAL_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-control'
        })
    )
    
    income = forms.DecimalField(
        min_value=0,
        max_digits=12,
        decimal_places=2,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Annual income in USD'
        })
    )
    
    credit_score = forms.IntegerField(
        min_value=300,
        max_value=850,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Credit score (300-850)'
        })
    )
