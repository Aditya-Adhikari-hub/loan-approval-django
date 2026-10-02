from django import forms

from .models import LoanPrediction


class LoanPredictionForm(forms.Form):
    age = forms.IntegerField(
        min_value=18, max_value=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter your age'}),
    )
    occupation = forms.ChoiceField(
        choices=LoanPrediction.OCCUPATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
    )
    education_level = forms.ChoiceField(
        choices=LoanPrediction.EDUCATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
    )
    marital_status = forms.ChoiceField(
        choices=LoanPrediction.MARITAL_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
    )
    income = forms.DecimalField(
        min_value=0, max_digits=12, decimal_places=2,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Annual income in USD'}),
    )
    credit_score = forms.IntegerField(
        min_value=300, max_value=850,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Credit score (300-850)'}),
    )