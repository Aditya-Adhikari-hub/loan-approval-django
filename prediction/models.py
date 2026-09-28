from django.db import models
from django.contrib.auth.models import User

class LoanPrediction(models.Model):
    """Model to store loan prediction history"""
    
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
    
    # Link to user
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions', null=True, blank=True)
    
    # Input fields
    age = models.IntegerField()
    gender = models.CharField(max_length=10, choices=GENDER_CHOICES)
    occupation = models.CharField(max_length=50, choices=OCCUPATION_CHOICES)
    education_level = models.CharField(max_length=20, choices=EDUCATION_CHOICES)
    marital_status = models.CharField(max_length=10, choices=MARITAL_CHOICES)
    income = models.DecimalField(max_digits=12, decimal_places=2)
    credit_score = models.IntegerField()
    
    # Prediction results
    prediction_result = models.CharField(max_length=20, default='PENDING')  # APPROVED or DENIED
    approval_probability = models.FloatField(default=0.0)
    denial_probability = models.FloatField(default=0.0)
    confidence = models.FloatField(default=0.0)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Loan Prediction'
        verbose_name_plural = 'Loan Predictions'
    
    def __str__(self):
        return f"Prediction #{self.id} - {self.prediction_result} ({self.confidence:.1f}%)"
