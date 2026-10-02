from django.contrib.auth.models import User
from django.db import models


class LoanPrediction(models.Model):
    """One loan prediction made by a user (input values + model output)."""

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
        (o, o) for o in [
            'Accountant', 'Analyst', 'Architect', 'Banker', 'Chef', 'Consultant',
            'Designer', 'Doctor', 'Engineer', 'IT', 'Lawyer', 'Manager',
            'Marketing', 'Nurse', 'Pharmacist', 'Researcher', 'Sales',
            'Student', 'Teacher', 'Writer',
        ]
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='predictions', null=True, blank=True)

    age = models.IntegerField()
    occupation = models.CharField(max_length=50, choices=OCCUPATION_CHOICES)
    education_level = models.CharField(max_length=20, choices=EDUCATION_CHOICES)
    marital_status = models.CharField(max_length=10, choices=MARITAL_CHOICES)
    income = models.DecimalField(max_digits=12, decimal_places=2)
    credit_score = models.IntegerField()

    prediction_result = models.CharField(max_length=20, default='PENDING')
    approval_probability = models.FloatField(default=0.0)
    denial_probability = models.FloatField(default=0.0)
    confidence = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Loan Prediction'
        verbose_name_plural = 'Loan Predictions'

    def __str__(self):
        return f"Prediction #{self.id} - {self.prediction_result} ({self.confidence:.1f}%)"