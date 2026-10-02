from django.contrib import admin

from .models import LoanPrediction
from .services import predict_loan


@admin.register(LoanPrediction)
class LoanPredictionAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'age', 'occupation', 'income', 'credit_score',
                    'prediction_result', 'confidence', 'created_at']
    list_filter = ['prediction_result', 'education_level', 'marital_status', 'created_at']
    search_fields = ['occupation', 'education_level', 'user__username']
    readonly_fields = ['prediction_result', 'approval_probability', 'denial_probability',
                       'confidence', 'created_at']
    ordering = ['-created_at']

    fieldsets = (
        ('User', {'fields': ('user',)}),
        ('Personal Information', {'fields': ('age', 'marital_status', 'education_level')}),
        ('Professional Information', {'fields': ('occupation', 'income', 'credit_score')}),
        ('Prediction Results', {
            'fields': ('prediction_result', 'approval_probability', 'denial_probability', 'confidence'),
            'classes': ('collapse',),
            'description': 'These fields are auto-calculated when you save.',
        }),
        ('Metadata', {'fields': ('created_at',), 'classes': ('collapse',)}),
    )

    def save_model(self, request, obj, form, change):
        if not change or obj.prediction_result == 'PENDING':
            try:
                result = predict_loan(
                    age=obj.age,
                    occupation=obj.occupation,
                    education_level=obj.education_level,
                    marital_status=obj.marital_status,
                    income=float(obj.income),
                    credit_score=obj.credit_score,
                )
                obj.prediction_result = result['status']
                obj.approval_probability = result['approval_probability']
                obj.denial_probability = result['denial_probability']
                obj.confidence = result['confidence']
            except Exception:
                obj.prediction_result = 'ERROR'
                obj.confidence = 0.0
        super().save_model(request, obj, form, change)