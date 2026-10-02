from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from .forms import LoanPredictionForm
from .models import LoanPrediction
from .services import ModelNotAvailable, explain_prediction, predict_loan


def home(request):
    return render(request, 'prediction/landing.html')


@login_required
def apply(request):
    if request.method == 'POST':
        form = LoanPredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            try:
                result = predict_loan(**data)
            except ModelNotAvailable as exc:
                messages.error(request, str(exc))
            except Exception:
                messages.error(request, "Something went wrong while making the prediction.")
            else:
                prediction = LoanPrediction.objects.create(
                    user=request.user,
                    prediction_result=result['status'],
                    approval_probability=result['approval_probability'],
                    denial_probability=result['denial_probability'],
                    confidence=result['confidence'],
                    **data,
                )
                return redirect('prediction_result', pk=prediction.pk)
    else:
        form = LoanPredictionForm()

    return render(request, 'prediction/home.html', {'form': form})


@login_required
def prediction_result(request, pk):
    try:
        prediction = LoanPrediction.objects.get(pk=pk, user=request.user)
    except LoanPrediction.DoesNotExist:
        messages.error(request, "Prediction not found")
        return redirect('home')

    try:
        explanation = explain_prediction(
            age=prediction.age,
            occupation=prediction.occupation,
            education_level=prediction.education_level,
            marital_status=prediction.marital_status,
            income=prediction.income,
            credit_score=prediction.credit_score,
        )
    except Exception:
        explanation = []  # the page still works without the explanation

    return render(request, 'prediction/result.html', {
        'prediction': prediction,
        'explanation': explanation,
    })


@login_required
def prediction_history(request):
    all_predictions = LoanPrediction.objects.filter(user=request.user).order_by('-created_at')
    approved_count = all_predictions.filter(prediction_result='APPROVED').count()
    denied_count = all_predictions.filter(prediction_result='DENIED').count()

    # Data for the charts: latest 20 predictions, oldest first
    recent = list(all_predictions[:20])[::-1]
    chart_data = {
        'approved': approved_count,
        'denied': denied_count,
        'labels': [p.created_at.strftime('%b %d %H:%M') for p in recent],
        'approval_probability': [round(p.approval_probability, 1) for p in recent],
    }

    return render(request, 'prediction/history.html', {
        'predictions': all_predictions[:50],
        'total_count': all_predictions.count(),
        'approved_count': approved_count,
        'denied_count': denied_count,
        'chart_data': chart_data,
    })