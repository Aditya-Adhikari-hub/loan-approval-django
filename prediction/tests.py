"""Tests for the prediction app.

Covers the areas listed in the project's Testing and Evaluation Plan:
form validation, authentication, prediction integration, and persistence
(a user can only see their own predictions).

Run with:  python manage.py test prediction
"""
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from .forms import LoanPredictionForm
from .models import LoanPrediction

GOOD_APPLICANT = {
    'age': 38,
    'occupation': 'Engineer',
    'education_level': "Master's",
    'marital_status': 'Married',
    'income': '120000',
    'credit_score': 790,
}

WEAK_APPLICANT = {
    'age': 30,
    'occupation': 'Chef',
    'education_level': 'High School',
    'marital_status': 'Single',
    'income': '25000',
    'credit_score': 540,
}


class FormValidationTests(TestCase):
    """The form should accept realistic values and reject invalid ones."""

    def test_valid_data_is_accepted(self):
        form = LoanPredictionForm(data=GOOD_APPLICANT)
        self.assertTrue(form.is_valid())

    def test_credit_score_below_300_is_rejected(self):
        data = {**GOOD_APPLICANT, 'credit_score': 250}
        form = LoanPredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('credit_score', form.errors)

    def test_credit_score_above_850_is_rejected(self):
        data = {**GOOD_APPLICANT, 'credit_score': 900}
        form = LoanPredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('credit_score', form.errors)

    def test_negative_income_is_rejected(self):
        data = {**GOOD_APPLICANT, 'income': '-500'}
        form = LoanPredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('income', form.errors)

    def test_age_below_18_is_rejected(self):
        data = {**GOOD_APPLICANT, 'age': 15}
        form = LoanPredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('age', form.errors)

    def test_missing_required_field_is_rejected(self):
        data = {**GOOD_APPLICANT}
        del data['occupation']
        form = LoanPredictionForm(data=data)
        self.assertFalse(form.is_valid())
        self.assertIn('occupation', form.errors)


class AuthenticationTests(TestCase):
    """Prediction pages should only be reachable while logged in."""

    def test_anonymous_user_is_redirected_from_apply(self):
        response = self.client.get(reverse('apply'))
        self.assertEqual(response.status_code, 302)
        self.assertIn(reverse('login'), response.url)

    def test_anonymous_user_is_redirected_from_history(self):
        response = self.client.get(reverse('prediction_history'))
        self.assertEqual(response.status_code, 302)

    def test_anonymous_user_is_redirected_from_result(self):
        response = self.client.get(reverse('prediction_result', args=[1]))
        self.assertEqual(response.status_code, 302)

    def test_logged_in_user_can_reach_apply_page(self):
        User.objects.create_user(username='alice', password='pass12345!')
        self.client.login(username='alice', password='pass12345!')
        response = self.client.get(reverse('apply'))
        self.assertEqual(response.status_code, 200)


class PredictionIntegrationTests(TestCase):
    """Submitting the form should run the model and store a result."""

    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pass12345!')
        self.client.login(username='alice', password='pass12345!')

    def test_strong_applicant_is_saved_and_approved(self):
        response = self.client.post(reverse('apply'), GOOD_APPLICANT)
        self.assertEqual(response.status_code, 302)  # redirected to the result page

        prediction = LoanPrediction.objects.latest('id')
        self.assertEqual(prediction.user, self.user)
        self.assertEqual(prediction.prediction_result, 'APPROVED')
        self.assertGreater(prediction.approval_probability, 50)

    def test_weak_applicant_is_saved_and_denied(self):
        self.client.post(reverse('apply'), WEAK_APPLICANT)
        prediction = LoanPrediction.objects.latest('id')
        self.assertEqual(prediction.prediction_result, 'DENIED')
        self.assertGreater(prediction.denial_probability, 50)

    def test_result_page_shows_the_prediction(self):
        self.client.post(reverse('apply'), GOOD_APPLICANT)
        prediction = LoanPrediction.objects.latest('id')
        response = self.client.get(reverse('prediction_result', args=[prediction.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'APPROVED')

    def test_invalid_submission_does_not_create_a_record(self):
        data = {**GOOD_APPLICANT, 'credit_score': 999}
        self.client.post(reverse('apply'), data)
        self.assertEqual(LoanPrediction.objects.count(), 0)

    def test_history_page_lists_the_users_predictions(self):
        self.client.post(reverse('apply'), GOOD_APPLICANT)
        self.client.post(reverse('apply'), WEAK_APPLICANT)
        response = self.client.get(reverse('prediction_history'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['total_count'], 2)
        self.assertEqual(response.context['approved_count'], 1)
        self.assertEqual(response.context['denied_count'], 1)


class PersistenceAndAccessTests(TestCase):
    """A user must never be able to see another user's prediction."""

    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pass12345!')
        self.bob = User.objects.create_user(username='bob', password='pass12345!')

    def test_user_cannot_view_another_users_result(self):
        self.client.login(username='alice', password='pass12345!')
        self.client.post(reverse('apply'), GOOD_APPLICANT)
        prediction = LoanPrediction.objects.latest('id')
        self.client.logout()

        self.client.login(username='bob', password='pass12345!')
        response = self.client.get(reverse('prediction_result', args=[prediction.pk]))
        self.assertEqual(response.status_code, 302)  # redirected away, not shown

    def test_history_only_shows_own_predictions(self):
        self.client.login(username='alice', password='pass12345!')
        self.client.post(reverse('apply'), GOOD_APPLICANT)
        self.client.logout()

        self.client.login(username='bob', password='pass12345!')
        self.client.post(reverse('apply'), WEAK_APPLICANT)
        response = self.client.get(reverse('prediction_history'))
        self.assertEqual(response.context['total_count'], 1)