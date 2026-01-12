from flask import Flask, render_template, request
import joblib
import numpy as np
import logging
from typing import Dict, Any

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# constant values
MODEL_PATHS = {
    'classifier': 'models/bst_diabetes_classifier_clusters.pkl',
    'scaler': 'models/scaler.pkl',
    'kmeans': 'models/kmeans.pkl'
}

# models cache (Singleton simple pattern)
_models_cache = {}

def load_models():
    """Loading ML models with cache."""
    if not _models_cache:
        try:
            _models_cache['classifier'] = joblib.load(MODEL_PATHS['classifier'])
            _models_cache['scaler'] = joblib.load(MODEL_PATHS['scaler'])
            _models_cache['kmeans'] = joblib.load(MODEL_PATHS['kmeans'])
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Loading models error: {e}")
            raise
    return _models_cache


class PatientDataValidator:
    """Patient data validation."""

    @staticmethod
    def validate_form_data(form_data: Dict[str, str]) -> Dict[str, Any]:
        """Form data validation."""
        errors = []

        # Required features
        required_features = ['age', 'gender', 'bmi', 'chol', 'tg', 'hdl', 'ldl', 'cr', 'bun']
        for feature in required_features:
            if feature not in form_data or not form_data[feature].strip():
                errors.append(f"Feature '{feature}' is required")

        if errors:
            raise ValueError("; ".join(errors))

        # Type conversion with error handling
        try:
            return {
                'age': int(form_data['age']),
                'gender': form_data['gender'].upper(),
                'bmi': float(form_data['bmi']),
                'chol': float(form_data['chol']),
                'tg': float(form_data['tg']),
                'hdl': float(form_data['hdl']),
                'ldl': float(form_data['ldl']),
                'cr': float(form_data['cr']),
                'bun': float(form_data['bun'])
            }
        except ValueError as e:
            raise ValueError(f"Incorrect data format: {str(e)}")


class DiabetesPredictor:
    """Main class for diabetes prediction."""

    def __init__(self):
        self.models = load_models()

    def prepare_features(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """Preparing features before usage."""
        gender_encoded = 1 if patient_data['gender'] == 'M' else 0

        return np.array([[
            patient_data['age'],
            gender_encoded,
            patient_data['bmi'],
            patient_data['chol'],
            patient_data['tg'],
            patient_data['hdl'],
            patient_data['ldl'],
            patient_data['cr'],
            patient_data['bun'],
        ]])

    def predict(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform prediction."""
        # Data preparation
        X = self.prepare_features(patient_data)

        # Scaling
        X_scaled = self.models['scaler'].transform(X)

        # Clustering
        cluster = int(self.models['kmeans'].predict(X_scaled)[0])

        # Add cluster to numpy array
        X_with_cluster = np.append(X, [[cluster]], axis=1)

        # Prediction
        prediction = int(self.models['classifier'].predict(X_with_cluster)[0])
        probability = float(self.models['classifier'].predict_proba(X_with_cluster)[0][1])

        return {
            'prediction': prediction,
            'probability': probability,
        }

# Flask Initialization
app = Flask(__name__)
predictor = DiabetesPredictor()
validator = PatientDataValidator()

@app.route("/")
def main():
    logger.info('Finished')
    return render_template('main.html')

@app.route("/predict", methods=["POST"])
def predict():
    """Prediction endpoint."""
    try:
        # Data validation
        patient_data = validator.validate_form_data(request.form)

        # Prediction
        result = predictor.predict(patient_data)

        # Results
        return render_template(
            "results.html",
            prediction=result['prediction'],
            probability=round(result['probability'] * 100, 2)
        )

    except ValueError as e:
        # Data validation error
        logger.warning(f"Validation error: {e}")
        return render_template(
            "error.html",
            error_message=f"Entered incorrect data: {str(e)}"
        ), 400

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return render_template(
            "error.html",
            error_message="An error occurred while processing. Please try again."
        ), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)