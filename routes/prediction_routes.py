from flask import Blueprint, request  # Tambahkan request
from controllers.prediction_controller import predict_cv

prediction_bp = Blueprint('prediction_bp', __name__)

@prediction_bp.route('/predict_umum', methods=['POST'])
def predict():
    return predict_cv(request)  # Kirim request sebagai argumen
