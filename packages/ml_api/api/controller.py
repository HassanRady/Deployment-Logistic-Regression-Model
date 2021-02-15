from flask import Blueprint, request, jsonify
from logistic_regression_model.predict import predict as model_predict
from api.config import get_logger
from logistic_regression_model import __version__ as model_version
from api import __version__ as api_version
from  api.validation import validate_inputs

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'



@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': model_version, 'api_version': api_version})


@prediction_app.route('/v1/predict/logistic', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        # _logger.info(f'Inputs: {json_data}')

        inputs, errors = validate_inputs(json_data)

        result = model_predict(data=inputs)
        _logger.debug(f'Outputs: {result}')

        predictions = result.get('predictions').tolist()
        version = result.get('version')

        return jsonify({'predictions': predictions,
                        'version': version,
                        'errors': errors})
