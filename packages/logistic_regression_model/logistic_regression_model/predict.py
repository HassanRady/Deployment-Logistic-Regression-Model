import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from logistic_regression_model.config import config
from logistic_regression_model.processing.data_management import load_dataset, load_pipeline
from logistic_regression_model import __version__ as _version

_logger = logging.getLogger(__name__)

def predict():
    data = load_dataset(config.DATA_FILE)
    pipe = load_pipeline(config.PIPELINE_SAVE_FILE+_version+".pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1), data[config.TARGET], test_size=0.2, random_state=config.SEED)

    _logger.info(f"Predicting with: {_version}")
    preds = pipe.predict(X_test)
    preds_proba = pipe.predict_proba(X_test)[:, 1]

    return preds, preds_proba, y_test

if __name__ == '__main__':
    preds, preds_proba, y_test = predict()

    print(f"test Accuracy: {accuracy_score(y_test, preds)}")
    print(f"test roc-auc: {roc_auc_score(y_test, preds_proba)}")
