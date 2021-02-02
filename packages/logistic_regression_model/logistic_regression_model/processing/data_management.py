import pandas as pd
import joblib
from logistic_regression_model.config import config
from logistic_regression_model import __version__ as _version
import logging
from sklearn.pipeline import Pipeline


_logger = logging.getLogger(__name__)


def load_dataset(filename: str) -> pd.DataFrame:
    _dataset = pd.read_csv(f"{config.DATASET_DIR}/{filename}")
    return _dataset


def load_pipeline(filename: str) -> Pipeline:
    file_path = config.TRAINED_MODEL_DIR / filename
    pipeline = joblib.load(file_path)
    return pipeline


def save_pipeline(pipeline: Pipeline) -> None:
    save_filename = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = config.TRAINED_MODEL_DIR / save_filename

    joblib.dump(pipeline, save_path)
    _logger.info(f"Saved Pipeline {save_filename}")
