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

    remove_old_pipelines(files_to_keep=save_filename)
    joblib.dump(pipeline, save_path)
    _logger.info(f"Saved Pipeline {save_filename}")


def remove_old_pipelines(*, files_to_keep) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """

    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, "__init__.py"]:
            model_file.unlink()