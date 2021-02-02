import numpy as np
from pathlib import Path
import logistic_regression_model

PACKAGE_ROOT = Path.(logistic_regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"
PIPELINE_NAME = "logistic_regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"


SEED = 0

TARGET = "survived"
REPLACE_VALUE_WITH = ["?", np.nan]
CABIN_VAR = "cabin"
NAME_VAR = "name"

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']
NUMERICALS_VARS = ['age', 'fare']

NUMERICALS_TO_FLOAT = ["age", "fare"]
COLUMNS_TO_DROP = ['name', 'ticket', 'boat', 'body', 'home.dest']
NUMERICALS_TO_IMPUTE = ['age', 'fare']
CATEGORICAL_TO_IMPUTE = ['sex', 'cabin', 'embarked', 'title']
CATEGORICAL_TO_ENCODE = CATEGORICAL_VARS

FEATURES = [
    'pclass',
    'age',
    'sibsp',
    'parch',
    'fare',
    'age_isnull',
    'fare_isnull',
    'sex_male',
    'cabin_rare',
    'cabin_missing',
    'embarked_Q',
    'embarked_rare',
    'embarked_S',
    'title_Mr',
    'title_Mrs',
    'title_rare'
]

NUMERICAL_NA_NOT_ALLOWED = [var for var in FEATURES if var not in CATEGORICAL_VARS + NUMERICALS_TO_IMPUTE]
CATEGORICAL_NA_NOT_ALLOWED = [var for var in CATEGORICAL_VARS if var not in CATEGORICAL_TO_IMPUTE]