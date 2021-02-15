import numpy as np
import pathlib 
import logistic_regression_model

PACKAGE_ROOT = pathlib.Path(logistic_regression_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
DATASET_DIR = PACKAGE_ROOT / "datasets"
PIPELINE_NAME = "logistic_regression"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

DATA_FILE = "data.csv"

SEED = 0

TARGET = "survived"
REPLACE_VALUE_WITH = ["?", np.nan]
CABIN_VAR = "cabin"
NAME_VAR = "name"

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked']
NUMERICALS_VARS = ['age', 'fare']

NUMERICALS_TO_FLOAT = ["age", "fare"]
COLUMNS_TO_DROP = ['name', 'ticket', 'boat', 'body', 'home.dest']
NUMERICALS_TO_IMPUTE = ['age', 'fare']
CATEGORICAL_TO_IMPUTE = ['sex', 'cabin', 'embarked', 'title']
CATEGORICAL_TO_ENCODE = CATEGORICAL_TO_IMPUTE

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

NUMERICAL_NA_NOT_ALLOWED = NUMERICALS_VARS
CATEGORICAL_NA_NOT_ALLOWED = CATEGORICAL_VARS