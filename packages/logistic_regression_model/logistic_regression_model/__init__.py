import logging
from logistic_regression_model.config import config, logging_config


file_logger = logging.getLogger(__name__)
file_logger.setLevel(logging.INFO)
file_logger.addHandler(logging_config.get_file_handler())

console_logger = logging.getLogger(__name__)
console_logger.setLevel(logging.INFO)
console_logger.addHandler(logging_config.get_console_handler())

VERSION_PATH = config.PACKAGE_ROOT / "VERSION"
with open(VERSION_PATH, 'r') as version_file:
    __version__ = version_file.read().strip()