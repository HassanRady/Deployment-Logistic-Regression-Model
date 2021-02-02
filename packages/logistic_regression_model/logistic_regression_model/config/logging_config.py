import logging
import sys

FORMATTER = logging.Formatter("%(asctime)s:%(name)s:%(levelname)s: %(funcName)s:%(lineno)d: %(message)s")

def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler

def get_file_handler():
    file_handler = logging.FileHandler("logs.log")
    file_handler.setFormatter(FORMATTER)
    return file_handler
