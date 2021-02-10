import os
import io
from pathlib import Path
from setuptools import setup, find_packages

# Package meta-data.
NAME = 'logistic_regression_model'
DESCRIPTION = 'Train and deploy logistic regression model.'
URL = 'https://github.com/HassanRady/Deployment-Logistic-Regression-Model.git'
EMAIL = 'hassan.khaled.rady@gmail.com'
AUTHOR = 'HassanRady'
REQUIRES_PYTHON = '>=3.6.0'


def read_reqs(fname="requirements.txt"):
    with open(fname) as file:
        return file.read().splitlines()


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="ytf-8") as file:
        long_description = "\n" + file.read()
except FileNotFoundError:
    long_description = DESCRIPTION

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests", )),
    packages_data={"logistic_regression_model": ["VERSION"]},
    install_requires=read_reqs(),
    extras_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
