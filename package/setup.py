from setuptools import setup

setup(
    name='package',
    version='0.1',
    description='A useful packages',
    author='Jagadish Devu',
    author_email='jagadishdevu523@icloud.com',
    packages=['package.feature','package.ml_training'],
    install_requires = ['numpy','pandas','scikit-learn','matplotlib','seaborn','mlflow']
)