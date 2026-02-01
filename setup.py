from setuptools import setup, find_packages

setup(
    name="lstm_ensemble_trading",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'lightgbm>=4.0.0',
        'xgboost>=2.0.0',
        'sqlalchemy>=2.0.0',
        'psycopg2-binary>=2.9.0',
        'polygon-api-client>=1.12.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'plotly>=5.14.0',
        'pyyaml>=6.0',
        'python-dotenv>=1.0.0',
    ],
    python_requires='>=3.8',
    author="Trading System",
    description="LSTM Ensemble Trading System for minute-level predictions",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
)
