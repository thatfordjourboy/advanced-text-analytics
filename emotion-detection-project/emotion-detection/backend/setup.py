from setuptools import setup, find_packages

setup(
    name="emotion-detection-backend",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.11,<3.12",
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "httpx==0.25.2",
        "requests==2.31.0",
        "scikit-learn==1.4.0",
        "numpy==1.26.4",
        "pandas==2.1.4",
        "joblib==1.3.2",
        "nltk==3.8.1",
        "python-dotenv==1.0.0",
        "pydantic==2.5.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
