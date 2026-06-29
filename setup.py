"""ASTRA — Automated Signal Transit Recognition Algorithm

A complete, physics-informed AI pipeline for automated exoplanet transit
detection and classification from TESS light curve data.

Team ONEROUS — Bharatiya Antariksh Hackathon 2026 (BAH 2026), Challenge 7
"""

from setuptools import setup, find_packages

setup(
    name="astra",
    version="1.0.0",
    description="Automated Signal Transit Recognition Algorithm",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Team ONEROUS — Raj Gupta, Saksham Gupta, Takshak Nikhil Khade, Dev Singi",
    author_email="",
    url="",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "lightkurve>=2.4.0",
        "astropy>=5.3.0",
        "wotan>=1.10",
        "transitleastsquares>=1.0.31",
        "batman-package>=2.4.9",
        "astroquery>=0.4.6",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.1.0",
        "imbalanced-learn>=0.11.0",
        "emcee>=3.1.4",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "reportlab>=4.0.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "astra-run=scripts.run_pipeline:main",
            "astra-train=scripts.run_training:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
    ],
)
