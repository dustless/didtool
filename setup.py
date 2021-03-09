import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="didtool",
    version="0.1.0",
    author="dustless",
    author_email="wuchenghui927@126.com",
    description="Tool set for feature engineering & modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dustless/didtool",
    packages=setuptools.find_packages(),
    install_requires=[
        "seaborn>=0.10.1",
        "scikit-learn>=0.24.1",
        "lightgbm>=3.1.0",
        "sklearn2pmml>=0.65.0",
        "bayesian-optimization==1.2.0",
        "pandas",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
