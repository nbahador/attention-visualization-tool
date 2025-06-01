from setuptools import setup, find_packages

setup(
    name="attention_viz",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
        "ipywidgets>=7.0.0",
    ],
    python_requires=">=3.7",
)