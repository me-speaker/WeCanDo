from setuptools import setup, find_packages

setup(
    name="deepind",
    version="1.5.0",
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    package_data={
        "src": ["*.yaml", "*.yml"],
    },
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "PyYAML>=6.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
    ],
    entry_points={
        "console_scripts": [
            "daeelm=src.cli:main",
        ],
    },
    python_requires=">=3.8",
    description="DAE-ELM - Chemical Formula Optimization System",
    author="DeepInd Team",
)
