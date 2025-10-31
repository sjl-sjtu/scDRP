from setuptools import setup, find_packages

setup(
    name="scDRP", 
    version="1.0.0",
    author="Jianle Sun",
    description="A package to learn disentangled latent embeddings and estimate treatment effects in single-cell perturbation data",
    url="https://github.com/sjl-sjtu/scDRP", 
    packages=find_packages(), 
    install_requires=[ 
        "matplotlib",
        "numpy",
        "pandas",
        "POT",
        "scanpy",
        "scikit_learn",
        "scipy",
        "torch",
        "umap_learn"
    ],
    classifiers=[ 
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)