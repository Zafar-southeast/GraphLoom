#!/usr/bin/env python3
"""
GraphLoom Setup Script
======================

Install GraphLoom as a package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="graphloom",
    version="0.1.0",
    author="GraphLoom Team",
    description="Multimodal KG-RAG Framework for Grounded Question Answering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/imrankh46/graphloom",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "transformers>=4.57.0",
        "torch>=2.0.0",
        "openai>=1.0.0",
        "groq>=0.4.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
        ],
        "full": [
            "spacy>=3.7.0",
            "sentence-transformers>=2.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "graphloom=graphloom.run:main",
        ],
    },
)
