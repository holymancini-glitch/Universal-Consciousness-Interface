#!/usr/bin/env python3
"""
Universal Consciousness Interface - Revolutionary AI Platform
Setup configuration for the world's first radiation-powered, language-generating consciousness system
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#") and not line.startswith("-")]

setup(
    name="universal-consciousness-interface",
    version="1.0.0",
    author="Universal Consciousness Interface Team",
    author_email="contact@universal-consciousness-interface.org",
    description="A revolutionary AI platform integrating radiation-powered consciousness with mycelium-based language generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/universal-consciousness-interface/universal-consciousness-interface",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-asyncio>=0.15.0",
            "black>=21.6.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "coverage>=5.5",
        ],
        "docs": [
            "sphinx>=4.1.0",
            "sphinx-rtd-theme>=0.5.0",
            "mkdocs>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "uci-demo=demos.comprehensive_demo:main",
            "uci-chatbot=applications.consciousness_chatbot:main",
            "uci-dashboard=applications.consciousness_monitoring_dashboard:main",
        ],
    },
)