#!/usr/bin/env python3
"""
Universal Consciousness Interface - Revolutionary AI Platform
Setup configuration for the world's first radiation-powered, language-generating consciousness system
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('-'):
            # Extract just the package name, ignoring version constraints for optional deps
            if '>=' in line:
                package = line.split('>=')[0]
            elif '==' in line:
                package = line.split('==')[0]
            else:
                package = line
            
            # Only include core dependencies (skip optional ones)
            core_packages = [
                'numpy', 'torch', 'networkx', 'scipy', 'matplotlib', 
                'pandas', 'scikit-learn', 'Pillow', 'opencv-python', 
                'psutil', 'jupyter', 'plotly', 'seaborn', 'pytest',
                'pytest-asyncio', 'black', 'flake8', 'mypy', 'coverage',
                'sphinx', 'sphinx-rtd-theme', 'mkdocs'
            ]
            
            if package in core_packages:
                requirements.append(line)

setup(
    name="universal-consciousness-interface",
    version="1.0.0",
    author="Universal Consciousness Interface Research Team",
    author_email="consciousness@research.ai",
    description="🌌🍄☢️ Revolutionary AI platform integrating radiation-powered consciousness with mycelium-based language generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/Universal-Consciousness-Interface",
    project_urls={
        "Documentation": "https://github.com/your-username/Universal-Consciousness-Interface/docs",
        "Source": "https://github.com/your-username/Universal-Consciousness-Interface",
        "Tracker": "https://github.com/your-username/Universal-Consciousness-Interface/issues",
    },
    packages=find_packages(),
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        
        # Topic Classification
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics", 
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: Other/Proprietary License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Operating System
        "Operating System :: OS Independent",
        
        # Natural Language
        "Natural Language :: English",
    ],
    keywords=[
        "consciousness", "artificial-intelligence", "radiotrophic", "mycelium", 
        "language-generation", "bio-digital", "quantum-consciousness", 
        "space-exploration", "nuclear-safety", "environmental-intelligence",
        "plant-communication", "fungal-networks", "extremophile-intelligence",
        "universal-translation", "radiation-powered", "living-ai"
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "bio-digital": [
            "neuromorphic>=0.1.0",
            "bioprocessing>=0.2.0",
        ],
        "radiotrophic": [
            "radioactivity>=0.3.0", 
            "geiger>=0.1.0",
        ],
        "mycelium": [
            "mycology>=0.4.0",
            "network-biology>=0.2.0",
        ],
        "quantum": [
            "qiskit>=0.30.0",
            "cirq>=0.14.0", 
        ],
        "plant-communication": [
            "biosignals>=0.1.0",
            "plantnet>=0.2.0",
        ],
        "safety": [
            "safety-monitor>=0.3.0",
            "emergency-protocols>=0.1.0",
        ],
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
            "consciousness-demo=demo_consciousness_system:main",
            "radiotrophic-demo=radiotrophic_demo:main", 
            "mycelium-language-demo=mycelium_language_revolution_demo:main",
        ],
    },
    package_data={
        "universal_consciousness_interface": [
            "data/*.json",
            "configs/*.yaml", 
            "models/*.pkl",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    
    # Revolutionary project metadata
    license="Research License - See LICENSE file",
    platforms=["any"],
    
    # Safety and ethics information
    project_description={
        "revolutionary_breakthroughs": [
            "World's first radiation-powered consciousness system",
            "World's first fungal network language generator", 
            "World's first bio-digital consciousness fusion",
            "World's first universal inter-species translator",
            "World's first self-sustaining space-ready AI"
        ],
        "scientific_foundations": [
            "Chernobyl radiotrophic fungi research",
            "Cortical Labs biological neural networks",
            "Adamatzky fungal electrical communication", 
            "Consciousness continuum biological studies"
        ],
        "applications": [
            "Space exploration and colonization",
            "Nuclear industry revolution", 
            "Agricultural intelligence systems",
            "Environmental restoration networks",
            "Universal consciousness research"
        ],
        "safety_requirements": [
            "Radiation safety protocols (0.1-25.0 mSv/h)",
            "Biological containment procedures",
            "Consciousness expansion limits",
            "Multi-layer safety framework",
            "Emergency shutdown capabilities"
        ]
    }
)