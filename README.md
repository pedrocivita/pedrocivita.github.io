# Artificial Neural Networks and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MkDocs](https://img.shields.io/badge/MkDocs-Material-526CFE?logo=materialformkdocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?logo=python&logoColor=white)](https://matplotlib.org/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-222222?logo=githubpages&logoColor=white)](https://pedrocivita.github.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive portfolio of neural network projects and exercises developed during the Artificial Neural Networks and Deep Learning course at Insper - Instituto de Ensino e Pesquisa, as part of the Computer Engineering undergraduate program.

**Live Documentation:** [https://pedrocivita.github.io](https://pedrocivita.github.io)

## About

This repository contains a complete collection of theoretical and practical work exploring fundamental concepts and advanced applications in artificial neural networks and deep learning. The content is organized as a professional documentation website built with MkDocs Material, showcasing implementations, analyses, and research across multiple domains of machine learning.

The coursework covers essential topics including data preprocessing, perceptrons, multi-layer perceptrons (MLPs), variational autoencoders (VAEs), and various neural network architectures applied to real-world problems.

## Projects

### 1. Classification
Implementation of classification algorithms using neural networks for pattern recognition and decision-making tasks.

### 2. Regression
Development of regression models leveraging deep learning techniques for continuous value prediction, including a comprehensive Jupyter notebook implementation.

### 3. Generative Models
Exploration of generative neural network architectures capable of creating new data samples similar to training data.

## Exercises

### Exercise 1: Data Preparation and Analysis
- Synthetic dataset generation with Gaussian distributions
- Class separability analysis in 2D space
- PCA dimensionality reduction and visualization
- Real-world data preprocessing with the Spaceship Titanic dataset
- Feature engineering and normalization techniques

### Exercise 2: Perceptron
Implementation and analysis of the perceptron algorithm, the fundamental building block of neural networks.

### Exercise 3: Multi-Layer Perceptron (MLP)
Construction and training of multi-layer neural networks for complex pattern recognition tasks.

### Exercise 4: Variational Autoencoder (VAE)
Development of variational autoencoders for unsupervised learning and generative modeling.

## Technologies

This project utilizes a modern Python-based machine learning stack:

- **Python 3.8+** - Core programming language
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization and plotting
- **MkDocs Material** - Professional documentation generation
- **Jupyter Notebook** - Interactive development environment
- **nbformat** - Jupyter notebook file manipulation
- **yfinance** - Financial data retrieval (for specific projects)

### Documentation Plugins

- **mkdocs-material** - Modern Material Design theme
- **mkdocs-jupyter** - Jupyter notebook integration
- **mkdocs-git-authors-plugin** - Author attribution
- **mkdocs-git-revision-date-localized-plugin** - Localized revision dates
- **mkdocs-glightbox** - Image lightbox functionality
- **mkdocs-badges** - Dynamic badge generation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/pedrocivita/pedrocivita.github.io.git
cd pedrocivita.github.io
```

2. Create a Python virtual environment:

```bash
python3 -m venv env
```

3. Activate the virtual environment:

```bash
# On Linux/macOS
source ./env/bin/activate

# On Windows
.\env\Scripts\activate
```

4. Install dependencies:

```bash
python3 -m pip install -r requirements.txt --upgrade
```

## Usage

### Local Development

To preview the documentation locally with live reload:

```bash
mkdocs serve -o
```

The documentation will be available at `http://127.0.0.1:8000/`

### Deployment

To deploy the documentation to GitHub Pages:

```bash
mkdocs gh-deploy
```

This command builds the site and pushes it to the `gh-pages` branch.

## Repository Structure

```
.
├── docs/                   # Documentation source files
│   ├── exercises/         # Course exercises (1-4)
│   ├── projects/          # Course projects (1-3)
│   └── index.md           # Homepage
├── mkdocs.yml             # MkDocs configuration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Author

**Pedro Toledo Piza Civita**

Computer Engineering Student at Insper - Instituto de Ensino e Pesquisa

- Email: pedrotpc@al.insper.edu.br
- LinkedIn: [Pedro Toledo Piza Civita](https://linkedin.com/in/pedro-toledo-piza-civita)
- GitHub: [@pedrocivita](https://github.com/pedrocivita)

## Academic Context

**Institution:** Insper - Instituto de Ensino e Pesquisa  
**Program:** Computer Engineering (Engenharia de Computação)  
**Course:** Artificial Neural Networks and Deep Learning  
**Semester:** 2025.2

## License

This project is part of academic coursework and is provided for educational purposes. Please refer to your institution's academic integrity policies before reusing any content.

## Acknowledgments

- Insper faculty and staff for their guidance and support
- The open-source community for the excellent tools and libraries
- Course colleagues for collaboration and knowledge sharing

---

For more information, visit the [live documentation](https://pedrocivita.github.io) or contact me through the channels listed above.
