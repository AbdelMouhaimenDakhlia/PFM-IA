# PFM-IA

*Transform Data Into Smarter Financial Decisions*

![last commit](https://img.shields.io/github/last-commit/AbdelMouhaimenDakhlia/PFM-IA)
![python version](https://img.shields.io/badge/python-97.4%25-blue)
![languages](https://img.shields.io/github/languages/count/AbdelMouhaimenDakhlia/PFM-IA)

**Built with the tools and technologies:**

![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
![Markdown](https://img.shields.io/badge/Markdown-000000?style=flat&logo=markdown&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)

## Overview

PFM-IA is a robust financial data analysis platform designed to deliver accurate transaction forecasts, classifications, and personalized recommendations through advanced machine learning models. It seamlessly integrates model training, evaluation, and deployment within a scalable architecture, empowering developers to build intelligent financial applications.

### Why PFM-IA?

This project simplifies complex financial analytics by providing scalable, real-time prediction and classification services. The core features include:

- 🔮 **Prediction API**: Real-time client-specific transaction forecasting using trained CatBoost and XGBoost models.

- 🧠 **Model Training & Serialization**: Tools for developing, evaluating, and saving high-performance predictive models.
- 🧮 **Classification Model**: Accurate categorization of financial transactions using machine learning models trained on labeled transaction data.

- 🐳 **Docker Deployment**: Containerized environment setup for consistent, seamless deployment across systems.

- 📊 **Data Processing & Feature Engineering**: Automated extraction, cleaning, and feature creation from raw transaction data.

- 🎯 **Recommendation System**: Personalized product suggestions based on transactional history.

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language**: Python
- **Package Manager**: Pip
- **Container Runtime**: Docker

### Installation

Build PFM-IA from the source and install dependencies:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AbdelMoulaSeenDakhil/PFM-IA
   ```

2. **Navigate to the project directory**:
   ```bash
   cd PFM-IA
   ```

3. **Install the dependencies**:

   Using **docker**:
   ```bash
   docker build -t AbdelMoulaSeenDakhil/PFM-IA .
   ```

   Using **pip**:
   ```bash
   pip install -r Prediction/requirements.txt, recommendation/requirements.txt
   ```

### Usage

Run the project with:

Using **docker**:
```bash
docker run -it {image_name}
```

Using **pip**:
```bash
python {entrypoint}
```

### Testing

Pfm-ia uses the **(test_framework)** test framework. Run the test suite with:

Using **docker**:
```bash
echo "INSERT-TEST-COMMAND-HERE"
```

Using **pip**:
```bash
pytest
```

---

[🔙 Return](#table-of-contents)
