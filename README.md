# SahAI Machine Learning: Skill Correlation Pipeline

This directory holds the Machine Learning training codebase (`sahai-ml-training`). It computes global and individual Pearson correlation networks based on student mastery matrices and outputs model weights to load into the Python worker cluster.

---

## 🧮 Mathematical Model

### 1. Student-Concept Mastery Matrix
We pull cognitive states from PostgreSQL and construct a matrix $M \in \mathbb{R}^{U \times C}$ where $U$ is the number of students, $C$ is the number of concepts, and:
$$M_{u, c} = E[K_{u, c}]$$

### 2. Pearson Correlation Network
We calculate the Pearson correlation coefficient $r$ between every pair of concept columns $X$ and $Y$ in the matrix:
$$r_{X,Y} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$
* **`concept_correlations` (Global):** Stores $r_{X,Y}$ as model weights, representing how mastery of one concept indicates mastery of another in the wider student population.
* **`user_concept_correlations` (Individual):** Calculates personalized weights by scaling the correlation coefficient with the student's variance. This feeds the personalized "Skill Mesh" rendering.

---

## 🚀 Execution Guide

1. Ensure the databases are running (`docker compose up -d`).
2. Run the training script locally using the Python 3.11 virtual environment:
```bash
source venv/bin/activate
pip install -r services/ml-training/requirements.txt
PYTHONPATH=services/ml-training python services/ml-training/src/train.py
```
This writes the learned networks back to PostgreSQL and saves the model coefficients to `services/ml-training/models/correlation_weights.json`.
