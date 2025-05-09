# Trait Prediction from Dutch Company Names

This repository contains the full codebase and outputs for the thesis project:  
**"Trait Prediction from Dutch Company Names using Surface and Semantic Features."**  
We investigate whether letter-level patterns and semantic embeddings can predict human perceptions across four traits:

> **Femininity**, **Evilness**, **Trustworthiness**, **Smartness**

---

## 📌 Overview

We trained and evaluated models using two algorithms:
- **ElasticNet** (for interpretable linear baselines)
- **Feedforward Neural Networks (FFNNs)** (for non-linear modeling)

We tested four feature sets:
- 🟦 **Unigrams** (letter counts)  
- 🟩 **Bigrams** (letter pairs)  
- 🟨 **RobBERT** (Dutch semantic embeddings)  
- 🟥 **Combined** (all features together)

---

## 📂 Folder Structure

| File/Folder           | Description                                                  |
|-----------------------|--------------------------------------------------------------|
| `main.ipynb`          | Full pipeline for all experiments and models                 |
| `models/`             | Trained models, metrics, feature importances, and plots      |

> ⚠️ Some preprocessing steps are not in the notebook but all final modeling data is included.

---

## 📊 What You Can Explore

- ✔️ Compare **model performance** across all four traits
- 🔍 See **ElasticNet feature importances** per trait (letters, bigrams, or embeddings)
- 🧪 View **charts not shown in the thesis**
- 🔁 Adapt the code to predict traits in other domains (e.g., product names)

---


## 🧪 Trained Models

We trained:
- **ElasticNet regressors** and **Feedforward Neural Networks (FFNNs)**
- Across 4 feature sets: Unigram, Bigram, RobBERT, and Combined

Each model’s results (train/val/test R², feature count, etc.) are stored in:
models/{model_name}/ffnn_metrics.csv
models/{model_name}/all_models_metrics.csv

yaml
Copy
Edit

All test plots and coefficient visualizations can be found in the `models/` folder — including additional ones not shown in the thesis.

---

## ⚙️ Setup

Tested on **Windows 10**, with **Python 3.13.2**.

Install dependencies using:

```bas
pip install -r requirements.txt


