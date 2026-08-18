# 🐟 Fish Weight Prediction: From Linear Baseline to Polynomial Magic 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

> **Ever wondered if you could guess a fish's weight just by measuring its length, height, and width?**  
> Welcome to the **Fish Weight Prediction Project** — a hands-on machine learning exploration comparing **Linear Regression** with **Polynomial Feature Transformation** to accurately estimate fish weights from physical dimensions! 🎣✨

---

## 📌 Table of Contents
- [🎯 Project Overview](#-project-overview)
- [📊 Dataset Info](#-dataset-info)
- [🧠 Machine Learning Journey](#-machine-learning-journey)
  - [Step 1: Data Exploration & Preprocessing](#step-1-data-exploration--preprocessing)
  - [Step 2: The Baseline: Linear Regression](#step-2-the-baseline--linear-regression)
  - [Step 3: The Power-Up: Polynomial Regression](#step-3-the-power-up--polynomial-regression)
- [📈 Results & Accuracy Boost](#-results--accuracy-boost)
- [🛠️ How to Run](#️-how-to-run)
- [💡 Key Takeaways](#-key-takeaways)

---

## 🎯 Project Overview

In this project, we analyze a dataset of **7 different fish species** to build a predictive regression model. 

While a straight line (Linear Regression) gives a good starting baseline, fish growth and volume in the real world are inherently **3-dimensional** (non-linear). By engineering **Polynomial Features**, we captured complex curve relationships and drastically improved model accuracy!

---

## 📊 Dataset Info

The project utilizes the `Fish.csv` dataset, which contains **159 entries** across 7 common fish species (Bream, Roach, Whitefish, Parkki, Perch, Pike, Smelt):

| Feature | Description | Unit |
| :--- | :--- | :--- |
| **Species** | Categorical name of the fish species | - |
| **Weight** | *Target Variable* — Weight of the fish | grams ($g$) |
| **Length1** | Vertical length | $cm$ |
| **Length2** | Diagonal length | $cm$ |
| **Length3** | Cross length | $cm$ |
| **Height** | Height of the fish | $cm$ |
| **Width** | Diagonal width of the fish | $cm$ |

---

## 🧠 Machine Learning Journey

### Step 1: Data Exploration & Preprocessing 🔍
Before throwing algorithms at the problem, we audited the dataset:
- Checked for missing values: **0 null entries** found! Clean data right off the hook.
- Examined feature distributions, summary statistics (`df.describe()`), and correlations between body measurements.

### Step 2: The Baseline: Linear Regression 📉
We first trained a standard **Linear Regression** model using `scikit-learn` to establish our baseline:
$$\text{Weight} = \beta_0 + \beta_1(\text{Length}) + \beta_2(\text{Height}) + \beta_3(\text{Width}) + \dots$$

**The Limitation:** Linear regression assumes a straight-line relationship. However, in nature, weight scales with **volume** ($V \propto L \times H \times W$), meaning the relationship is cubic or quadratic. As a result, the linear model struggles with larger fish and curved relationships.

### Step 3: The Power-Up: Polynomial Regression ⚡
To solve the non-linear limitation, we engineered new features using `PolynomialFeatures`:
$$\text{Weight} = \beta_0 + \beta_1 x_1 + \beta_2 x_1^2 + \beta_3 (x_1 \cdot x_2) + \dots$$

By expanding our feature space to include higher-degree terms and interaction features:
1. The model learns non-linear curvature in fish dimensions.
2. The interaction terms capture how length, height, and width jointly affect total body volume.
3. Model accuracy ($R^2$ score) shot up significantly, while prediction error (MSE) plummeted!

---

## 📈 Results & Accuracy Boost

| Model | Technique | $R^2$ Accuracy Score | Error (MSE) | Verdict |
| :--- | :--- | :---: | :---: | :--- |
| **Model 1** | Multiple Linear Regression | Baseline (~88-90%) | Higher | Good baseline, but misses 3D volumetric curvature. |
| **Model 2** | **Polynomial Regression** | **Significantly Higher (~98%)** | **Drastically Lower** | 🏆 **Winner!** Captures non-linear fish geometry flawlessly. |

---

## 🛠️ How to Run

### Prerequisites
Make sure you have Python 3 installed along with the following libraries:
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/regression_project.git
   cd regression_project
   ```
2. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook fish_regression.ipynb
   ```
3. Run all cells to reproduce the data analysis, model training, and performance plots!

---

## 💡 Key Takeaways

1. **Domain Knowledge Matters:** Physical objects like fish scale in weight by volume, making non-linear models naturally superior for geometric predictors.
2. **Feature Engineering > Complex Models:** Adding polynomial features allowed a simple linear estimator to outperform basic linear models dramatically without needing overly complex black-box algorithms.

---
*Created with ❤️ for AI/ML Module at NSBM Green University.*