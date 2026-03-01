# 📊 Data Science Projects

> A collection of end-to-end data science projects — from exploratory analysis to model training and evaluation.

![Python](https://img.shields.io/badge/Python-≥3.14-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📁 Projects

| # | Project | Description | Key Tech |
|---|---------|-------------|----------|
| 1 | [🏠 House Price Prediction](house-price-prediction/) | Regression on the Ames Housing dataset (2,930 samples, 82 features). Compares Linear Regression vs Random Forest with full preprocessing pipeline. | scikit-learn, pandas, matplotlib |

*More projects coming soon…*

---

## 🛠️ Setup

All projects share a single dependency root managed with [**uv**](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/DanciVasile/data-science-projects.git
cd data-science-projects
uv sync
```

Then navigate into any project folder and follow its README.

---

## 📂 Repository Structure

```
data-science-projects/
├── .gitignore
├── pyproject.toml          # Shared dependencies (managed by uv)
├── uv.lock
├── README.md               # ← You are here
│
└── house-price-prediction/ # Project 1
    ├── README.md
    ├── data/
    ├── docs/
    ├── models/             # Auto-generated (.gitignored)
    ├── notebooks/
    ├── reports/figures/    # Auto-generated (.gitignored)
    └── src/
```

Each project lives in its own self-contained folder with its own README, data, notebooks, and source code.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/DanciVasile">Vasile Danci</a>
  <br/>
  <a href="https://github.com/DanciVasile">GitHub</a> · 
  <a href="https://www.linkedin.com/in/vasile-danci-m/">LinkedIn</a>
</p>
