# 📊 Data Science Projects

> A collection of end-to-end data science projects — from exploratory analysis to model training and evaluation.

---

### Tech Stack

![Python](https://img.shields.io/badge/Python-≥3.14-3776AB?logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-≥3.0.1-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-≥2.4.2-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.8.0-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-≥3.2.0-FF6600?logo=xgboost&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-≥3.13.2-D00000?logo=keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-≥3.10.8-11557C?logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-≥0.13.2-444876?logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-≥1.1.1-F37626?logo=jupyter&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-≥0.133.1-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-≥1.19.0-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📁 Projects

| # | Project | Description | Status |
|---|---------|-------------|--------|
| 1 | [🏠 House Price Prediction](house_price_prediction/) | Regression on the Ames Housing dataset (2,930 samples, 82 features). Linear Regression vs Random Forest with full sklearn pipeline. | ✅ Complete |
| 2 | [💳 Loan Default Prediction](loan_default_prediction/) | Binary classification on loan repayment data. | 🔜 Coming soon |
| 3 | 🩺 Heart Disease Classification | Multi-class classification on clinical data. | 📋 Planned |
| 4 | 🛒 Customer Churn Prediction | Churn analysis with gradient boosting and feature importance. | 📋 Planned |
| 5 | 📝 Sentiment Analysis (NLP) | Text classification with deep learning (Keras). | 📋 Planned |
| … | *More projects added over time* | | |

---

## 📂 Repository Structure

```
data-science-projects/
├── pyproject.toml                  # Shared dependencies (managed by uv)
├── init.ps1                        # Run once after cloning (installs deps + kernel)
├── setup.ps1                       # New-project generator script
├── create_notebook.py              # EDA notebook template generator
├── README.md
├── LICENSE
│
├── house_price_prediction/
│   ├── data/ames-housing.csv
│   ├── docs/data-dictionary.md
│   ├── models/
│   │   ├── linear_regression.pkl
│   │   └── random_forest.pkl
│   ├── notebooks/exploration.ipynb
│   ├── reports/figures/
│   ├── src/train.py
│   └── README.md
│
├── loan_default_prediction/
│   ├── data/
│   ├── docs/
│   ├── models/
│   ├── notebooks/exploration.ipynb
│   ├── reports/figures/
│   ├── src/
│   └── README.md
│
├── <project_3>/                    # Same structure for every project
│   ├── data/
│   ├── docs/
│   ├── models/
│   ├── notebooks/exploration.ipynb
│   ├── reports/figures/
│   ├── src/train.py
│   └── README.md
│
└── ...
```

Every project follows the same layout: `data/` → `notebooks/` (EDA) → `src/` (training) → `models/` (artefacts) → `reports/figures/` (plots).

---

## 🛠️ Setup

### Prerequisites — install uv (recommended)

This repo uses [**uv**](https://docs.astral.sh/uv/) as the Python package manager.  
uv is **10–100× faster** than pip, handles virtual environments automatically, and locks exact dependency versions for full reproducibility via `uv.lock`.

**Install uv:**

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> See the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/) for more options.

### Clone & install

```bash
git clone https://github.com/DanciVasile/data-science-projects.git
cd data-science-projects
.\init.ps1       # installs dependencies + registers Jupyter kernel
```

The `init.ps1` script runs `uv sync` (creates `.venv`) and registers it as a Jupyter kernel named **"Data Science Projects"** so every notebook uses the correct environment out of the box.

Then navigate into any project folder and follow its README.

### Alternative (pip)

If you prefer pip over uv:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
pip install -e .
```

---

## 🆕 Starting a New Project

Two scripts automate new project creation so every project starts with the same structure and conventions:

### `setup.ps1` — Generate a full project folder

```powershell
mkdir my_new_project; cd my_new_project
& ..\setup.ps1
```

Creates:

```
my_new_project/
├── data/
├── docs/
├── models/
├── notebooks/
│   └── exploration.ipynb   ← generated from EDA template
├── reports/figures/
└── src/
```

### `create_notebook.py` — Generate the EDA notebook only

```bash
uv run create_notebook.py notebooks       # into a subfolder
uv run create_notebook.py                  # into current directory
```

The notebook template includes 10 sections (imports, config, loading, target analysis, missing values, cleaning, univariate EDA, bivariate EDA, outlier detection, summary) with a `TASK` variable that switches between regression and classification behaviour automatically.

---

## 🤖 Automated Notebook Setup

Every generated notebook comes with **two automations** so you (and anyone who clones the repo) can start working immediately — no manual path editing or kernel hunting.

### Automatic data path resolution

The configuration cell resolves the project's `data/` directory automatically, regardless of where the kernel's working directory is set:

| Environment | How it resolves |
|---|---|
| **VS Code** | Uses the `__vsc_ipynb_file__` variable injected by the Jupyter extension |
| **JupyterLab / Classic Jupyter** | Walks up from the kernel's CWD (which Jupyter sets to the notebook's directory) |

**All you need to do** is drop your dataset into the project's `data/` folder and set the filename:

```python
DATA_FILE = "my-dataset.csv"   # ← only thing you change
```

The full path is built automatically:

```python
DATA_PATH = PROJECT_DIR / "data" / DATA_FILE
```

### Automatic kernel selection

Notebooks are pre-configured to use the **"Data Science Projects"** kernel registered by `init.ps1`.  
After running `.\init.ps1` once, every notebook will pick up the correct `.venv` kernel automatically — no need to manually select an interpreter.

> **Tip:** If you open a notebook and the kernel shows "Select Kernel", just run `.\init.ps1` again to re-register it.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ by <strong>Vasile-Marian Danci</strong>
  <br/><br/>
  <a href="https://github.com/DanciVasile">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
  &nbsp;
  <a href="https://www.linkedin.com/in/vasile-danci-m/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"/>
  </a>
</p>
