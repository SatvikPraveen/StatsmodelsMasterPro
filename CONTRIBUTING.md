# 🤝 Contributing Guidelines

Welcome, and thank you for your interest in contributing to this project!  
We’re excited to collaborate with developers, data scientists, and learners of all levels.

---

## 📌 Project Overview

This project is designed as a **modular, educational, and production-ready repository** to master core concepts in:
- 📊 Statistical modeling (using `statsmodels`)
- 📈 Interactive data dashboards (using `streamlit`)
- 🐳 Containerization and reproducibility (via `Docker`)

We aim to make this a reliable reference for building clean, reproducible, and extensible workflows in data science.

---

## 🛠️ How to Contribute

### 🔧 1. Fork the Repository
Start by forking this repository and cloning it to your local machine:
```bash
git clone https://github.com/SatvikPraveen/StatsmodelsMasterPro.git
cd REPO_NAME
```

### 🧪 2. Set Up the Environment

You can either:

* Use Docker (`docker-compose up --build`)
* Or set up manually using the `requirements.txt` file:

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 📂 3. Follow the Project Structure

Ensure your additions follow the existing modular structure:

```
project/
├── streamlit_app/
├── notebooks/
├── utils/
├── synthetic_data/
├── exports/
├── cheatsheets/
└── ...
```

### 🧹 4. Code Style and Formatting

Please maintain clean and readable code:

* Use **PEP8** guidelines for Python code
* Modularize logic into functions or utilities where appropriate
* Add **docstrings** and **inline comments** if needed

> ✨ Tip: Run `black` or `isort` if you use formatters.

---

## 🧪 Tests & Validation

Before pushing your changes:

* Run the app locally via `streamlit run Home.py` or check Jupyter notebooks.
* Make sure your feature or fix doesn't break existing functionality.
* Use assert statements for small utilities where appropriate.

---

## 🚀 Submitting a Pull Request

1. Create a feature branch:
   `git checkout -b feature/your-feature-name`
2. Commit your changes:
   `git commit -m "Add: brief summary of the change"`
3. Push to your fork:
   `git push origin feature/your-feature-name`
4. Open a pull request to the `main` branch of this repo

---

## 💬 Code of Conduct

By participating in this project, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md). We strive to create a welcoming, inclusive, and respectful environment.

---

## 🙌 Ways to Contribute

Even if you don’t write code, you can help in many ways:

* Improve documentation or cheatsheets
* Report bugs or suggest features
* Share real-world use cases
* Review open pull requests

---

## 📬 Need Help?

Open a [Discussion](https://github.com/SatvikPraveen/StatsmodelsMasterPro/discussions) or create an [Issue](https://github.com/SatvikPraveen/StatsmodelsMasterPro/issues) — we’re happy to support you.

---

Thank you again for contributing — your work helps make this project better for everyone! 🌟
