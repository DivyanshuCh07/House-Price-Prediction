# 🏡 House Price Prediction - Machine Learning Project

This project uses regression models (XGBoost + Lasso) to predict housing prices based on a variety of features like area, neighborhood, number of bathrooms, and more.

---

## 📂 Project Structure

| File / Folder          | Description                                  |
|------------------------|----------------------------------------------|
| `House_Price_Prediction.py` | Main Python script for training and prediction |
| `train.csv`            | Training dataset (from Kaggle)               |
| `test.csv`             | Test dataset (from Kaggle)                   |
| `sample_submission.csv`| Sample submission format from Kaggle         |
| `submission.csv`       | Final predictions after running the model    |
| `shap_summary.png`     | SHAP feature importance plot (XGBoost)       |
| `data_description.txt` | Dataset documentation from Kaggle            |
| `venv/`                | Python virtual environment folder (optional) |

---

## ⚙️ How to Run the Script

> 🛠️ **Make sure Python is installed** on your system before proceeding.

### 1. Activate the virtual environment

**For Windows (PowerShell):**
```bash
.\venv\Scripts\activate
