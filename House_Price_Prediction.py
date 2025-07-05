# INSTRUCTIONS TO RUN THE SCRIPT:

# 1. Unzip the project folder.

# -- Please download the dataset manually (only test & train files - rest have been zipped properly) as due to size limit on portal, the dataset is not zipped.

# Please make sure to add all the dataset folder in the same directorey only.

# 2. Open terminal (or PowerShell) and navigate to the project folder:
#    cd path\to\unzipped-folder

# 3. Create a virtual environment (optional but recommended):
#    python -m venv venv

# 4. Activate the virtual environment:
#    - Windows:
#        venv\Scripts\activate

# 5. Make sure 'train.csv' and 'test.csv' are present in the same directory.

# 6. Run the script:
#    python House_Price_Prediction.py

# OUTPUT:
# - SHAP summary plot will be saved as 'shap_summary.png'
# - Submission file will be saved as 'submission.csv'


# House_Price_Prediction.py

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso
import xgboost as xgb
import shap

# 1. Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Add indicator flags BEFORE concat
train['ind_train'] = 1
test['ind_train'] = 0
train['SalePrice'] = train['SalePrice']
test['SalePrice'] = np.nan  # Placeholder to unify schema

df = pd.concat([train, test], ignore_index=True)

# 2. Imputation & initial cleanup
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

for col in df.select_dtypes(include='object'):
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(exclude='object'):
    df[col] = df[col].fillna(df[col].median())

# 3. Feature engineering
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
df['TotalBath'] = (df['FullBath'] + df['BsmtFullBath'] + 0.5*(df['HalfBath'] + df['BsmtHalfBath']))

# Price per sqft by neighborhood (for training rows only)
df['Price_per_SqFt_Neigh'] = df['SalePrice'] / df['TotalSF']
grp = df[df['ind_train'] == 1].groupby('Neighborhood')['Price_per_SqFt_Neigh']
df['Price_per_SqFt_Neigh'] = df['Price_per_SqFt_Neigh'].fillna(df['Neighborhood'].map(grp.median()))

# Random dummy feature trick
df['random_dummy'] = np.random.randn(len(df))

# 4. Encoding
qual_cols = [c for c in df.columns if 'Qual' in c or 'Cond' in c or 'Quality' in c]
ord_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
for c in qual_cols:
    if df[c].dtype == 'object':
        df[c] = df[c].map(ord_map).fillna(0)

# One-hot encode rest of categoricals
df = pd.get_dummies(df, drop_first=True)

# 5. Split back to train/test
train_df = df[df['ind_train'] == 1]
test_df = df[df['ind_train'] == 0]

X = train_df.drop(['SalePrice', 'ind_train', 'Id'], axis=1)
y = np.log1p(train_df['SalePrice'])
X_test = test_df.drop(['SalePrice', 'ind_train', 'Id'], axis=1)

# 6. Feature importance with random dummy
model_rf = xgb.XGBRegressor(n_estimators=100, random_state=0)
model_rf.fit(X, y)
importances = pd.Series(model_rf.feature_importances_, index=X.columns)
features_to_drop = importances[importances < importances['random_dummy']].index

X.drop(columns=features_to_drop, inplace=True)
X_test.drop(columns=features_to_drop, inplace=True)

# 7. Scaling
scaler = RobustScaler()
X_s = scaler.fit_transform(X)
X_test_s = scaler.transform(X_test)

# 8. Modeling
kf = KFold(n_splits=5, shuffle=True, random_state=0)
lasso = Lasso(alpha=0.0005, random_state=0)
xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)

mean_rmse = lambda m: np.mean(np.sqrt(-cross_val_score(m, X_s, y, scoring='neg_mean_squared_error', cv=kf)))
print("Lasso CV RMSE:", mean_rmse(lasso))
print("XGB CV RMSE:", mean_rmse(xgb_model))

# 9. Fit ensemble
lasso.fit(X_s, y)
xgb_model.fit(X_s, y)
preds = 0.3 * np.expm1(lasso.predict(X_test_s)) + 0.7 * np.expm1(xgb_model.predict(X_test_s))

# 10. SHAP interpretation
explainer = shap.Explainer(xgb_model, X_s)
shap_values = explainer(X_s)
shap.summary_plot(shap_values, pd.DataFrame(X, columns=X.columns), plot_type="bar", show=False)
plt.savefig('shap_summary.png')

# 11. Submission
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': preds})
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
