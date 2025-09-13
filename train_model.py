# train_model.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import sklearn
from math import sqrt

# ---- Load dataset ----
possible = ['medical_insurance.csv', './data/medical_insurance.csv', '/mnt/data/medical_insurance.csv']
csv_path = next((p for p in possible if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("medical_insurance.csv not found. Place it in the project root (or ./data/).")

df = pd.read_csv(csv_path)
print("Loaded", csv_path, "shape:", df.shape)

if 'charges' not in df.columns:
    raise ValueError("'charges' column (target) not found in CSV.")

# ---- Features & target ----
X = df.drop(columns=['charges'])
y = df['charges']

numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# ---- Handle OneHotEncoder across sklearn versions ----
try:
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # sklearn >=1.2
except TypeError:
    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)  # older sklearn

num_pipeline = Pipeline([('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features),
        ('cat', cat_encoder, categorical_features)
    ],
    remainder='drop'
)

# ---- Build pipeline with XGBoost ----
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1))
])

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
pipeline.fit(X_train, y_train)

# ---- Evaluate ----
preds = pipeline.predict(X_test)

# Handle sklearn version differences for RMSE
try:
    rmse = mean_squared_error(y_test, preds, squared=False)  # sklearn >=0.22
except TypeError:
    rmse = sqrt(mean_squared_error(y_test, preds))  # older sklearn

r2 = r2_score(y_test, preds)

print(f"Test RMSE: {rmse:.2f}  R2: {r2:.3f}")

# ---- Save model ----
model_path = 'model_xgb.pkl'
joblib.dump(pipeline, model_path)
print("Saved trained pipeline to", model_path)
