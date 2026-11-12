import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("../data/student_performance_cleaned.csv")

print(df.head())
print("Dataset shape:", df.shape)

# 2. Feature Engineering
df["Study_Attendance_Index"] = (
    df["Study_Hours"] * df["Attendance(%)"] / 100
)

# 3. Tạo X và y
X = df.drop(
    columns=["Test_Score", "Student_ID", "Grade"],
    errors="ignore"
)

y = df["Test_Score"]

# 4. Phân loại features
numeric_features = [
    "Age",
    "Study_Hours",
    "Attendance(%)",
    "Study_Attendance_Index"
]

categorical_features = ["Gender"]

# 5. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 6. Random Forest Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# 8. Train model
model.fit(X_train, y_train)

# 9. Prediction
y_pred = model.predict(X_test)

# 10. Evaluation
print("\n--- RANDOM FOREST EVALUATION ---")

print("MAE:", mean_absolute_error(y_test, y_pred))

print("MSE:", mean_squared_error(y_test, y_pred))

print("R2:", r2_score(y_test, y_pred))