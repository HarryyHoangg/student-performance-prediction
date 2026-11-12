import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

# 1. Load dataset
df = pd.read_csv("../data/student_performance_cleaned.csv")

# 2. Feature Engineering
df["Study_Attendance_Index"] = (
    df["Study_Hours"] * df["Attendance(%)"] / 100
)

# 3. X và y
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

# 8. Train model
model.fit(X_train, y_train)

# 9. Permutation Importance
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

# 10. Tạo DataFrame
importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": result.importances_mean,
    "Importance_Std": result.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance_Mean",
    ascending=False
)

print("\n--- PERMUTATION IMPORTANCE ---")
print(importance_df.to_string(index=False))

# 11. Vẽ biểu đồ
plt.figure(figsize=(9, 5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance_Mean"],
    xerr=importance_df["Importance_Std"]
)

plt.xlabel("Increase in MAE after permutation")
plt.ylabel("Features")
plt.title("Random Forest - Permutation Importance")

plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()