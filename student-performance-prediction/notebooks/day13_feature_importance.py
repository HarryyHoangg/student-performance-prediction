import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# 1. Load dataset
df = pd.read_csv("../data/student_performance_cleaned.csv")

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

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Tạo hai mô hình
models = {
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=2,
            min_samples_split=5,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
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
}

# 8. Huấn luyện và lấy Feature Importance
importance_results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    # Lấy tên features sau preprocessing
    feature_names = (
        model.named_steps["preprocessor"]
        .get_feature_names_out()
    )

    # Lấy mô hình bên trong Pipeline
    regressor = model.named_steps["regressor"]

    # Lấy Feature Importance
    importances = regressor.feature_importances_

    # Tạo DataFrame
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    importance_results[name] = importance_df

    print(f"\n--- {name} FEATURE IMPORTANCE ---")
    print(importance_df.to_string(index=False))

# 9. Vẽ biểu đồ
for name, importance_df in importance_results.items():

    plt.figure(figsize=(9, 5))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(f"{name} - Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()