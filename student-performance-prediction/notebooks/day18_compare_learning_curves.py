import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("../data/student_performance_cleaned.csv")

# Feature Engineering
df["Study_Attendance_Index"] = (
    df["Study_Hours"] * df["Attendance(%)"] / 100
)

# Features
X_all = df[
    [
        "Age",
        "Gender",
        "Study_Hours",
        "Attendance(%)",
        "Study_Attendance_Index"
    ]
]

X_study = df[["Study_Hours"]]

# Target
y = df["Test_Score"]

# Train-test split
X_train_all, X_test_all, y_train, y_test = train_test_split(
    X_all,
    y,
    test_size=0.2,
    random_state=42
)

# Tách Study_Hours theo đúng các dòng train
X_train_study = X_train_all[["Study_Hours"]]    

#Model A
model_a = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    ))
])

#Model B
numeric_features = [
    "Age",
    "Study_Hours",
    "Attendance(%)",
    "Study_Attendance_Index"
]

categorical_features = ["Gender"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

model_b = Pipeline([
    ("preprocessor", preprocessor),

    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1
    ))
])

#Learning Curve
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_sizes = np.linspace(0.1, 1.0, 8)

def calculate_learning_curve(model, X, y, model_name):

    train_sizes_abs, train_scores, validation_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    train_mae = -train_scores
    validation_mae = -validation_scores

    results = pd.DataFrame({
        "Training_Size": train_sizes_abs,
        "Train_MAE": train_mae.mean(axis=1),
        "Train_Std": train_mae.std(axis=1),
        "Validation_MAE": validation_mae.mean(axis=1),
        "Validation_Std": validation_mae.std(axis=1)
    })

    print(f"\n===== {model_name} =====")
    print(results.round(4).to_string(index=False))

    return results

results_a = calculate_learning_curve(
    model_a,
    X_train_study,
    y_train,
    "Model A - Study_Hours only"
)

results_b = calculate_learning_curve(
    model_b,
    X_train_all,
    y_train,
    "Model B - All features"
)


#Biểu đồ so sánh
plt.figure(figsize=(10, 6))

plt.plot(
    results_a["Training_Size"],
    results_a["Validation_MAE"],
    marker="o",
    label="Study_Hours only"
)

plt.plot(
    results_b["Training_Size"],
    results_b["Validation_MAE"],
    marker="o",
    label="All features"
)

plt.xlabel("Training Size")
plt.ylabel("Validation MAE")
plt.title("Learning Curve Comparison")
plt.legend()
plt.grid(True)

plt.savefig("../reports/day18_learning_curve_comparison.png")
plt.show()

#So sánh kết quả
comparison = pd.DataFrame({
    "Model": [
        "Study_Hours only",
        "All features"
    ],

    "Validation_MAE": [
        results_a["Validation_MAE"].iloc[-1],
        results_b["Validation_MAE"].iloc[-1]
    ],

    "Validation_Std": [
        results_a["Validation_Std"].iloc[-1],
        results_b["Validation_Std"].iloc[-1]
    ]
})

print("\n===== FINAL COMPARISON =====")
print(comparison.round(4).to_string(index=False))
