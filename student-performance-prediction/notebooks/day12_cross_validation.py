import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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

# 4. Features
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

# 7. Tuned Decision Tree
decision_tree = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", DecisionTreeRegressor(
        max_depth=5,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=42
    ))
])

# 8. Random Forest
random_forest = Pipeline([
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

# 9. KFold
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 10. Cross Validation
models = {
    "Tuned Decision Tree": decision_tree,
    "Random Forest": random_forest
}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kfold,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    mae_scores = -scores

    print(f"\n--- {name} ---")
    print("Fold MAE:", mae_scores)
    print("Mean MAE:", mae_scores.mean())
    print("Std MAE:", mae_scores.std())

# 11. Đánh giá trên Test
print("\n--- TEST SET EVALUATION ---")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))