import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("../data/student_performance_cleaned.csv")

df["Study_Attendance_Index"] = (
    df["Study_Hours"] * df["Attendance(%)"] / 100
)

X = df.drop(
    columns=["Test_Score", "Student_ID", "Grade"],
    errors="ignore"
)

y = df["Test_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

feature_sets = {
    "All features": [
        "Age",
        "Gender",
        "Study_Hours",
        "Attendance(%)",
        "Study_Attendance_Index"
    ],

    "Study Hours only": [
        "Study_Hours"
    ],

    "Without Gender": [
        "Age",
        "Study_Hours",
        "Attendance(%)",
        "Study_Attendance_Index"
    ],

    "Without engineered index": [
        "Age",
        "Gender",
        "Study_Hours",
        "Attendance(%)"
    ]
}