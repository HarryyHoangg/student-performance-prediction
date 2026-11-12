import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../data/student_performance_cleaned.csv")

print(df.head())


#Xác định X,y
df["Study_Attendance_Index"] = (
    df["Study_Hours"] * df["Attendance(%)"] / 100
)

df_encoded = pd.get_dummies(
    df,
    columns=["Gender"],
    dtype=int
)

X = df_encoded.drop(
    columns=["Test_Score", "Student_ID", "Grade"],
    errors="ignore"
)

y = df_encoded["Test_Score"]

#Chia train test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# Xử lý giá trị khuyết
imputer = SimpleImputer(strategy="median")

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Kiểm tra kết quả
import numpy as np

print("Mean:", X_train_scaled.mean(axis=0))
print("Std:", X_train_scaled.std(axis=0))

print("NaN trong train:", np.isnan(X_train_scaled).sum())
print("NaN trong test:", np.isnan(X_test_scaled).sum())
