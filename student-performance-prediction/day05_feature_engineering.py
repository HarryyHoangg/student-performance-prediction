import pandas as pd

df = pd.read_csv("data/student_performance_cleaned.csv")
print(df.head())

#Tạo đặc trưng mới
df["Study_Attendance_Index"] = (
    df["Study_Hours"] * df["Attendance(%)"] / 100
)

print(df[[
    "Study_Hours",
    "Attendance(%)",
    "Study_Attendance_Index"
]].head())

# Phân loại các đặc trưng
numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns

print("Numeric:", numeric_columns)
print("Categorical:", categorical_columns)

#Thử mã hóa biến phân loại
df_encoded = pd.get_dummies(
    df,
    columns=["Gender"],
    dtype=int
)

print(df_encoded.head())

#Xác định X và y
X = df_encoded.drop(
    columns=["Test_Score", "Student_ID", "Grade"],
    errors="ignore"
)

y = df_encoded["Test_Score"]

print(X.head())
print(y.head())