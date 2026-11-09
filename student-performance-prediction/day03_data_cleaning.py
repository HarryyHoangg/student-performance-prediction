import pandas as pd
import numpy as np

# Doc du lieu
df = pd.read_csv("data/student_performance.csv")
# Tao ban sao, giu nguyen file goc
df_clean = df.copy()

# 1. Xoa dong trung lap hoan toan
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print("Removed duplicates:", before - len(df_clean))

# 2. Chuyen cac cot ve dang so
columns = ["Study_Hours", "Attendance(%)", "Test_Score"]

for col in columns:
    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

# 3. Xu ly Study_Hours am
df_clean.loc[df_clean["Study_Hours"] < 0, "Study_Hours"] = np.nan

# 4. Xu ly Attendance ngoai khoang 0-100
invalid_attendance = ~df_clean["Attendance(%)"].between(0, 100)
df_clean.loc[invalid_attendance, "Attendance(%)"] = np.nan

# 5. Xu ly Test_Score ngoai khoang 0-100
invalid_score = ~df_clean["Test_Score"].between(0, 100)
df_clean.loc[invalid_score, "Test_Score"] = np.nan

# 6. Kiem tra missing values sau khi lam sach
print(df_clean.isnull().sum())

# 7. Luu dataset moi, khong ghi de file goc
df_clean.to_csv("data/student_performance_cleaned.csv", index=False)

print("New shape:", df_clean.shape)