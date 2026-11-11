import pandas as pd

# 1. Doc dataset
df = pd.read_csv("data/student_performance.csv")

# 2. Kiem tra kich thuoc
print("Shape:", df.shape)

# 3. Kiem tra 5 dong dau
print(df.head())

# 4. Kiem tra kieu du lieu
print(df.info())

# 5. Kiem tra missing values
print(df.isnull().sum())

# 6. Kiem tra duplicate
print("Duplicates:", df.duplicated().sum())

# 7. Kiem tra Student_ID trung
print("Duplicate IDs:",
      df["Student_ID"].duplicated().sum())

# 8. Kiem tra gia tri bat thuong
print("Negative study hours:",
      (df["Study_Hours"] < 0).sum())

print("Attendance > 100:",
      (df["Attendance(%)"] > 100).sum())