import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/student_performance_cleaned.csv")

print(df.head())
print(df.describe())


#Phân tích phân phối điểm số
plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Test_Score",
    bins=15,
    kde=True
)

plt.title("Distribution of Test Scores")
plt.xlabel("Test Score")
plt.ylabel("Number of Students")

plt.show()


#Phát hiện Outliers bằng Boxplot
plt.figure(figsize=(8, 5))

sns.boxplot(data=df, y="Test_Score")

plt.title("Test Score Boxplot")
plt.show()


#Phân tích Study Hours và Test Score
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Study_Hours",
    y="Test_Score",
    hue="Gender"
)

sns.regplot(
    data=df,
    x="Study_Hours",
    y="Test_Score",
    scatter=False,
    color="red"
)

plt.title("Study Hours vs Test Score")
plt.show()


#Phân tích Study Hours và Test Score
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Attendance(%)",
    y="Test_Score"
)

sns.regplot(
    data=df,
    x="Attendance(%)",
    y="Test_Score",
    scatter=False,
    color="red"
)

plt.title("Attendance vs Test Score")
plt.show()

    
#Correlation Matrix
numeric_df = df.select_dtypes(include="number")

correlation = numeric_df.corr()

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Matrix")
plt.show()