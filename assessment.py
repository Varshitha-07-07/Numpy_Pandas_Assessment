
import numpy as np
import pandas as pd

df = pd.read_csv("student_scores.csv")

print("Dataset Loaded Successfully\n")

print("PART 1 : NumPy Operations \n")

math_scores = np.array(df['math_score'])

print("Math Score Array:")
print(math_scores)

# Mean
print("\nMean of Math Scores:")
print(np.mean(math_scores))

# Median
print("\nMedian of Math Scores:")
print(np.median(math_scores))

# Maximum
print("\nMaximum Math Score:")
print(np.max(math_scores))

# Minimum
print("\nMinimum Math Score:")
print(np.min(math_scores))

# Normalize Scores
normalized = (math_scores - np.min(math_scores)) / (np.max(math_scores) - np.min(math_scores))

print("\nNormalized Scores:")
print(normalized)



import pandas as pd

# Load dataset
df = pd.read_csv("student_scores.csv")

# Display first 5 rows
print("First 5 Rows:")
print(df.head())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Students with attendance below 70%
df["attendance"] = pd.to_numeric(
    df["attendance"],
    errors="coerce"
)

print("\nStudents with attendance below 70%:")
print(df[df["attendance"] < 70])





print("\nPART 3 - DATA PREPROCESSING")

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Convert incorrect formats to proper numeric format
df["age"] = pd.to_numeric(
    df["age"],
    errors="coerce"
)

df["math_score"] = pd.to_numeric(
    df["math_score"],
    errors="coerce"
)

df["science_score"] = pd.to_numeric(
    df["science_score"],
    errors="coerce"
)

df["attendance"] = pd.to_numeric(
    df["attendance"],
    errors="coerce"
)

# Handle missing values
df["age"].fillna(
    df["age"].mean(),
    inplace=True
)

df["math_score"].fillna(
    df["math_score"].mean(),
    inplace=True
)

df["science_score"].fillna(
    df["science_score"].mean(),
    inplace=True
)

df["attendance"].fillna(
    df["attendance"].mean(),
    inplace=True
)

# Fill missing gender values
df["gender"].fillna(
    df["gender"].mode()[0],
    inplace=True
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Convert exam_date to datetime format
df["exam_date"] = pd.to_datetime(
    df["exam_date"],
    errors="coerce"
)

print("\nExam Date Converted:")
print(df["exam_date"].head())

# Detect outliers
Q1_math = df["math_score"].quantile(0.25)
Q3_math = df["math_score"].quantile(0.75)
IQR_math = Q3_math - Q1_math

lower_math = Q1_math - 1.5 * IQR_math
upper_math = Q3_math + 1.5 * IQR_math

Q1_science = df["science_score"].quantile(0.25)
Q3_science = df["science_score"].quantile(0.75)
IQR_science = Q3_science - Q1_science

lower_science = (
    Q1_science - 1.5 * IQR_science
)

upper_science = (
    Q3_science + 1.5 * IQR_science
)

# Remove outliers
df = df[
    (df["math_score"] >= lower_math)
    &
    (df["math_score"] <= upper_math)
]

df = df[
    (df["science_score"] >= lower_science)
    &
    (df["science_score"] <= upper_science)
]

print("\nOutliers Removed")

# Remove duplicate rows
duplicates = df.duplicated().sum()

print("\nDuplicate Rows Found:")
print(duplicates)

df = df.drop_duplicates()

print("\nDuplicates Removed")





print("\nPART 4 - DATA ANALYSIS")

# Create average_score column
df["average_score"] = (
    df["math_score"] +
    df["science_score"]
) / 2

print("\nAverage Score Column Added")
print(df[[
    "name",
    "average_score"
]].head())

# Top 5 students
top_5_students = df.sort_values(
    by="average_score",
    ascending=False
).head(5)

print("\nTop 5 Students Based on Average Score:")
print(top_5_students[[
    "student_id",
    "name",
    "average_score"
]])

# Correlation between attendance and marks
correlation = df["attendance"].corr(
    df["average_score"]
)

print("\nCorrelation Between Attendance and Marks:")
print(correlation)

# Group students by gender
gender_average = df.groupby(
    "gender"
)["average_score"].mean()

print("\nAverage Marks by Gender:")
print(gender_average)