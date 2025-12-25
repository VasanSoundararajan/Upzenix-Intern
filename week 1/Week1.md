# Student Performance Dataset Analysis

This project refactors procedural data analysis code into a reusable, object-oriented structure for analyzing student performance data.

## Overview

The core of this project is the `StudentPerformanceDataset` class, which encapsulates the entire data processing pipeline, from loading to machine learning.

### Key Features

1.  **Data Loading & Cleaning**:
    *   Handles CSV loading with custom `NA` values.
    *   Automatically removes missing values and duplicates.
    *   Removes non-predictive columns like `student_id`.

2.  **Robust Preprocessing**:
    *   **Automated Encoding**: Identifies and encodes **all** categorical (object) columns using `LabelEncoder`. This ensures compatibility with numerical machine learning models like Linear Regression.

3.  **Visualization**:
    *   Built-in methods to generate insights:
        *   `plot_exam_vs_mental_health()`: Bar chart analysis.
        *   `plot_gender_distribution()`: Pie chart of demographics.
        *   `plot_correlation()`: Heatmap showing relationships between features.

4.  **Machine Learning Ready**:
    *   `get_features_targets()` method to seamlessly split data into `X` (Features) and `y` (Target) for Scikit-Learn pipelines.

## Requirements

*   Python 3.x
*   pandas
*   matplotlib
*   seaborn
*   scikit-learn

## Usage Example

```python
from Week1 import StudentPerformanceDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Initialize and Process Data
dataset = StudentPerformanceDataset("student_habits_performance.csv")
df = dataset.load_and_clean()

# 2. Visualize
dataset.plot_exam_vs_mental_health()

# 3. Encode Categorical Variables
dataset.encode_categorical()

# 4. Get Features and Target
X, y = dataset.get_features_targets(target_column='exam_score')

# 5. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
```

## Structure

*   `Week1.ipynb`: The main notebook containing the class definition and execution logic.
*   `student_habits_performance.csv`: The input dataset.

## Dataset

*   `student_habits_performance.csv`: Student performance dataset containing various features like exam score, mental health, gender, etc.
*   Kaggle Dataset Link: [Student Performance Dataset by Jayanta Nath](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance)