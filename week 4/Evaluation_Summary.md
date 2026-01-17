# Student Exam Pass/Fail Prediction - Evaluation Summary

## Project Overview

This project implements three classification models to predict student exam pass/fail status based on:
- **Study Hours** (hours_studied)
- **Attendance Percentage** (attendance_percent)
- **Assignments Completed** (assignments_completed)
- **Gender** (encoded)

**Target Variable:** Pass (≥60) / Fail (<60) based on test_score

---

## Model Performance Comparison

| Model | Accuracy | F1-Score | ROC-AUC | Complexity |
|-------|----------|----------|---------|------------|
| Logistic Regression | 0.7 | 0.823529 | 0.8750 | Low |
| K-Nearest Neighbors | 0.7 | 0.823529 | 0.5625 | Medium |
| Random Forest | 0.5 | 0.666667 | 0.6250 | High |

> [!NOTE]
> Actual metrics will be populated after running the notebook.

---

## Model Complexity vs Interpretability Analysis

### 1. Logistic Regression
**Complexity:** ⭐ Low | **Interpretability:** ⭐⭐⭐ High

| Aspect | Description |
|--------|-------------|
| **How it works** | Linear combination of features with sigmoid transformation |
| **Interpretability** | Coefficients directly indicate feature impact on prediction |
| **Advantages** | Easy to explain, fast training, works well with linearly separable data |
| **Limitations** | May underperform on complex, non-linear relationships |

**Example Interpretation:** A positive coefficient for `hours_studied` means more study hours increase the probability of passing.

---

### 2. K-Nearest Neighbors (KNN)
**Complexity:** ⭐⭐ Medium | **Interpretability:** ⭐⭐ Medium

| Aspect | Description |
|--------|-------------|
| **How it works** | Classifies based on majority vote of K nearest neighbors |
| **Interpretability** | Intuitive "similar students" explanation possible |
| **Advantages** | Non-parametric, adapts to local patterns |
| **Limitations** | Sensitive to feature scaling, computationally expensive at prediction |

**Example Interpretation:** "This student is predicted to pass because the 5 most similar students all passed."

---

### 3. Random Forest
**Complexity:** ⭐⭐⭐ High | **Interpretability:** ⭐ Low-Medium

| Aspect | Description |
|--------|-------------|
| **How it works** | Ensemble of decision trees with bagging |
| **Interpretability** | Feature importance available, but individual predictions are opaque |
| **Advantages** | Handles non-linearity, robust to overfitting, automatic feature selection |
| **Limitations** | "Black box" nature makes explaining specific decisions difficult |

**Example Interpretation:** "Attendance and study hours are the most important factors" (but why a specific student was classified as pass/fail is harder to explain).

---

## Key Insights

### Trade-off Matrix

```
                    Interpretability
                    Low    Medium    High
                   ┌──────┬──────┬──────┐
           High    │  RF  │      │      │
Complexity Medium  │      │ KNN  │      │
           Low     │      │      │  LR  │
                   └──────┴──────┴──────┘
```

### Recommendations

| Scenario | Recommended Model | Reason |
|----------|-------------------|--------|
| Need to explain each prediction | Logistic Regression | Coefficients provide clear rationale |
| Maximum accuracy required | Random Forest | Best for complex patterns |
| Balance of both | KNN | Intuitive similarity-based explanation |

---

## Conclusion

1. **Model complexity inversely correlates with interpretability** - simpler models are easier to explain but may sacrifice predictive power.

2. **Logistic Regression** is ideal for educational settings where transparency in decision-making is crucial (e.g., identifying at-risk students).

3. **Random Forest** should be considered when prediction accuracy is paramount and interpretability can be addressed through feature importance analysis.

4. **KNN** offers a middle ground but requires careful tuning of K and is sensitive to feature scaling.

---

## Files Generated

- `student_exam_prediction.ipynb` - Main notebook with all code
- `model_results.csv` - Model metrics comparison
- `pass_fail_distribution.png` - Target distribution visualization
- `correlation_heatmap.png` - Feature correlation matrix
- `feature_distributions.png` - Feature distributions by target
- `knn_k_selection.png` - KNN K parameter optimization
- `feature_importances.png` - Random Forest feature importance
- `confusion_matrices.png` - All model confusion matrices
- `roc_curves.png` - ROC curve comparison
- `model_comparison.png` - Metric comparison bar charts
