# Accuracy Report

| Item | Value |
|---|---|
| Dataset | data/iris.csv |
| Rows before / after cleaning | 150 / 149 |
| Train / Test split | 119 / 30 |
| Algorithm | Decision Tree Classifier |
| **Accuracy** | **0.9333** |

## Confusion matrix (rows = actual, cols = predicted)

```
            Setosa  Versicolor  Virginica
Setosa          10           0          0
Versicolor       0           9          1
Virginica        0           1          9
```

## Classification report

```
              precision    recall  f1-score   support

      Setosa       1.00      1.00      1.00        10
  Versicolor       0.90      0.90      0.90        10
   Virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30
```
