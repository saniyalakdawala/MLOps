from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "iris.csv"
CLEAN_FILE = ROOT / "data" / "iris_clean.csv"
MODEL_FILE = ROOT / "models" / "iris_decision_tree.joblib"
REPORT_FILE = ROOT / "docs" / "Accuracy_Report.md"

FEATURES = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
TARGET = "variety"

# ---------------------------------------------------------------- US1: load
df = pd.read_csv(RAW_FILE)
print("=== US1: Dataset loaded ===")
print(f"File: {RAW_FILE.name} | rows: {len(df)} | columns: {list(df.columns)}\n")

# ---------------------------------------------------------------- US2: clean
print("=== US2: Cleaning ===")
rows_start = len(df)

df.columns = df.columns.str.strip().str.lower()           # tidy column names
df[TARGET] = df[TARGET].astype(str).str.strip().str.title()  # tidy labels
for col in FEATURES:                                       # force numeric
    df[col] = pd.to_numeric(df[col], errors="coerce")

missing = int(df.isna().sum().sum())
df = df.dropna()                                           # missing values
dupes = int(df.duplicated().sum())
df = df.drop_duplicates()                                  # duplicate rows
invalid = int((df[FEATURES] <= 0).any(axis=1).sum())
df = df[(df[FEATURES] > 0).all(axis=1)]                    # impossible sizes
df = df.reset_index(drop=True)

print(f"Missing values removed : {missing}")
print(f"Duplicate rows removed : {dupes}")
print(f"Invalid rows removed   : {invalid}")
print(f"Rows: {rows_start} -> {len(df)}")
print("Class counts:\n" + df[TARGET].value_counts().to_string() + "\n")
df.to_csv(CLEAN_FILE, index=False)
print(f"Clean data saved to {CLEAN_FILE.relative_to(ROOT)}\n")

# ---------------------------------------------------------------- split
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------- US3: train
print("=== US3: Training Decision Tree ===")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")

# ---------------------------------------------------------------- US4: evaluate
print("=== US4: Evaluation ===")
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
labels = sorted(y.unique())
cm = confusion_matrix(y_test, pred, labels=labels)
report = classification_report(y_test, pred)
print("Accuracy:", accuracy)
print("\nConfusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
print("\nClassification report:\n" + report)

# ---------------------------------------------------------------- US5: save
print("=== US5: Save model ===")
MODEL_FILE.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_FILE)
print(f"Model saved to {MODEL_FILE.relative_to(ROOT)}")

# sample prediction using the saved model
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=FEATURES)
print("Sample prediction for [5.1, 3.5, 1.4, 0.2]:",
      joblib.load(MODEL_FILE).predict(sample)[0])

# ---------------------------------------------------------------- report
REPORT_FILE.parent.mkdir(exist_ok=True)
REPORT_FILE.write_text(
    "# Accuracy Report\n\n"
    "| Item | Value |\n|---|---|\n"
    f"| Dataset | data/iris.csv |\n"
    f"| Rows before / after cleaning | {rows_start} / {len(df)} |\n"
    f"| Train / Test split | {len(X_train)} / {len(X_test)} |\n"
    "| Algorithm | Decision Tree Classifier |\n"
    f"| **Accuracy** | **{accuracy:.4f}** |\n\n"
    "## Confusion matrix (rows = actual, cols = predicted)\n\n```\n"
    + pd.DataFrame(cm, index=labels, columns=labels).to_string()
    + "\n```\n\n## Classification report\n\n```\n" + report + "```\n"
)
print(f"Report written to {REPORT_FILE.relative_to(ROOT)}")
