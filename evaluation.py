"""
Session 04 – Step 4: Evaluation
Loads trained model, evaluates on dataset, and prints metrics.
"""

import pickle
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score
from preprocessing import preprocess_data


def evaluate_model():
    # load dataset
    df = pd.read_csv("data/ingested/train.csv")

    X, y, feature_columns = preprocess_data(df, is_train=True)

    # load model
    with open("model/logistic_model.pkl", "rb") as f:
        model = pickle.load(f)

    # prediction
    preds = model.predict(X)

    # metrics
    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec  = recall_score(y, preds)

    print(
        f"Evaluation | Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}"
    )

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")

    print("==============================")

    return acc, prec, rec


if __name__ == "__main__":
    evaluate_model()