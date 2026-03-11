"""
Session 04 – Pipeline Runner
Orchestrates: ingest → preprocess → train → evaluate
"""

import pandas as pd

from data_ingestion import ingest_data
from preprocessing import preprocess_data
from train import baseline_lr, tune_lr, train_final_model
from evaluation import evaluate_model

ACCURACY_THRESHOLD = 0.9


def run_pipeline():

    print("=" * 50)
    print("Step 1: Data Ingestion")

    ingest_data()

    print("\nStep 2: Preprocessing")

    df = pd.read_csv("data/ingested/train.csv")

    X, y, feature_columns = preprocess_data(df, is_train=True)

    print("Features used:", len(feature_columns))

    print("\nStep 3: Training")

    baseline_lr(X, y)

    best_params = tune_lr(X, y)

    train_final_model(X, y, best_params)

    print("\nStep 4: Evaluation")

    accuracy, precision, recall = evaluate_model()

    print("\n" + "=" * 50)

    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED for deployment (accuracy={accuracy:.3f})")
    else:
        print(f"Model REJECTED (accuracy={accuracy:.3f} < threshold={ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    run_pipeline()