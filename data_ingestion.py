"""
Session 04 – Step 1: Data Ingestion
Reads raw Spaceship Titanic dataset and saves it to ingested folder.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent

RAW_DIR = BASE_DIR / "data" / "raw"
INGESTED_DIR = BASE_DIR / "data" / "ingested"

TRAIN_FILE = RAW_DIR / "train.csv"
TEST_FILE = RAW_DIR / "test.csv"

OUTPUT_TRAIN = INGESTED_DIR / "train.csv"
OUTPUT_TEST = INGESTED_DIR / "test.csv"


def ingest_data():

    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    assert not train_df.empty, "Train dataset is empty"
    assert not test_df.empty, "Test dataset is empty"

    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print(f"Train data ingested → {OUTPUT_TRAIN}")
    print(f"Test data ingested → {OUTPUT_TEST}")


if __name__ == "__main__":
    ingest_data()