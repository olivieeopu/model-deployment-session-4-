"""
Session 04 – Step 2: Preprocessing
Applies feature engineering and preprocessing for Spaceship Titanic dataset
"""

import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =========================
# FEATURE ENGINEERING
# =========================
def feature_engineering(df):
    """Apply comprehensive feature engineering"""

    df = df.copy()

    # Extract features from Cabin
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown')
    df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split('/')[1] if pd.notna(x) else -1).astype(float)
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown')

    # Extract group features from PassengerId
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    df['Group_size'] = df.groupby('Group')['Group'].transform('count')
    df['Solo'] = (df['Group_size'] == 1).astype(int)

    # Extract first and last name
    df['FirstName'] = df['Name'].apply(lambda x: x.split()[0] if pd.notna(x) else 'Unknown')
    df['LastName'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notna(x) else 'Unknown')
    df['Family_size'] = df.groupby('LastName')['LastName'].transform('count')

    # Spending features
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)

    # Spending ratios
    for col in spending_cols:
        df[f'{col}_ratio'] = df[col] / (df['TotalSpending'] + 1)

    # Age groups
    df['Age_group'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 30, 50, 100],
        labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior']
    ).astype(str)

    # Missing value indicators
    df['Age_missing'] = df['Age'].isna().astype(int)
    df['CryoSleep_missing'] = df['CryoSleep'].isna().astype(int)

    return df


# =========================
# PREPROCESSING
# =========================
def preprocess_data(df, is_train=True):
    """Preprocess the dataset for modeling"""

    df = df.copy()

    # Apply feature engineering
    df = feature_engineering(df)

    # Categorical features
    categorical_features = [
        'HomePlanet',
        'CryoSleep',
        'Destination',
        'VIP',
        'Deck',
        'Side',
        'Age_group'
    ]

    # Numerical features
    numerical_features = [
        'Age',
        'RoomService',
        'FoodCourt',
        'ShoppingMall',
        'Spa',
        'VRDeck',
        'Cabin_num',
        'Group_size',
        'Solo',
        'Family_size',
        'TotalSpending',
        'HasSpending',
        'NoSpending',
        'Age_missing',
        'CryoSleep_missing'
    ] + [col for col in df.columns if '_ratio' in col]

    # Fill missing categorical
    for col in categorical_features:
        df[col] = df[col].fillna('Unknown')

    # Fill missing numerical
    for col in numerical_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical features
    label_encoders = {}

    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Save encoders (training only)
    if is_train:
        with open("model/label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)

    # Select features
    feature_columns = categorical_features + numerical_features

    X = df[feature_columns]

    # Scaling
    scaler = StandardScaler()

    if is_train:
        X_scaled = scaler.fit_transform(X)

        with open("model/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    else:
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

    if is_train:
        y = df['Transported'].astype(int)
        return X_scaled, y, feature_columns
    else:
        return X_scaled, feature_columns


# =========================
# TEST RUN
# =========================
if __name__ == "__main__":

    df = pd.read_csv("data/ingested/train.csv")

    X, y, features = preprocess_data(df, is_train=True)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeatures used: {len(features)}")
    print(features)