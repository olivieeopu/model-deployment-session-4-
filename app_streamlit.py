"""
Session 04 – Streamlit App
Serve the trained Spaceship Titanic Logistic Regression model
Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from preprocessing import feature_engineering, preprocess_data

# load trained model
model = joblib.load(Path(__file__).parent / "model/logistic_model.pkl")


def main():

    st.title("ASG 04 MD - Fransiska - Spaceship Titanic Model Deployment")
    st.write("Enter passenger information")

# fitur

    PassengerId = st.text_input("Passenger ID", "0001_01")

    HomePlanet = st.selectbox(
        "Home Planet",
        ["Earth", "Europa", "Mars"]
    )

    CryoSleep = st.selectbox(
        "CryoSleep",
        [True, False]
    )

    Cabin = st.text_input(
        "Cabin (Contoh input: B/45/P)",
        "B/45/P"
    )

    Destination = st.selectbox(
        "Destination",
        ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"]
    )

    Age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=25
    )

    VIP = st.selectbox(
        "VIP",
        [True, False]
    )

    RoomService = st.number_input("RoomService", value=0.0)
    FoodCourt = st.number_input("FoodCourt", value=0.0)
    ShoppingMall = st.number_input("ShoppingMall", value=0.0)
    Spa = st.number_input("Spa", value=0.0)
    VRDeck = st.number_input("VRDeck", value=0.0)
    Name = st.text_input("Name", "John Doe")

    # PREDICTION 
    if st.button("Predict"):

        features = pd.DataFrame({

            "PassengerId": [PassengerId],
            "HomePlanet": [HomePlanet],
            "CryoSleep": [CryoSleep],
            "Cabin": [Cabin],
            "Destination": [Destination],
            "Age": [Age],
            "VIP": [VIP],
            "RoomService": [RoomService],
            "FoodCourt": [FoodCourt],
            "ShoppingMall": [ShoppingMall],
            "Spa": [Spa],
            "VRDeck": [VRDeck],
            "Name": [Name]

        })

        result = make_prediction(features)

        if result == 1:
            st.success("Passenger will be transported")
        else:
            st.error("Passenger will NOT be transported")


def make_prediction(features):

    # apply feature engineering
    df = feature_engineering(features)

    # preprocessing
    X, _ = preprocess_data(df, is_train=False)

    # model prediction
    prediction = model.predict(X)

    return prediction[0]


if __name__ == "__main__":
    main()