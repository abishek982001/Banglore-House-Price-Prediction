import pickle
import streamlit as st
import json
import numpy as np


def get_estimated_price(location, sqft, bhk, bath):
    global model
    global data_columns
    # st.write(len(data_columns))
    try:
        loc_index = data_columns.index(location.lower())
    except Exception:
        loc_index = -1
    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 3)


st.title("Banglore House Price Prediction")
f = open("columns.json")
data = json.load(f)
data_columns = data["data_columns"]
area = st.sidebar.number_input("Area", 300)
locations = data_columns[3:]
location = st.sidebar.selectbox("Location", locations+["Others"])
bhk = st.sidebar.slider("BHK", 1, 16)
bath = st.sidebar.slider("Bath", 1, 16)

with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)


price = get_estimated_price(location, area, bhk, bath)
st.subheader("Predicted Price")
st.write(price)
