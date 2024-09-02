import streamlit as st
import numpy as np
import joblib

# Load the scaler and model
Scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Streamlit app title
st.title("Real Estate Price Prediction")

st.divider()

# Input fields for user to enter number of bedrooms, bathrooms, and house size
bed = st.number_input("Enter the number of bedrooms", value=2, step=1)
bath = st.number_input("Enter the number of bathrooms", value=1, step=1)
size = st.number_input("Enter the house size (in square feet)", value=1000, step=50)

# Combine inputs into a list
X = [bed, bath, size]

st.divider()

# Button for triggering the prediction
predictbutton = st.button("Please press the button for prediction")

st.divider()

# Prediction logic
if predictbutton:
    st.balloons()

    # Convert the list to a numpy array and reshape it to be 2D
    X1 = np.array(X).reshape(1, -1)

    # Scale the input features
    X_array = Scaler.transform(X1)

    # Make the prediction
    prediction = model.predict(X_array)[0]

    # Display the prediction
    st.write(f"The predicted price is ${prediction:.2f}")

else:
    st.write("Please use the button for prediction")
  


