import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_wine

# Load the Wine dataset
wine = load_wine()

# Load the saved model
with open('wine_classifier.pkl', 'rb') as file:
    model = pickle.load(file)



# function to take input and return the output answer
def predict_wine(data):
    # Make a prediction
    prediction = model.predict([data])
    return prediction[0]



# main app from our streamlit starts working here
def main():
    st.title("Wine Classification")

    # Input fields for all 13 features
    features = wine.feature_names
    inputs = []

    for feature in features:
        value = st.number_input(f"{feature}:", key=feature)
        inputs.append(value)

    if st.button("Predict"):
        prediction = predict_wine(inputs)
        target_names = wine.target_names
        st.success(f"The predicted wine class is: {target_names[prediction]}")

if __name__ == '__main__':
    main()
