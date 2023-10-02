import numpy as np
import pandas as pd
import pickle
import json
import streamlit as st

# Load the pickle model and data columns JSON file
with open("mum.pickle", 'rb') as file:
    loaded_model = pickle.load(file)

with open("cols.json", 'r') as obj:
    data_columns = json.load(obj)["cols"]
    area_types = data_columns[2:7]
    locations = data_columns[7:]

# Create a Streamlit app
def main():
    st.title("Mumbai House Price Predictor")

    # User input widgets
    area_type = st.selectbox("Area Type", area_types)
    location = st.selectbox("Location", locations)
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    area = st.number_input("Area (sqft)", min_value=100, max_value=5000, step=100, value=1000)

    # Make predictions based on user input
    if st.button("Predict"):
        sample_data = {
            "area_type": area_type,
            "location": location,
            "bhk": bhk,
            "area": area
        }

        # Create a DataFrame from the sample data (only one row)
        sample_df = pd.DataFrame([sample_data])

        # Preprocess the sample data (one-hot encoding for categorical variables)
        sample_df_encoded = pd.get_dummies(sample_df, columns=["area_type", "location"])

        # Initialize a NumPy array for the input features
        sample_features = np.zeros(len(data_columns))

        # Map the sample data to the feature array based on column names
        for col_name in sample_df_encoded.columns:
            if col_name in data_columns:
                col_index = data_columns.index(col_name)
                sample_features[col_index] = sample_df_encoded[col_name].values[0]

        # Make predictions for the sample data
        predicted_price = loaded_model.predict([sample_features])[0]
        predicted_price = round(predicted_price, 2)

        st.success(f"Predicted Price: {predicted_price} lakhs")

if __name__ == "__main__":
    main()
