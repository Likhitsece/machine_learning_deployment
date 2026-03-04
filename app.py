import numpy as np 
from flask import jsonify, request, Flask

app = Flask(__name__)
import joblib

# Define the filename for the exported model
model_filename = 'logistic_regression_model.joblib'

# Load the saved dictionary containing the model, label_encoder, and scaler
loaded_assets_retrained = joblib.load(model_filename)

# Extract the model, label_encoder, and scaler from the loaded dictionary
loaded_model_retrained = loaded_assets_retrained['model']
loaded_label_encoder_retrained = loaded_assets_retrained['label_encoder']
loaded_scaler_retrained = loaded_assets_retrained['scaler']

print(f"Retrained model, label encoder, and scaler loaded successfully from {model_filename}")
print(f"Loaded Model Type: {type(loaded_model_retrained)}")
print(f"Loaded Label Encoder Type: {type(loaded_label_encoder_retrained)}")
print(f"Loaded Label Encoder classes: {loaded_label_encoder_retrained.classes_}")
print(f"Loaded Scaler Type: {type(loaded_scaler_retrained)}")
# Define a root endpoint
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Milk Quality Prediction API!"})

# Define a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json(force=True)
    
    # Assuming the input data is a dictionary matching the feature names
    # Example expected input: {"pH": 6.8, "Temp (°C)": 4, "Gas (ppm)": 120, "Turbidity (NTU)": 10, "Storage Time (hrs)": 2, "Shelf Life (hrs)": 48}
    
    # Convert data to DataFrame to maintain column order for scaler
    import pandas as pd
    input_df = pd.DataFrame([data])
    
    # Ensure the columns are in the same order as X_train used for fitting the scaler and model
    # 'Sample' and 'Status' were dropped, so use the columns from the original X DataFrame
    expected_columns = ['pH', 'Temp (°C)', 'Gas (ppm)', 'Turbidity (NTU)', 'Storage Time (hrs)', 'Shelf Life (hrs)']
    
    # Check if all expected columns are present in the input_df
    if not all(col in input_df.columns for col in expected_columns):
        return jsonify({"error": "Missing one or more required features."}), 400
        
    # Reorder input_df columns to match the training data's feature order
    input_df = input_df[expected_columns]

    # Scale the input features
    scaled_features = loaded_scaler_retrained.transform(input_df)

    # Make prediction
    prediction_encoded = loaded_model_retrained.predict(scaled_features)

    # Decode the prediction
    prediction_label = loaded_label_encoder_retrained.inverse_transform(prediction_encoded)[0]

    return jsonify({"prediction": prediction_label})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
