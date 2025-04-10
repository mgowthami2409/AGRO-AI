import streamlit as st
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

# Load Models
npk_model = load_model(r"C:\Users\Admin\Downloads\Major_Project_New\best_biGRU_model.keras")
disease_model = load_model(r"C:\Users\Admin\Downloads\Major_Project_New\riceleaf_newmodel.h5")

# Constants
SEQUENCE_LENGTH = 6  # Number of images per sequence
FEATURE_DIM = 512  # Adjust this if necessary based on your dataset

# Function to Compute Vegetation Indices 
def compute_indices(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    VARI = (G - R) / (G + R - B)
    GLI = (2 * G - R - B) / (2 * G + R + B)
    ExG = 2 * G - R - B

    return np.array([VARI.mean(), GLI.mean(), ExG.mean()])

# Function to Predict NPK Values
def predict_npk(image):
    indices = compute_indices(image)
    resnet_features = np.random.rand(512)  # Replace with actual ResNet feature extraction
    features = np.concatenate([resnet_features, indices])
    features = np.tile(features, (SEQUENCE_LENGTH, 1))  # Repeat to match sequence length
    features = features.reshape(1, SEQUENCE_LENGTH, FEATURE_DIM + 3)
    npk_values = npk_model.predict(features)
    return npk_values.flatten()

# Function to Predict Disease
def predict_disease(image):
    image_resized = cv2.resize(image, (180, 180)) / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)
    disease_pred = disease_model.predict(image_resized)
    disease_label = np.argmax(disease_pred)
    
    disease_dict = {
        0: "Bacterial Leaf Blight",
        1: "Brown Spot",
        2: "Smut",
        3: "Blast",
        4: "Tungro"
    }
    
    # Set a threshold for confidence
    confidence_threshold = 0.5  # Adjust this value as needed
    if np.max(disease_pred) < confidence_threshold:
        return "Healthy"
    
    return disease_dict[disease_label]

# Function to Suggest Fertilizers
def suggest_fertilizer(n, p, k):
    recommendations = {
        "Nitrogen": {
            "Chemical": "Urea (46% N) or Ammonium Nitrate",
            "Organic": "Compost, Vermicompost, or Cow Manure"
        },
        "Phosphorus": {
            "Chemical": "Single Super Phosphate (SSP) or DAP",
            "Organic": "Bone Meal, Rock Phosphate"
        },
        "Potassium": {
            "Chemical": "Muriate of Potash (MOP) or Sulfate of Potash",
            "Organic": "Wood Ash, Banana Peel Fertilizer"
        }
    }

    suggestions = {}

    if n < 20:
        suggestions["Nitrogen"] = recommendations["Nitrogen"]
    if p < 30:
        suggestions["Phosphorus"] = recommendations["Phosphorus"]
    if k < 40:
        suggestions["Potassium"] = recommendations["Potassium"]

    return suggestions

# Streamlit UI
st.title("Paddy Leaf Analysis for NPK Prediction & Disease Detection")

uploaded_file = st.file_uploader("Upload a Paddy Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        # Predictions
        n_pred, p_pred, k_pred = predict_npk(image)
        disease = predict_disease(image)
        recommendations = suggest_fertilizer(n_pred, p_pred, k_pred)

        # Display Results
        st.subheader("Predicted NPK Values")
        st.write(f"**Nitrogen (N):** {n_pred:.2f}")
        st.write(f"**Phosphorus (P):** {p_pred:.2f}")
        st.write(f"**Potassium (K):** {k_pred:.2f}")

        st.subheader("Fertilizer Recommendations")
        for nutrient, advice in recommendations.items():
            st.write(f"{nutrient}:")
            st.write(f"  🔹 Chemical Fertilizer: {advice['Chemical']}")
            st.write(f"  🔹 Organic Fertilizer: {advice['Organic']}")

        st.subheader("Disease Detection Result")
        st.write(f"**Detected Disease:** {disease}")

        # Save to CSV
        df = pd.DataFrame([{
            "N": n_pred, "P": p_pred, "K": k_pred,
            "Nitrogen_Chemical": recommendations.get("Nitrogen", {}).get("Chemical", ""),
            "Nitrogen_Organic": recommendations.get("Nitrogen", {}).get("Organic", ""),
            "Phosphorus_Chemical": recommendations.get("Phosphorus", {}).get("Chemical", ""),
            "Phosphorus_Organic": recommendations.get("Phosphorus", {}).get("Organic", ""),
            "Potassium_Chemical": recommendations.get("Potassium", {}).get("Chemical", ""),
            "Potassium_Organic": recommendations.get("Potassium", {}).get("Organic", "")
        }])

        df.to_csv("fertilizer_recommendations.csv", mode='a', index=False, header=False)
        