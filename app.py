import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ================================
# Load Model & Vectorizer
# ================================
model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("🛋️ Furniture Sales Prediction App")
st.write("Enter product details to predict number of items sold.")

# ================================
# User Inputs
# ================================
product_title = st.text_input("Product Title", "Modern Sofa Set")
original_price = st.number_input("Original Price ($)", min_value=0.0, step=100.0, value=1000.0)
price = st.number_input("Selling Price ($)", min_value=0.0, step=100.0, value=800.0)
tag = st.selectbox("Tag", ["Free shipping", "Others"])

# ================================
# Preprocess Inputs
# ================================
if st.button("Predict"):
    # Encode tag
    tag_encoded = 1 if tag == "Free shipping" else 0

    # Discount %
    discount_percentage = ((original_price - price) / original_price * 100) if original_price > 0 else 0

    # Revenue
    revenue = price * 1  # assume 1 unit for feature consistency

    # TF-IDF for product title
    title_tfidf = tfidf.transform([product_title]).toarray()
    tfidf_features = pd.DataFrame(title_tfidf, columns=[f"title_tfidf_{i}" for i in range(title_tfidf.shape[1])])

    # Create feature row (must match training features)
    features = pd.DataFrame({
        "originalPrice": [original_price],
        "price": [price],
        "tagText_encoded": [tag_encoded],
        "discount_percentage": [discount_percentage],
        "revenue": [revenue],
    })

    # Combine numeric + tfidf
    features = pd.concat([features.reset_index(drop=True), tfidf_features], axis=1)

    # ================================
    # Make Prediction
    # ================================
    prediction = model.predict(features)[0]

    # Clamp negatives to 0
    prediction = max(0, int(round(prediction)))

    # ================================
    # Show Results
    # ================================
    if price > original_price:
        st.warning("⚠️ Note: Selling Price is higher than Original Price. Prediction may be unreliable.")

    st.success(f"✅ Predicted number of items sold: {prediction}")

    # Explanation message
    if prediction == 0:
        st.info("This product may not sell at this price (low demand predicted).")
    elif prediction <= 10:
        st.info("This product may sell a moderate amount.")
    else:
        st.info("This product is expected to sell well (high demand predicted).")
