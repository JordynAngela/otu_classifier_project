import streamlit as st
import pandas as pd
from otu_classifier import (
    create_labels,
    prepare_data,
    train_and_evaluate,
    plot_feature_importance
)
from PIL import Image

st.title("🧠 OTU Classifier Web App 2.0-- The do it yourself model")
st.write("This app classifies samples (e.g. healthy vs. sick) using microbiome OTU data.")

# Upload or use default demo
uploaded_file = st.file_uploader("You can upload your own OTU CSV file (if you want...)", type=["csv"])

if uploaded_file is not None:
    try:
        otu_table = pd.read_csv(uploaded_file, index_col=0)
        st.success(" You did it! Uploaded file loaded!")
    except Exception as e:
        st.error(f"HEY! Error loading uploaded file: {e}")
        st.stop()
else:
    st.info("ℹ️o file uploaded. Using built-in demo data.")
    otu_table = pd.read_csv("otu_table.csv", index_col=0)

# Show table preview
st.subheader("The OTU Table Preview")
st.dataframe(otu_table.head())

# Generate labels, train model, display accuracy and importance
labels = create_labels(otu_table.columns)
(data_split, feature_names) = prepare_data(otu_table, labels)
X_train, X_test, y_train, y_test = data_split

model, accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)
st.success(f"Model Accuracy: {accuracy:.2f}")

plot_feature_importance(model, feature_names)
st.image(Image.open("otu_importance.png"), caption="Top 10 Important OTUs")

