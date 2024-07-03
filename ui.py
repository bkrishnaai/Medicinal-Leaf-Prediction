import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
import os
import uuid

# Define a custom layer ResidualUnit
class ResidualUnit(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(filters, kernel_size=3, strides=strides,
                                   padding="same", kernel_initializer="he_normal",
                                   use_bias=False),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1,
                                   padding="same", kernel_initializer="he_normal",
                                   use_bias=False),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(filters, kernel_size=1, strides=strides,
                                       padding="same", kernel_initializer="he_normal",
                                       use_bias=False),
                tf.keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)

# Load the model with custom objects scope
model_path = "D:/code plants/model.h5"
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model = tf.keras.models.load_model(model_path, custom_objects={'ResidualUnit': ResidualUnit}, compile=False)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Define class indices
class_indices = {
    'Aloevera': 0,
    'Amla': 1,
    'Bamboo': 2,
    'Beans': 3,
    'Betel': 4,
    'Chilly': 5,
    'Coffee': 6,
    'Coriender': 7,
    'Drumstick': 8,
    'Ganigale': 9,
    'Ginger': 10,
    'Guava': 11,
    'Henna': 12,
    'Hibiscus': 13,
    'Jasmine': 14,
    'Lemon': 15,
    'Mango': 16,
    'Marigold': 17,
    'Mint': 18,
    'Neem': 19,
    'Onion': 20,
    'Palak': 21,
    'Papaya': 22,
    'Parijatha': 23,
    'Pea': 24,
    'Pomoegranate': 25,
    'Pumpkin': 26,
    'Raddish': 27,
    'Rose': 28,
    'Sampige': 29,
    'Sapota': 30,
    'Seethapala': 31,
    'Spinach1': 32,
    'Tamarind': 33,
    'Tomato': 34,
    'Tulsi': 35,
    'Turmeric': 36,
    'ashoka': 37,
    'camphor': 38
}

# Load the description dataset
desc_path = "D:/code plants/Plants_Desc.csv"
df = pd.read_csv(desc_path, encoding='latin1')

# Streamlit UI
st.set_page_config(page_title="Medicinal Leaf Prediction", page_icon=":leaves:", layout="wide")

# Title
st.title('Medicinal Leaf Prediction')

# Sidebar
st.sidebar.title("About")
st.sidebar.write("Upload an image of a leaf and get the prediction of the medicinal plant family.")
st.sidebar.subheader("Feedback")

# Directory to store incorrect images
base_directory = "D:/Code/incorrect"

# Create base directory if it doesn't exist
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG/JPEG/PNG)", type=["jpg", "jpeg", "png"])

# Prediction and feedback
if uploaded_file is not None:
    # Display the selected image
    st.subheader("Uploaded Image")
    img = Image.open(uploaded_file)
    resized_img = img.resize((128, 128))  # Resize the image to match the model's input size
    st.image(resized_img, caption='Uploaded Image.', width=200)

    # Preprocess the image
    img_array = image.img_to_array(resized_img)
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = [key for key, value in class_indices.items() if value == predicted_class_index][0]

    # Display prediction
    st.subheader("Prediction")
    st.success(f"The uploaded leaf is {predicted_class_label} .")

    # Display description
    st.subheader("**Description**")
    description = df.loc[df['Type'].str.contains(predicted_class_label, case=False, na=False), 'Description'].values
    if len(description) > 0:
        st.markdown(f"**{description[0]}**")

    # Display uses
    st.subheader("**Uses**")
    uses = df.loc[df['Type'].str.contains(predicted_class_label, case=False, na=False), 'Uses'].values
    if len(uses) > 0:
        st.markdown(f"**{uses[0]}**")

    # Ask for feedback
    feedback_options = ["Correct", "Incorrect"]
    feedback = st.radio("Is the prediction correct?", feedback_options)

    # If feedback is incorrect, ask for correct label and store image in respective directory
    if feedback == "Incorrect":
        # Ask for correct label
        st.subheader("Select Correct Label")
        remaining_classes = [key for key in class_indices.keys() if key != predicted_class_label]
        correct_label = st.selectbox("Select correct label:", remaining_classes)

        # Generate unique filename for the uploaded image
        filename = str(uuid.uuid4()) + ".png"

        # Save the uploaded image to the corresponding directory only when the correct label is selected
        if st.button("FeedBack"):
            class_directory = os.path.join(base_directory, correct_label)
            if not os.path.exists(class_directory):
                os.makedirs(class_directory)

            img_path = os.path.join(class_directory, filename)
            with open(img_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Show motivational animation
            st.balloons()
    else:
        st.sidebar.success("Thanks for your feedback!")
