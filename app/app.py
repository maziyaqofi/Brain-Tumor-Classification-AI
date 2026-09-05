import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from pathlib import Path


# ============================================
# CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

IMAGE_SIZE = (224, 224)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "efficientnetb0_finetuned.h5"
)


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ============================================
# PREPROCESS IMAGE
# ============================================

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image).astype(np.float32)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================
# PREDICTION
# ============================================

def predict_image(model, image):

    processed_image = preprocess_image(image)

    predictions = model.predict(
        processed_image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = probabilities[predicted_index]

    return (
        predicted_class,
        confidence,
        probabilities,
        predicted_index,
        processed_image
    )


# ============================================
# GRAD-CAM
# ============================================

def make_gradcam_heatmap(
    image_array,
    model,
    predicted_index
):

    # Get EfficientNetB0 backbone
    backbone = model.get_layer("efficientnetb0")

    # Target convolutional layer
    last_conv_layer = backbone.get_layer("top_conv")

    # Model that returns:
    # 1. convolutional feature maps
    # 2. backbone output
    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, backbone_output = grad_model(
            image_array
        )

        # Outer classifier layers
        gap = model.get_layer(
            "global_average_pooling"
        )

        dropout = model.get_layer(
            "dropout"
        )

        classifier = model.get_layer(
            "classifier"
        )

        pooled_output = gap(backbone_output)

        dropout_output = dropout(
            pooled_output,
            training=False
        )

        predictions = classifier(
            dropout_output
        )

        class_channel = predictions[:, predicted_index]

    # Gradient of predicted class
    # with respect to convolutional feature maps
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # Average gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # Weighted feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )

    return heatmap.numpy()


# ============================================
# CREATE GRAD-CAM OVERLAY
# ============================================

def create_gradcam_overlay(
    image,
    heatmap
):

    # Convert image to RGB
    image_rgb = image.convert("RGB")

    image_array = np.array(
        image_rgb
    )

    # Resize heatmap
    heatmap_resized = cv2.resize(
        heatmap,
        (image_array.shape[1], image_array.shape[0])
    )

    # Convert to 0-255
    heatmap_uint8 = np.uint8(
        255 * heatmap_resized
    )

    # Apply colormap
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # OpenCV uses BGR
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    overlay = cv2.addWeighted(
        image_array,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return overlay


# ============================================
# MAIN UI
# ============================================

st.title(
    "🧠 Brain Tumor MRI Classification"
)

st.markdown(
    """
Upload a brain MRI image and let the fine-tuned
EfficientNetB0 model classify it into one of four
categories.
"""
)

st.warning(
    """
**Educational and Research Purpose Only**

This application is not a medical diagnostic tool
and should not be used to make medical decisions.
"""
)


# ============================================
# LOAD MODEL
# ============================================

try:

    model = load_model()

except Exception as e:

    st.error(
        "Failed to load the model."
    )

    st.exception(e)

    st.stop()


# ============================================
# FILE UPLOAD
# ============================================

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    )

    st.subheader(
        "Uploaded MRI"
    )

    st.image(
        image,
        caption="Uploaded MRI Image",
        width="stretch"
    )

    st.divider()

    # ========================================
    # CLASSIFY
    # ========================================

    if st.button(
        "🔍 Classify MRI",
        type="primary",
        width="stretch"
    ):

        with st.spinner(
            "Analyzing MRI image..."
        ):

            (
                predicted_class,
                confidence,
                probabilities,
                predicted_index,
                processed_image
            ) = predict_image(
                model,
                image
            )


        # ====================================
        # PREDICTION RESULT
        # ====================================

        st.subheader(
            "Prediction Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Class",
                predicted_class.upper()
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )


        # ====================================
        # PROBABILITIES
        # ====================================

        st.subheader(
            "Class Probabilities"
        )

        probability_data = {

            CLASS_NAMES[i].capitalize():
            float(probabilities[i])

            for i in range(
                len(CLASS_NAMES)
            )
        }

        st.bar_chart(
            probability_data,
            horizontal=True
        )


        # ====================================
        # GRAD-CAM
        # ====================================

        st.divider()

        st.subheader(
            "Grad-CAM Visualization"
        )

        st.caption(
            "The visualization highlights image regions "
            "that contributed to the model's prediction."
        )

        try:

            heatmap = make_gradcam_heatmap(
                processed_image,
                model,
                predicted_index
            )

            overlay = create_gradcam_overlay(
                image,
                heatmap
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Original MRI",
                    width="stretch"
                )

            with col2:

                st.image(
                    overlay,
                    caption="Grad-CAM Overlay",
                    width="stretch"
                )

        except Exception as e:

            st.warning(
                "Grad-CAM could not be generated."
            )

            st.exception(e)


# ============================================
# FOOTER
# ============================================

st.divider()

st.caption(
    "Brain Tumor MRI Classification | "
    "EfficientNetB0 | Educational Project"
)