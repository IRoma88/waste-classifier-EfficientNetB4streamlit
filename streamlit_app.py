import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

MODEL_PATH = "EfficientNetB4_finetuned.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1xlzVWU680kSKIpJGl6i0mgTdct4QE_La"
    gdown.download(url, MODEL_PATH, quiet=False)

# Ajusta según tus clases finales (en el mismo orden que usaste en entrenamiento)
CLASS_NAMES = [
    "BlueRecyclable_Cardboard",
    "BlueRecyclable_Glass",
    "BlueRecyclable_Metal",
    "BlueRecyclable_Paper",
    "BlueRecyclable_Plastics",
    "BrownCompost",
    "GrayTrash",
    "SPECIAL_DropOff",
    "SPECIAL_TakeBackShop",
    "SPECIAL_MedicalTakeOff",
    "SPECIAL_HHW"
]

IMG_SIZE = (380, 380)  # EfficientNetB4 prefiere 380x380

# --- CARGA DE MODELO ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# --- FUNCIONES ---
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    return x, img

def predict(img_array):
    preds = model.predict(img_array)
    class_id = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    return CLASS_NAMES[class_id], confidence

# --- INTERFAZ ---
st.title("♻️ Clasificador de Residuos/Waste Classifier - EfficientNetB4")
st.write("Sube una imagen y el modelo te dirá a qué categoría pertenece./Upload an image and the model will tell you what category it belongs to")

uploaded_file = st.file_uploader("Sube una imagen/Upload an image", type=["jpg","jpeg","png","webp"])

if uploaded_file:
    img_array, img_disp = preprocess_image(uploaded_file)
    st.image(img_disp, caption="Imagen subida/Image uploaded", use_column_width=True)

    with st.spinner("Clasificando/Classifing..."):
        pred_class, conf = predict(img_array)

    st.success(f"✅ Predicción/Prediction: **{pred_class}** ({conf*100:.2f}%)")
