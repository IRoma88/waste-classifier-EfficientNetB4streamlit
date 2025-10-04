import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
import requests

MODEL_PATH = "EfficientNetB4_finetuned.keras"

# --- DESCARGAR MODELO ---
@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        try:
            st.info("📥 Descargando modelo... Esto puede tomar unos minutos.")
            url = "https://drive.google.com/uc?id=1xlzVWU680kSKIpJGl6i0mgTdct4QE_La"
            
            # Método alternativo si gdown falla
            gdown.download(url, MODEL_PATH, quiet=False)
            
            # Verificar que el archivo se descargó correctamente
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
                st.success("✅ Modelo descargado exitosamente")
                return True
            else:
                st.error("❌ El archivo del modelo está vacío o no se descargó correctamente")
                return False
                
        except Exception as e:
            st.error(f"❌ Error descargando el modelo: {e}")
            return False
    else:
        st.success("✅ Modelo ya existe localmente")
        return True

# Descargar modelo al inicio
download_success = download_model()

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

IMG_SIZE = (380, 380)

# --- CARGA DE MODELO ---
@st.cache_resource
def load_model():
    if not download_success:
        st.error("No se pudo descargar el modelo")
        return None
        
    try:
        # Verificar que el archivo existe y tiene tamaño
        if not os.path.exists(MODEL_PATH):
            st.error(f"Archivo {MODEL_PATH} no encontrado")
            return None
            
        file_size = os.path.getsize(MODEL_PATH)
        st.info(f"Tamaño del archivo del modelo: {file_size} bytes")
        
        if file_size == 0:
            st.error("El archivo del modelo está vacío")
            return None
            
        # Intentar cargar el modelo
        with st.spinner("🔄 Cargando modelo..."):
            model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False
            )
            
        # Compilar el modelo si es necesario
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        st.success("✅ Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        return None

model = load_model()

# --- FUNCIONES ---
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0
    return x, img

def predict(img_array):
    if model is None:
        return "Modelo no disponible", 0.0
    
    try:
        preds = model.predict(img_array)
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        return CLASS_NAMES[class_id], confidence
    except Exception as e:
        return f"Error en predicción: {str(e)}", 0.0

# --- INTERFAZ ---
st.title("♻️ Clasificador de Residuos/Waste Classifier - EfficientNetB4")
st.write("Sube una imagen y el modelo te dirá a qué categoría pertenece./Upload an image and the model will tell you what category it belongs to")

# Mostrar estado del modelo
if model is None:
    st.error("⚠️ El modelo no se pudo cargar. Por favor, revisa los logs.")
    
    # Botón para reintentar descarga
    if st.button("Reintentar descarga del modelo"):
        st.cache_resource.clear()
        st.experimental_rerun()
else:
    st.success("✅ Modelo listo para clasificar")

uploaded_file = st.file_uploader("Sube una imagen/Upload an image", type=["jpg","jpeg","png","webp"])

if uploaded_file and model is not None:
    img_array, img_disp = preprocess_image(uploaded_file)
    st.image(img_disp, caption="Imagen subida/Image uploaded", use_column_width=True)

    with st.spinner("Clasificando/Classifing..."):
        pred_class, conf = predict(img_array)

    st.success(f"✅ Predicción/Prediction: **{pred_class}** ({conf*100:.2f}%)")
