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
    # Eliminar archivo corrupto si existe y es muy pequeño
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 10000:  # Menos de 10KB = corrupto
        os.remove(MODEL_PATH)
        st.info("🗑️ Archivo corrupto eliminado, descargando nuevo...")
    
    if not os.path.exists(MODEL_PATH):
        try:
            st.info("📥 Descargando modelo desde Google Drive...")
            
            # URL de Google Drive (asegúrate de que el archivo sea público)
            file_id = "1xlzVWU680kSKIpJGl6i0mgTdct4QE_La"
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # Descargar usando gdown con force download
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            
            # Verificar descarga
            if os.path.exists(MODEL_PATH):
                file_size = os.path.getsize(MODEL_PATH)
                st.info(f"📊 Tamaño del archivo descargado: {file_size} bytes")
                
                if file_size > 100000:  # Más de 100KB = probablemente válido
                    st.success("✅ Modelo descargado exitosamente")
                    return True
                else:
                    st.error(f"❌ Archivo demasiado pequeño ({file_size} bytes), probablemente no es un modelo válido")
                    return False
            else:
                st.error("❌ No se pudo descargar el modelo")
                return False
                
        except Exception as e:
            st.error(f"❌ Error descargando el modelo: {e}")
            return False
    else:
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 100000:
            st.success(f"✅ Modelo ya existe ({file_size} bytes)")
            return True
        else:
            st.warning(f"⚠️ Archivo existente muy pequeño ({file_size} bytes), reintentando descarga...")
            os.remove(MODEL_PATH)
            return download_model()

# Descargar modelo
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
    "SPECIAL_MedicalTakeBack",
    "SPECIAL_HHW"
]

IMG_SIZE = (380, 380)

# --- CARGA DE MODELO ---
@st.cache_resource
def load_model():
    if not download_success:
        st.error("No se pudo descargar el modelo correctamente")
        return None
        
    try:
        # Verificar que el archivo existe y tiene tamaño adecuado
        if not os.path.exists(MODEL_PATH):
            st.error(f"Archivo {MODEL_PATH} no encontrado")
            return None
            
        file_size = os.path.getsize(MODEL_PATH)
        st.info(f"📊 Verificando modelo: {file_size} bytes")
        
        if file_size < 100000:  # Menos de 100KB = no es un modelo válido
            st.error(f"❌ El archivo del modelo es demasiado pequeño ({file_size} bytes). Debe ser > 100MB aproximadamente.")
            return None
            
        # Intentar cargar el modelo
        with st.spinner("🔄 Cargando modelo en memoria..."):
            model = tf.keras.models.load_model(MODEL_PATH)
            
        st.success("✅ Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        # Mostrar más detalles del error
        st.info("💡 Posibles soluciones:")
        st.info("1. Verifica que el archivo en Google Drive sea público")
        st.info("2. Asegúrate de que sea un modelo .keras válido")
        st.info("3. Revisa que las versiones de TensorFlow coincidan")
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
st.write("Sube una imagen y el modelo te dirá a qué categoría pertenece.")

# Estado de la aplicación
st.sidebar.header("Estado del Sistema")
if model is not None:
    st.sidebar.success("✅ Modelo listo")
else:
    st.sidebar.error("❌ Modelo no disponible")

if model is None:
    st.error("⚠️ El modelo no está disponible para clasificación.")
    
    # Información de diagnóstico
    with st.expander("🔧 Diagnóstico y Soluciones"):
        st.write("**Problema detectado:** El modelo no se descargó/cargó correctamente")
        st.write("**Soluciones:**")
        st.write("1. **Verificar permisos de Google Drive**: Asegúrate de que el archivo sea público")
        st.write("2. **Tamaño esperado**: Un modelo EfficientNetB4 debería tener >100MB")
        st.write("3. **Formato del modelo**: Verifica que sea un archivo .keras válido")
        
        if st.button("🔄 Reintentar Carga Completa"):
            # Limpiar cache y reintentar
            st.cache_resource.clear()
            st.rerun()

uploaded_file = st.file_uploader("Sube una imagen/Upload an image", type=["jpg","jpeg","png","webp"])

if uploaded_file and model is not None:
    img_array, img_disp = preprocess_image(uploaded_file)
    st.image(img_disp, caption="Imagen subida/Image uploaded", use_column_width=True)

    with st.spinner("Clasificando..."):
        pred_class, conf = predict(img_array)

    st.success(f"✅ Predicción: **{pred_class}** ({conf*100:.2f}%)")
