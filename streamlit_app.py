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
    # Eliminar archivo corrupto si existe
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 10000:
        os.remove(MODEL_PATH)
    
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Descargando modelo... Esto puede tomar unos minutos para un modelo grande.")
        
        try:
            # Tu enlace público de Google Drive
            file_id = "1xlzVWU680kSKIpJGl6i0mgTdct4QE_La"
            
            # URL para descarga directa (formato correcto)
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            st.write("🔗 Usando enlace público de Google Drive...")
            
            # Método 1: Gdown con el file_id
            gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)
            
            # Verificar si la descarga fue exitosa
            if os.path.exists(MODEL_PATH):
                file_size = os.path.getsize(MODEL_PATH)
                st.write(f"📊 Tamaño descargado: {file_size} bytes")
                
                if file_size > 100000:  # Más de 100KB = probablemente válido
                    st.success("✅ Modelo descargado exitosamente!")
                    return True
                else:
                    st.warning("⚠️ Archivo muy pequeño, intentando método alternativo...")
                    os.remove(MODEL_PATH)
                    
                    # Método alternativo: requests
                    response = requests.get(download_url, stream=True)
                    total_size = int(response.headers.get('content-length', 0))
                    
                    if total_size > 100000:
                        with open(MODEL_PATH, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        if os.path.getsize(MODEL_PATH) > 100000:
                            st.success("✅ Modelo descargado con método alternativo!")
                            return True
                    
                    st.error("❌ El archivo descargado es demasiado pequeño")
                    return False
            else:
                st.error("❌ No se pudo crear el archivo del modelo")
                return False
                
        except Exception as e:
            st.error(f"❌ Error en la descarga: {e}")
            return False
    else:
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 100000:
            st.success(f"✅ Modelo ya disponible ({file_size} bytes)")
            return True
        else:
            st.warning("⚠️ Archivo existente corrupto, reintentando descarga...")
            os.remove(MODEL_PATH)
            return download_model()

# Ejecutar descarga
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
        st.info(f"📊 Cargando modelo: {file_size} bytes")
        
        if file_size < 100000:
            st.error(f"❌ El archivo es demasiado pequeño para ser un modelo ({file_size} bytes)")
            return None
            
        # Mostrar progreso de carga
        with st.spinner("🔄 Cargando modelo en memoria (esto puede tomar unos segundos)..."):
            model = tf.keras.models.load_model(MODEL_PATH)
            
        st.success("✅ ¡Modelo cargado exitosamente!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        
        # Información de diagnóstico
        with st.expander("🔍 Detalles técnicos del error"):
            st.write(f"**Tipo de error:** {type(e).__name__}")
            st.write(f"**Mensaje:** {str(e)}")
            st.write("**Posibles causas:**")
            st.write("- Versiones incompatibles de TensorFlow")
            st.write("- Archivo de modelo corrupto")
            st.write("- Formato de archivo incorrecto")
        
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
        with st.spinner("🔍 Analizando imagen..."):
            preds = model.predict(img_array)
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        return CLASS_NAMES[class_id], confidence
    except Exception as e:
        return f"Error en predicción: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL ---
st.title("♻️ Clasificador de Residuos - EfficientNetB4")
st.write("Sube una imagen de un residuo y el modelo te dirá en qué categoría clasificarlo")

# Estado del sistema
col1, col2 = st.columns(2)
with col1:
    if download_success:
        st.success("📥 Descarga: OK")
    else:
        st.error("📥 Descarga: Falló")
        
with col2:
    if model is not None:
        st.success("🧠 Modelo: Cargado")
    else:
        st.error("🧠 Modelo: No disponible")

# Solo mostrar el uploader si el modelo está cargado
if model is not None:
    uploaded_file = st.file_uploader(
        "Sube una imagen de residuo", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Sube una imagen clara del residuo que quieres clasificar"
    )
    
    if uploaded_file is not None:
        # Preprocesar y mostrar imagen
        img_array, img_disp = preprocess_image(uploaded_file)
        st.image(img_disp, caption="📷 Imagen subida", use_column_width=True)
        
        # Realizar predicción
        pred_class, conf = predict(img_array)
        
        # Mostrar resultados
        st.success(f"✅ **Categoría:** {pred_class}")
        
        # Barra de confianza
        st.progress(conf)
        st.write(f"**Confianza:** {conf*100:.2f}%")
        
        # Información adicional sobre la categoría
        if "BlueRecyclable" in pred_class:
            st.info("🔵 **Contenedor Azul - Reciclable**")
        elif "BrownCompost" in pred_class:
            st.info("🟤 **Contenedor Marrón - Orgánico**")
        elif "GrayTrash" in pred_class:
            st.info("⚪ **Contenedor Gris - Resto**")
        elif "SPECIAL" in pred_class:
            st.warning("🟡 **Categoría Especial - Sigue instrucciones específicas**")

else:
    st.error("""
    ❌ **El sistema no está listo todavía**
    
    Si el problema persiste:
    1. Verifica que el archivo en Google Drive sea mayor a 100MB
    2. Espera unos minutos y recarga la aplicación
    3. Contacta al administrador si necesitas ayuda
    """)

# Footer
st.markdown("---")
st.caption("Clasificador de Residuos usando EfficientNetB4 - Versión Streamlit")
