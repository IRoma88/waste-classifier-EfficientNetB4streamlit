import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
import requests
from PIL import Image

MODEL_PATH = "EfficientNetB4_finetuned.keras"

# --- DESCARGAR MODELO ---
@st.cache_resource
def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) < 10000:
        os.remove(MODEL_PATH)
    
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Descargando modelo... Esto puede tomar unos minutos para un modelo grande.")
        
        try:
            file_id = "1xlzVWU680kSKIpJGl6i0mgTdct4QE_La"
            
            st.write("🔗 Usando enlace público de Google Drive...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)
            
            if os.path.exists(MODEL_PATH):
                file_size = os.path.getsize(MODEL_PATH)
                st.write(f"📊 Tamaño descargado: {file_size} bytes")
                
                if file_size > 100000:
                    st.success("✅ Modelo descargado exitosamente!")
                    return True
                else:
                    return False
            else:
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
            os.remove(MODEL_PATH)
            return download_model()

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

# --- CARGA DE MODELO SIMPLIFICADA ---
@st.cache_resource
def load_model():
    if not download_success:
        return None
        
    try:
        file_size = os.path.getsize(MODEL_PATH)
        st.info(f"📊 Cargando modelo: {file_size} bytes")
        
        if file_size < 100000:
            return None
            
        with st.spinner("🔄 Cargando modelo..."):
            # Carga simple sin pruebas adicionales
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            
            # Compilar si es necesario
            model.compile(optimizer='adam', 
                         loss='categorical_crossentropy', 
                         metrics=['accuracy'])
            
        st.success("✅ ¡Modelo cargado exitosamente!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {str(e)}")
        
        # Mostrar solución específica
        st.markdown("""
        ### 🔧 Solución Requerida:
        
        El modelo fue entrenado con una configuración diferente. Necesitas:
        
        1. **Volver a Google Colab**
        2. **Guardar el modelo con esta configuración:**
        ```python
        # Guardar con formato compatible
        model.save('EfficientNetB4_compatible.keras', save_format='keras')
        
        # O mejor aún, guardar como .h5
        model.save('EfficientNetB4_compatible.h5')
        ```
        3. **Subir el nuevo modelo a Google Drive**
        4. **Actualizar el enlace en esta aplicación**
        """)
        return None

model = load_model()

# --- FUNCIONES MEJORADAS ---
def preprocess_image(uploaded_file):
    """Preprocesa la imagen asegurando que tenga 3 canales (RGB) y tamaño correcto"""
    try:
        # Abrir la imagen y convertir a RGB (asegura 3 canales)
        img = Image.open(uploaded_file)
        
        # Convertir a RGB si es escala de grises o RGBA
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar al tamaño que espera el modelo (380x380 para EfficientNetB4)
        img = img.resize(IMG_SIZE)
        
        # Convertir a array y normalizar
        img_array = np.array(img) / 255.0
        
        # Verificar la forma
        st.write(f"🔍 Imagen procesada: {img_array.shape}")
        
        # Añadir dimensión del batch
        img_array = np.expand_dims(img_array, axis=0)
        
        st.write(f"🔍 Forma final para el modelo: {img_array.shape}")
        
        return img_array, img
        
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {e}")
        return None, None

def predict(img_array):
    if model is None:
        return "Modelo no disponible", 0.0
    
    try:
        # Verificar la forma de la imagen
        st.write(f"🔍 Forma de entrada al modelo: {img_array.shape}")
        
        # Asegurarse de que la imagen tenga la forma correcta: (1, 380, 380, 3)
        if img_array.shape != (1, 380, 380, 3):
            st.error(f"❌ Forma incorrecta: {img_array.shape}. Debe ser (1, 380, 380, 3)")
            # Intentar corregir la forma
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)
            if img_array.shape[1:3] != (380, 380):
                st.error("❌ Tamaño incorrecto. No se puede corregir automáticamente.")
                return "Error: Tamaño de imagen incorrecto", 0.0
            if img_array.shape[-1] != 3:
                st.error("❌ Número de canales incorrecto. No se puede corregir automáticamente.")
                return "Error: Imagen debe tener 3 canales RGB", 0.0
        
        with st.spinner("🔍 Analizando imagen..."):
            preds = model.predict(img_array, verbose=0)
        
        st.write(f"🔍 Forma de las predicciones: {preds.shape}")
        
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        return CLASS_NAMES[class_id], confidence
        
    except Exception as e:
        return f"Error en predicción: {str(e)}", 0.0

# --- INTERFAZ ---
st.title("♻️ Clasificador de Residuos - EfficientNetB4")
st.write("Sube una imagen de un residuo y el modelo te dirá en qué categoría clasificarlo")

# Estado del sistema
col1, col2, col3 = st.columns(3)
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

with col3:
    st.info(f"📐 Tamaño: {IMG_SIZE[0]}x{IMG_SIZE[1]}")

# Solo mostrar el uploader si el modelo está cargado
if model is not None:
    uploaded_file = st.file_uploader(
        "Sube una imagen de residuo", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Sube una imagen RGB (color) de 380x380 píxeles para mejor clasificación"
    )
    
    if uploaded_file is not None:
        # Preprocesar y mostrar imagen
        img_array, img_disp = preprocess_image(uploaded_file)
        
        if img_array is not None:
            st.image(img_disp, caption="📷 Imagen subida (convertida a RGB)", use_column_width=True)
            
            # Realizar predicción
            pred_class, conf = predict(img_array)
            
            if "Error" not in pred_class:
                # Mostrar resultados
                st.success(f"✅ **Categoría:** {pred_class}")
                
                # Barra de confianza
                st.progress(conf)
                st.write(f"**Confianza:** {conf*100:.2f}%")
                
                # Información adicional sobre la categoría
                st.markdown("---")
                if "BlueRecyclable" in pred_class:
                    st.info("🔵 **Contenedor Azul - Reciclable**")
                    st.write("Materiales como papel, cartón, vidrio, metales y plásticos")
                elif "BrownCompost" in pred_class:
                    st.info("🟤 **Contenedor Marrón - Orgánico**")
                    st.write("Restos de comida, frutas, verduras, poda del jardín")
                elif "GrayTrash" in pred_class:
                    st.info("⚪ **Contenedor Gris - Resto**")
                    st.write("Materiales no reciclables ni compostables")
                elif "SPECIAL" in pred_class:
                    st.warning("🟡 **Categoría Especial**")
                    st.write("Sigue las instrucciones específicas de tu municipio")
            else:
                st.error(f"❌ {pred_class}")

else:
    st.error("❌ El modelo no se pudo cargar correctamente")
    
    st.markdown("""
    ### 🚨 Solución Definitiva:
    
    **Necesitas volver a Google Colab y guardar el modelo de forma compatible:**
    
    ```python
    # OPCIÓN 1: Guardar como .h5 (más compatible)
    model.save('EfficientNetB4_compatible.h5')
    
    # OPCIÓN 2: Guardar con formato específico
    tf.keras.models.save_model(
        model,
        'EfficientNetB4_compatible.keras',
        save_format='keras'
    )
    
    # OPCIÓN 3: Instalar misma versión de TensorFlow en Colab
    !pip install tensorflow==2.13.0
    # Luego entrenar y guardar el modelo
    ```
    
    **Luego:**
    1. Sube el nuevo modelo a Google Drive
    2. Actualiza el file_id en este código
    3. Recarga la aplicación en Streamlit
    """)

# Footer
st.markdown("---")
st.caption(f"TensorFlow {tf.__version__} | Streamlit {st.__version__}")
