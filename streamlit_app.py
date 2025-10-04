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
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
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

# --- CARGA DE MODELO CON COMPATIBILIDAD ---
@st.cache_resource
def load_model():
    if not download_success:
        return None
        
    try:
        file_size = os.path.getsize(MODEL_PATH)
        st.info(f"📊 Cargando modelo: {file_size} bytes")
        
        if file_size < 100000:
            return None
            
        with st.spinner("🔄 Cargando modelo (puede tomar unos segundos)..."):
            # Intentar diferentes métodos de carga para compatibilidad
            try:
                # Método 1: Carga normal
                model = tf.keras.models.load_model(MODEL_PATH)
                st.success("✅ ¡Modelo cargado con método estándar!")
            except Exception as e1:
                st.warning("⚠️ Método estándar falló, intentando con custom_objects...")
                try:
                    # Método 2: Con compatibilidad
                    model = tf.keras.models.load_model(
                        MODEL_PATH,
                        compile=False,
                        custom_objects={
                            'Functional': tf.keras.Model,
                            'Adam': tf.keras.optimizers.Adam
                        }
                    )
                    # Compilar el modelo si es necesario
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    st.success("✅ ¡Modelo cargado con compatibilidad!")
                except Exception as e2:
                    st.warning("⚠️ Segundo método falló, intentando carga básica...")
                    try:
                        # Método 3: Solo cargar pesos o arquitectura básica
                        model = tf.keras.models.load_model(
                            MODEL_PATH,
                            compile=False
                        )
                        st.success("✅ ¡Modelo cargado en modo básico!")
                    except Exception as e3:
                        st.error(f"❌ Todos los métodos fallaron:")
                        st.error(f"Error 1: {str(e1)}")
                        st.error(f"Error 2: {str(e2)}")
                        st.error(f"Error 3: {str(e3)}")
                        return None
        
        # Verificar que el modelo se cargó correctamente
        if model is not None:
            st.success(f"✅ Modelo cargado: {type(model)}")
            # Probar una predicción simple para verificar
            try:
                test_input = np.random.random((1, 380, 380, 3)).astype(np.float32)
                test_pred = model.predict(test_input)
                st.success(f"✅ Modelo verificado: Output shape {test_pred.shape}")
            except Exception as e:
                st.warning(f"⚠️ Modelo cargado pero falló en prueba: {e}")
            
            return model
        else:
            return None
        
    except Exception as e:
        st.error(f"❌ Error general cargando el modelo: {str(e)}")
        return None

model = load_model()

# --- FUNCIONES ---
def preprocess_image(uploaded_file):
    """Preprocesa la imagen asegurando que tenga 3 canales (RGB)"""
    try:
        # Abrir la imagen y convertir a RGB (asegura 3 canales)
        img = Image.open(uploaded_file)
        
        # Convertir a RGB si es escala de grises o RGBA
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img = img.resize(IMG_SIZE)
        
        # Convertir a array y normalizar
        img_array = np.array(img) / 255.0
        
        # Añadir dimensión del batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
        
    except Exception as e:
        st.error(f"Error procesando imagen: {e}")
        return None, None

def predict(img_array):
    if model is None:
        return "Modelo no disponible", 0.0
    
    try:
        # Verificar la forma de la imagen
        st.write(f"🔍 Forma de la imagen de entrada: {img_array.shape}")
        
        # Asegurarse de que la imagen tenga 3 canales
        if img_array.shape[-1] != 3:
            st.error(f"❌ La imagen tiene {img_array.shape[-1]} canales, pero se necesitan 3 canales RGB")
            return "Error: Imagen debe ser RGB", 0.0
        
        with st.spinner("🔍 Analizando imagen..."):
            preds = model.predict(img_array)
        
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

# Información para el usuario
st.info("""
💡 **Recomendaciones para mejores resultados:**
- Usa imágenes con buena iluminación
- Enfoca claramente el objeto
- Toma la foto sobre fondo neutro
- Evita imágenes borrosas o oscuras
""")

# Solo mostrar el uploader si el modelo está cargado
if model is not None:
    uploaded_file = st.file_uploader(
        "Sube una imagen de residuo", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Sube una imagen RGB (color) para mejor clasificación"
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
                elif "BrownCompost" in pred_class:
                    st.info("🟤 **Contenedor Marrón - Orgánico**")
                elif "GrayTrash" in pred_class:
                    st.info("⚪ **Contenedor Gris - Resto**")
                elif "SPECIAL" in pred_class:
                    st.warning("🟡 **Categoría Especial**")
            else:
                st.error(pred_class)

else:
    st.error("""
    ❌ **El sistema no está listo todavía**
    
    **Posibles soluciones:**
    1. **Recarga la página** (F5 o Ctrl+R)
    2. **Verifica el requirements.txt** en tu repositorio
    3. **El modelo puede necesitar ser reentrenado** con una versión compatible
    
    **Para futuros entrenamientos en Colab:**
    ```python
    # Al final de tu notebook, guarda el modelo así:
    model.save('modelo.h5')  # Formato .h5 es más compatible
    # O especifica la versión de TensorFlow
    !pip install tensorflow==2.13.0
    ```
    """)

# Información de versión
st.sidebar.markdown("---")
st.sidebar.write("**Versiones:**")
st.sidebar.write(f"TensorFlow: {tf.__version__}")
st.sidebar.write(f"Streamlit: {st.__version__}")

# Footer
st.markdown("---")
st.caption("Clasificador de Residuos usando EfficientNetB4 - Versión Streamlit")
