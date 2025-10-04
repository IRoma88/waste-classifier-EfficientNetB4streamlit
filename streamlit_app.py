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
        st.info("📥 Descargando modelo...")
        try:
            file_id = "1xlzVWU680kSKIpJGl6i0mgTdct4QE_La"
            gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)
            
            if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100000:
                st.success("✅ Modelo descargado exitosamente!")
                return True
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

# --- CARGA DE MODELO CON MÚLTIPLES INTENTOS ---
@st.cache_resource
def load_model():
    if not download_success:
        return None
        
    try:
        file_size = os.path.getsize(MODEL_PATH)
        st.info(f"📊 Cargando modelo: {file_size} bytes")
        
        # INTENTO 1: Carga directa con compile=False
        try:
            with st.spinner("🔄 Intentando carga directa..."):
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                # Compilar manualmente
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                st.success("✅ ¡Modelo cargado con éxito!")
                return model
        except Exception as e1:
            st.warning(f"⚠️ Intento 1 falló: {e1}")
            
            # INTENTO 2: Cargar solo la arquitectura y pesos por separado
            try:
                with st.spinner("🔄 Intentando carga con custom objects..."):
                    # Crear un modelo EfficientNetB4 base y cargar pesos
                    base_model = tf.keras.applications.EfficientNetB4(
                        include_top=False,
                        weights=None,
                        input_shape=(380, 380, 3)
                    )
                    
                    # Construir modelo personalizado
                    inputs = tf.keras.Input(shape=(380, 380, 3))
                    x = base_model(inputs, training=False)
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    x = tf.keras.layers.Dense(512, activation='relu')(x)
                    x = tf.keras.layers.Dropout(0.3)(x)
                    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)
                    
                    model = tf.keras.Model(inputs, outputs)
                    
                    # Cargar pesos del modelo guardado
                    model.load_weights(MODEL_PATH)
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    
                    st.success("✅ ¡Modelo reconstruido con éxito!")
                    return model
                    
            except Exception as e2:
                st.warning(f"⚠️ Intento 2 falló: {e2}")
                
                # INTENTO 3: Usar el modelo directamente sin verificación
                try:
                    with st.spinner("🔄 Cargando modelo sin verificaciones..."):
                        model = tf.keras.models.load_model(MODEL_PATH)
                        st.success("✅ ¡Modelo cargado en modo simple!")
                        return model
                except Exception as e3:
                    st.error(f"❌ Intento 3 falló: {e3}")
                    return None
                    
    except Exception as e:
        st.error(f"❌ Error general: {e}")
        return None

model = load_model()

# --- FUNCIONES SIMPLIFICADAS ---
def preprocess_image(uploaded_file):
    """Preprocesa la imagen de forma robusta"""
    try:
        # Leer la imagen
        img = Image.open(uploaded_file)
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img_resized = img.resize(IMG_SIZE)
        
        # Convertir a array y normalizar
        img_array = np.array(img_resized) / 255.0
        
        # Añadir dimensión del batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
        
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {e}")
        return None, None

def predict(img_array):
    if model is None:
        return "Modelo no disponible", 0.0
    
    try:
        # Verificar forma
        if img_array.shape != (1, 380, 380, 3):
            st.error(f"❌ Forma incorrecta: {img_array.shape}")
            return "Error en formato de imagen", 0.0
        
        # Realizar predicción
        with st.spinner("🔍 Analizando imagen..."):
            preds = model.predict(img_array, verbose=0)
        
        # Obtener resultados
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        
        return CLASS_NAMES[class_id], confidence
        
    except Exception as e:
        return f"Error en predicción: {str(e)}", 0.0

# --- INTERFAZ ---
st.title("♻️ Clasificador de Residuos - EfficientNetB4")
st.write("Sube una imagen de un residuo para clasificarlo")

# Estado
if download_success:
    st.success("📥 Modelo descargado")
else:
    st.error("📥 Error descargando modelo")

if model is not None:
    st.success("🧠 Modelo cargado - ¡Listo para usar!")
    
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        img_array, img_disp = preprocess_image(uploaded_file)
        
        if img_array is not None:
            st.image(img_disp, caption="Imagen subida", use_column_width=True)
            
            pred_class, conf = predict(img_array)
            
            if "Error" not in pred_class:
                st.success(f"✅ **{pred_class}**")
                st.info(f"📊 **Confianza:** {conf*100:.2f}%")
                
                # Color code por tipo
                if "BlueRecyclable" in pred_class:
                    st.markdown("🔵 **Contenedor Azul - Reciclable**")
                elif "BrownCompost" in pred_class:
                    st.markdown("🟤 **Contenedor Marrón - Orgánico**")
                elif "GrayTrash" in pred_class:
                    st.markdown("⚪ **Contenedor Gris - Resto**")
                elif "SPECIAL" in pred_class:
                    st.markdown("🟡 **Categoría Especial**")
            else:
                st.error(pred_class)
else:
    st.error("❌ No se pudo cargar el modelo")
    
    st.markdown("""
    ### 🔧 Soluciones Alternativas:
    
    **Opción 1: Usar un modelo preentrenado público**
    - Podemos usar EfficientNetB4 con ImageNet y ajustarlo
    
    **Opción 2: Entrenar un modelo más simple**
    - Un modelo CNN básico que sea 100% compatible
    
    **Opción 3: Usar un servicio externo**
    - Hugging Face, TensorFlow Hub, etc.
    
    ¿Quieres que implemente alguna de estas alternativas?
    """)

st.markdown("---")
st.caption("Clasificador de Residuos - EfficientNetB4")
