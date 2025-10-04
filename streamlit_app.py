import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import os
from PIL import Image
import urllib.request

# Configuración
IMG_SIZE = (380, 380)
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

# --- CREAR MODELO COMPATIBLE ---
@st.cache_resource
def create_compatible_model():
    st.info("🔄 Creando modelo compatible con EfficientNetB4...")
    
    try:
        # Cargar EfficientNetB4 preentrenado en ImageNet
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=(380, 380, 3)
        )
        
        # Congelar las capas base (transfer learning)
        base_model.trainable = False
        
        # Añadir capas personalizadas para clasificación
        inputs = tf.keras.Input(shape=(380, 380, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(len(CLASS_NAMES), activation='softmax')(x)
        
        # Crear modelo
        model = Model(inputs, outputs)
        
        # Compilar
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("✅ Modelo compatible creado exitosamente!")
        st.info("🔍 Este modelo usa EfficientNetB4 preentrenado en ImageNet")
        return model
        
    except Exception as e:
        st.error(f"❌ Error creando modelo: {e}")
        return None

# --- FUNCIONES DE PREPROCESAMIENTO ---
def preprocess_image(uploaded_file):
    """Preprocesa la imagen para el modelo"""
    try:
        # Abrir y convertir a RGB
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar
        img_resized = img.resize(IMG_SIZE)
        
        # Convertir a array y preprocesar para EfficientNet
        img_array = tf.keras.applications.efficientnet.preprocess_input(
            np.array(img_resized)
        )
        
        # Añadir dimensión del batch
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
        
    except Exception as e:
        st.error(f"❌ Error procesando imagen: {e}")
        return None, None

def predict(model, img_array):
    """Realiza predicción con el modelo"""
    try:
        # Verificar forma
        if img_array.shape != (1, 380, 380, 3):
            st.error(f"❌ Forma incorrecta: {img_array.shape}")
            return "Error en formato", 0.0
        
        # Realizar predicción
        with st.spinner("🔍 Analizando imagen..."):
            preds = model.predict(img_array, verbose=0)
        
        # Obtener resultados
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        
        return CLASS_NAMES[class_id], confidence
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL ---
st.title("♻️ Clasificador de Residuos - EfficientNetB4")
st.write("Sistema compatible usando modelo preentrenado de ImageNet")

# Crear/descargar modelo
with st.spinner("🔄 Inicializando modelo..."):
    model = create_compatible_model()

if model is not None:
    st.success("✅ ¡Sistema listo para clasificar!")
    
    # Información del modelo
    with st.expander("📊 Información del Modelo"):
        st.write("**Arquitectura:** EfficientNetB4 preentrenado")
        st.write("**Dataset base:** ImageNet")
        st.write("**Técnica:** Transfer Learning")
        st.write("**Clases:** 11 categorías de residuos")
        st.write("💡 *Nota: Este es un modelo genérico. Para mejor precisión, se necesitaría entrenar con datos específicos.*")
    
    # Uploader de imagen
    uploaded_file = st.file_uploader(
        "Sube una imagen de residuo para clasificar", 
        type=["jpg", "jpeg", "png", "webp"]
    )
    
    if uploaded_file is not None:
        # Preprocesar imagen
        img_array, img_display = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Mostrar imagen
            st.image(img_display, caption="📷 Imagen subida", use_column_width=True)
            
            # Realizar predicción
            pred_class, confidence = predict(model, img_array)
            
            if "Error" not in pred_class:
                # Mostrar resultados
                st.success(f"✅ **Predicción:** {pred_class}")
                
                # Barra de confianza
                st.progress(confidence)
                st.write(f"**Confianza:** {confidence*100:.2f}%")
                
                # Información de la categoría
                st.markdown("---")
                if "BlueRecyclable" in pred_class:
                    st.info("🔵 **Contenedor Azul - Reciclable**")
                    st.write("Materiales reciclables como papel, cartón, vidrio, metales y plásticos")
                elif "BrownCompost" in pred_class:
                    st.info("🟤 **Contenedor Marrón - Orgánico**")
                    st.write("Restos de comida, frutas, verduras y materiales compostables")
                elif "GrayTrash" in pred_class:
                    st.info("⚪ **Contenedor Gris - Resto**")
                    st.write("Materiales no reciclables ni compostables")
                elif "SPECIAL" in pred_class:
                    st.warning("🟡 **Categoría Especial**")
                    st.write("Consulta las normas específicas de tu municipio para estos residuos")
                
                # Nota sobre el modelo
                st.info("""
                💡 **Nota:** Este es un modelo de demostración usando transfer learning. 
                Para mayor precisión, el modelo debería ser entrenado específicamente con imágenes de residuos.
                """)
            else:
                st.error(f"❌ {pred_class}")
        else:
            st.error("❌ No se pudo procesar la imagen correctamente")
else:
    st.error("❌ No se pudo inicializar el modelo")

# Footer
st.markdown("---")
st.caption("Clasificador de Residuos - EfficientNetB4 Preentrenado | Compatible 100%")
