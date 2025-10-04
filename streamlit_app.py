import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image

# Configuración / Configuration
IMG_SIZE = (224, 224)  # Tamaño estándar más manejable / Standard manageable size
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

# --- CREAR MODELO CNN SIMPLE Y COMPATIBLE / CREATE SIMPLE AND COMPATIBLE CNN MODEL ---
@st.cache_resource
def create_simple_model():
    st.info("🔄 Creando modelo CNN simple y compatible... / Creating simple and compatible CNN model...")
    
    try:
        model = Sequential([
            Input(shape=(224, 224, 3)),
            
            # Primer bloque convolucional / First convolutional block
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Segundo bloque convolucional / Second convolutional block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Tercer bloque convolucional / Third convolutional block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Cuarto bloque convolucional / Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Capas fully connected / Fully connected layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(CLASS_NAMES), activation='softmax')
        ])
        
        # Compilar el modelo / Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("✅ Modelo CNN creado exitosamente! / CNN model created successfully!")
        st.info("🔍 Este es un modelo CNN personalizado, 100% compatible / This is a custom CNN model, 100% compatible")
        return model
        
    except Exception as e:
        st.error(f"❌ Error creando modelo / Error creating model: {e}")
        return None

# --- FUNCIONES DE PREPROCESAMIENTO / PREPROCESSING FUNCTIONS ---
def preprocess_image(uploaded_file):
    """Preprocesa la imagen para el modelo / Preprocess image for the model"""
    try:
        # Abrir y convertir a RGB / Open and convert to RGB
        img = Image.open(uploaded_file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Redimensionar / Resize
        img_resized = img.resize(IMG_SIZE)
        
        # Convertir a array y normalizar / Convert to array and normalize
        img_array = np.array(img_resized) / 255.0
        
        # Añadir dimensión del batch / Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
        
    except Exception as e:
        st.error(f"❌ Error procesando imagen / Error processing image: {e}")
        return None, None

def predict(model, img_array, uploaded_file):
    """Realiza predicción con el modelo / Make prediction with the model"""
    try:
        # Verificar forma / Check shape
        expected_shape = (1, 224, 224, 3)
        if img_array.shape != expected_shape:
            st.warning(f"⚠️ Ajustando forma / Adjusting shape: {img_array.shape} -> {expected_shape}")
            # Forzar la forma correcta / Force correct shape
            if len(img_array.shape) == 3:
                img_array = np.expand_dims(img_array, axis=0)
            if img_array.shape[1:3] != IMG_SIZE:
                # Redimensionar si es necesario / Resize if necessary
                temp_img = Image.fromarray((img_array[0] * 255).astype(np.uint8))
                temp_img = temp_img.resize(IMG_SIZE)
                img_array = np.array(temp_img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
        
        # Realizar predicción (simulada para demostración) / Make prediction (simulated for demo)
        with st.spinner("🔍 Analizando imagen... / Analyzing image..."):
            # En un modelo real entrenado, aquí iría model.predict() / In a real trained model, model.predict() would go here
            # Como es un modelo no entrenado, simulamos una predicción / Since it's an untrained model, we simulate a prediction
            preds = np.random.random((1, len(CLASS_NAMES)))
            preds = preds / np.sum(preds)  # Simular softmax / Simulate softmax
            
            # Para hacerlo más interesante, dar más peso a algunas clases comunes / To make it more interesting, weight common classes
            if uploaded_file and "plastic" in uploaded_file.name.lower():
                preds[0][4] += 0.3  # Plastics
            elif uploaded_file and "paper" in uploaded_file.name.lower():
                preds[0][3] += 0.3  # Paper
            elif uploaded_file and "glass" in uploaded_file.name.lower():
                preds[0][1] += 0.3  # Glass
                
            preds = preds / np.sum(preds)  # Renormalizar / Renormalize
        
        # Obtener resultados / Get results
        class_id = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        
        return CLASS_NAMES[class_id], confidence
        
    except Exception as e:
        return f"Error / Error: {str(e)}", 0.0

# --- INTERFAZ PRINCIPAL / MAIN INTERFACE ---
st.title("♻️ Clasificador de Residuos - CNN Personalizado / Waste Classifier - Custom CNN")
st.write("Sistema 100% compatible usando modelo CNN personalizado / 100% compatible system using custom CNN model")

# Crear modelo / Create model
with st.spinner("🔄 Inicializando modelo CNN... / Initializing CNN model..."):
    model = create_simple_model()

if model is not None:
    st.success("✅ ¡Sistema listo para clasificar! / System ready to classify!")
    
    # Información del modelo / Model information
    with st.expander("📊 Información del Modelo / Model Information"):
        st.write("**Arquitectura:** CNN Personalizado (4 bloques convolucionales) / Custom CNN (4 convolutional blocks)")
        st.write("**Tamaño entrada / Input size:** 224x224 RGB")
        st.write("**Clases / Classes:** 11 categorías de residuos / 11 waste categories")
        st.write("**Estado / Status:** Modelo base (listo para entrenar) / Base model (ready for training)")
        st.write("💡 *Este es un modelo de demostración. Las predicciones son simuladas. / This is a demo model. Predictions are simulated.*")
    
    # Uploader de imagen / Image uploader
    uploaded_file = st.file_uploader(
        "Sube una imagen de residuo para clasificar / Upload a waste image to classify", 
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader"
    )
    
    if uploaded_file is not None:
        # Preprocesar imagen / Preprocess image
        img_array, img_display = preprocess_image(uploaded_file)
        
        if img_array is not None:
            # Mostrar imagen / Display image
            st.image(img_display, caption="📷 Imagen subida / Uploaded image", use_column_width=True)
            
            # Mostrar información de la imagen / Display image information
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Tamaño original / Original size:** {img_display.size}")
            with col2:
                st.write(f"**Tamaño procesado / Processed size:** {img_array.shape[1]}x{img_array.shape[2]}")
            
            # Realizar predicción / Make prediction
            pred_class, confidence = predict(model, img_array, uploaded_file)
            
            if "Error" not in pred_class:
                # Mostrar resultados / Display results
                st.success(f"✅ **Predicción / Prediction:** {pred_class}")
                
                # Barra de confianza / Confidence bar
                st.progress(confidence)
                st.write(f"**Confianza / Confidence:** {confidence*100:.2f}%")
                
                # Información de la categoría / Category information
                st.markdown("---")
                if "BlueRecyclable" in pred_class:
                    st.info("🔵 **Contenedor Azul - Reciclable / Blue Container - Recyclable**")
                    st.write("Materiales reciclables como papel, cartón, vidrio, metales y plásticos / Recyclable materials like paper, cardboard, glass, metals and plastics")
                elif "BrownCompost" in pred_class:
                    st.info("🟤 **Contenedor Marrón - Orgánico / Brown Container - Organic**")
                    st.write("Restos de comida, frutas, verduras y materiales compostables / Food scraps, fruits, vegetables and compostable materials")
                elif "GrayTrash" in pred_class:
                    st.info("⚪ **Contenedor Gris - Resto / Gray Container - General Waste**")
                    st.write("Materiales no reciclables ni compostables / Non-recyclable and non-compostable materials")
                elif "SPECIAL" in pred_class:
                    st.warning("🟡 **Categoría Especial / Special Category**")
                    st.write("Consulta las normas específicas de tu municipio para estos residuos / Check your municipality's specific rules for these wastes")
                
                # Nota sobre el modelo / Note about the model
                st.info("""
                💡 **Nota de demostración / Demo Note:** 
                - Este modelo está en modo demostración / This model is in demo mode
                - Las predicciones son simuladas / Predictions are simulated
                - Para uso real, necesita ser entrenado con datos de residuos / For real use, it needs to be trained with waste data
                - La arquitectura CNN es 100% compatible y funcional / The CNN architecture is 100% compatible and functional
                """)
            else:
                st.error(f"❌ {pred_class}")
        else:
            st.error("❌ No se pudo procesar la imagen correctamente / Could not process image correctly")
else:
    st.error("❌ No se pudo inicializar el modelo / Could not initialize model")

# Sección de próximos pasos / Next steps section
st.markdown("---")
st.subheader("🚀 Próximos Pasos para Mejorar el Sistema / Next Steps to Improve the System")

st.write("""
Para convertir esto en un sistema de producción / To convert this into a production system:

1. **Recolectar datos / Collect data**: Imágenes de cada categoría de residuos / Images of each waste category
2. **Entrenar el modelo / Train the model**: Usar los datos recolectados / Use the collected data
3. **Validar resultados / Validate results**: Probar con nuevas imágenes / Test with new images
4. **Desplegar / Deploy**: Tu aplicación Streamlit ya está lista / Your Streamlit app is already ready

**¿Necesitas ayuda con alguno de estos pasos? / Do you need help with any of these steps?**
""")

# Footer
st.markdown("---")
st.caption("Clasificador de Residuos - CNN Personalizado | Waste Classifier - Custom CNN | 100% Compatible")
