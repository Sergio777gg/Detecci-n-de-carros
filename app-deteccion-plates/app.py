import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO

# -------------------------------
# Configuración de página
# -------------------------------
st.set_page_config(page_title="Detección de Carros", layout="wide")

st.title("🚗 Detección de vehículos")

# -------------------------------
# Ruta del modelo (FIX IMPORTANTE)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

# -------------------------------
# Cargar modelo (cacheado)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.header("⚙️ Configuración")

conf_threshold = st.sidebar.slider(
    "Confianza",
    min_value=0.0,
    max_value=1.0,
    value=0.5
)

# -------------------------------
# Subida de imagen
# -------------------------------
uploaded_file = st.file_uploader(
    "📤 Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Procesamiento
# -------------------------------
if uploaded_file is not None:

    # Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Predicción
    results = model.predict(image, conf=conf_threshold)

    # Imagen con detecciones
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Layout en columnas
    col1, col2 = st.columns(2)

    # -------------------------------
    # Columna 1: Imagen con detecciones
    # -------------------------------
    with col1:
        st.subheader("📌 Detecciones")
        st.image(annotated_image)

    # -------------------------------
    # Columna 2: Recortes
    # -------------------------------
    with col2:
        st.subheader("🔍 Carros detectados")

        boxes = results[0].boxes

        num_carros = len(boxes) if boxes is not None else 0
        st.success(f"🚗 Carros detectados: {num_carros}")

        if boxes is not None and len(boxes) > 0:

            h, w, _ = image.shape

            for i, box in enumerate(boxes):

                # Coordenadas
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Limitar dentro de la imagen
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Recorte
                cropped = image[y1:y2, x1:x2]

                # Validación
                if cropped.size > 0:
                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

                    st.image(
                        cropped_rgb,
                        caption=f"Carro {i+1}"
                    )

        else:
            st.warning("No se detectaron carros en la imagen")