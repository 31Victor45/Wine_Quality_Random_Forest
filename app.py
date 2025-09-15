import streamlit as st
import pandas as pd
import joblib

# Cargamos el modelo optimizado
try:
    model = joblib.load('p_items/best_random_forest_model.pkl')
except FileNotFoundError:
    st.error("Error: El modelo no ha sido entrenado. Por favor, ejecuta el script 'model_optimizer.py' primero.")

# Título y descripción de la aplicación
st.title('Predictor de Calidad de Vinos 🍷')

# Agregamos la imagen principal y ajustamos su tamaño
try:
    st.image('p_items/img/img_5.png', caption='Un viaje al corazón de la calidad del vino', width=350) # Cambiado
except FileNotFoundError:
    st.warning("Advertencia: No se encontró el archivo de imagen 'wine_image.png'. Asegúrate de que está en la misma carpeta que 'app.py'.")

st.markdown("""
Esta aplicación utiliza un modelo de Bosques Aleatorios optimizado para predecir si un vino es de alta o baja calidad basado en sus propiedades químicas.
""")

# Interfaz de usuario para la entrada de datos
st.sidebar.header('Parámetros de Entrada del Vino')

def get_user_input():
    fixed_acidity = st.sidebar.slider('Acidez Fija (Fixed Acidity)', 4.6, 15.9, 7.4)
    volatile_acidity = st.sidebar.slider('Acidez Volátil (Volatile Acidity)', 0.12, 1.58, 0.70)
    citric_acid = st.sidebar.slider('Ácido Cítrico (Citric Acid)', 0.0, 1.0, 0.0)
    residual_sugar = st.sidebar.slider('Azúcar Residual (Residual Sugar)', 0.9, 15.5, 1.9)
    chlorides = st.sidebar.slider('Cloruros (Chlorides)', 0.012, 0.612, 0.076)
    free_sulfur_dioxide = st.sidebar.slider('Dióxido de Azufre Libre (Free SO2)', 1.0, 72.0, 11.0)
    total_sulfur_dioxide = st.sidebar.slider('Dióxido de Azufre Total (Total SO2)', 6.0, 289.0, 34.0)
    density = st.sidebar.slider('Densidad (Density)', 0.99, 1.004, 0.9978)
    pH = st.sidebar.slider('pH', 2.74, 4.01, 3.51)
    sulphates = st.sidebar.slider('Sulfatos (Sulphates)', 0.33, 2.0, 0.56)
    alcohol = st.sidebar.slider('Alcohol', 8.4, 14.9, 9.4)

    user_data = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur_dioxide,
        'total sulfur dioxide': total_sulfur_dioxide,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

input_df = get_user_input()

st.subheader('Parámetros de Entrada Seleccionados')
st.write(input_df)

# Realizar la predicción
if st.sidebar.button('Predecir Calidad del Vino'):
    if 'model' in locals():
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Resultado de la Predicción')
        
        if prediction[0] == 1:
            st.success('¡El modelo predice que este es un **Vino de Alta Calidad**! 🏆')
        else:
            st.error('¡El modelo predice que este es un **Vino de Baja Calidad**! 📉')
            
        st.markdown("---")
        st.subheader('Probabilidad de la Predicción')
        st.write(f"Probabilidad de Baja Calidad: {prediction_proba[0][0]*100:.2f}%")
        st.write(f"Probabilidad de Alta Calidad: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.warning("El modelo no está cargado. Por favor, asegúrate de que 'best_random_forest_model.pkl' existe.")

# Botón nativo de Streamlit que enlaza a la web
st.sidebar.markdown("---")
st.sidebar.link_button(
    "Conoce el Proyecto Completo",
    "TU_ENLACE_A_LA_WEB"
)