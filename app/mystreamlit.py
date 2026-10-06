import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Configuración de la página
st.set_page_config(page_title="Predicción de Depósitos", page_icon="🏦", layout="centered")

# Función para cargar el modelo (usamos caché para que no lo cargue cada vez que tocas un botón)
@st.cache_resource
def cargar_modelo():
    # Ruta relativa desde la carpeta 'app' a la carpeta 'modelos'
    ruta_modelo = os.path.join(os.path.dirname(__file__), '../modelos/modelo_final.joblib')
    return joblib.load(ruta_modelo)

modelo = cargar_modelo()

st.title("🏦 Predicción de Suscripción a Depósito")
st.write("Introduce los datos del cliente para predecir si contratará el depósito a plazo.")

# --- FORMULARIO DE ENTRADA DE DATOS ---
with st.form("cliente_form"):
    st.subheader("Datos Demográficos y Financieros")
    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Edad", min_value=18, max_value=100, value=30)
    job = col2.selectbox("Trabajo", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = col3.selectbox("Estado Civil", ['divorced', 'married', 'single', 'unknown'])
    
    col4, col5, col6 = st.columns(3)
    education = col4.selectbox("Educación", ['primary', 'secondary', 'tertiary', 'unknown'])
    default = col5.selectbox("¿Crédito en mora? (default)", ['no', 'yes', 'unknown'])
    balance = col6.number_input("Balance Medio Anual", value=1000)
    
    col7, col8 = st.columns(2)
    housing = col7.selectbox("¿Hipoteca? (housing)", ['no', 'yes', 'unknown'])
    loan = col8.selectbox("¿Préstamo personal? (loan)", ['no', 'yes', 'unknown'])

    st.subheader("Datos del Contacto de Marketing")
    col9, col10, col11 = st.columns(3)
    contact = col9.selectbox("Tipo de contacto", ['cellular', 'telephone', 'unknown'])
    day = col10.number_input("Día del mes", min_value=1, max_value=31, value=15)
    month = col11.selectbox("Mes", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    
    col12, col13 = st.columns(2)
    duration = col12.number_input("Duración última llamada (seg)", min_value=0, value=200)
    campaign = col13.number_input("Contactos en esta campaña", min_value=1, value=1)
    
    col14, col15, col16 = st.columns(3)
    pdays = col14.number_input("Días desde último contacto (-1 = nunca)", min_value=-1, value=-1)
    previous = col15.number_input("Contactos previos", min_value=0, value=0)
    poutcome = col16.selectbox("Resultado campaña anterior", ['failure', 'nonexistent', 'success', 'unknown'])

    # Botón para predecir
    submit_button = st.form_submit_button(label="🔮 Predecir Suscripción")

# --- LÓGICA DE PREDICCIÓN ---
if submit_button:
    # 1. Recoger los datos en un DataFrame
    datos_entrada = pd.DataFrame([{
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
        'contact': contact, 'day': day, 'month': month, 'duration': duration,
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
    }])
    
    # 2. Preprocesamiento manual exacto al del Notebook (variable pdays)
    datos_entrada['contacto_previo'] = np.where(datos_entrada['pdays'] == -1, 0, 1)
    datos_entrada['pdays'] = np.where(datos_entrada['pdays'] == -1, 0, datos_entrada['pdays'])
    
    # 3. Hacer la predicción (El Pipeline hace el StandardScaler y el OneHotEncoding automáticamente)
    prediccion = modelo.predict(datos_entrada)[0]
    
    # 4. Mostrar resultado
    st.divider()
    if prediccion == 1:
        st.success("¡ALTA PROBABILIDAD! 🌟 El modelo predice que el cliente **SÍ** suscribirá el depósito.")
    else:
        st.error("BAJA PROBABILIDAD ❌ El modelo predice que el cliente **NO** suscribirá el depósito.")