import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN BÁSICA DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Predictor de Churn",
    layout="wide"
)

st.title("Predictor de Churn de Clientes Telco")
st.markdown("### Evaluacion Final Aprendizaje de Maquina")

# Las 19 características completas
FEATURES_COMPLETAS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Las 7 características más importantes
FEATURES_TOP_7 = [
    'TotalCharges', 'MonthlyCharges', 'tenure', 'InternetService', 
    'PaymentMethod', 'Contract', 'gender'
]


@st.cache_data  
def cargar_dataset():

    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        st.success("✅ Dataset cargado correctamente")
        return df
    
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo CSV: 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")
        st.error("Por favor, asegúrate de que el archivo esté en el directorio de la aplicación")
        return None


def limpiar_datos(df_original):

    if df_original is None:
        return None, None
        
    df = df_original.copy()
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    if 'Churn' in df.columns:
        y = df['Churn'].map({'No': 0, 'Yes': 1})  # Convertir a números
        X = df.drop('Churn', axis=1)
    else:
        st.error("❌ No se encontró la columna 'Churn' en el dataset")
        return None, None
    
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)
        
    
    servicios_arreglar = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                         'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for servicio in servicios_arreglar:
        if servicio in X.columns:
            X[servicio] = X[servicio].replace('No internet service', 'No')
    
    
    return X, y


@st.cache_resource  
def cargar_modelos():

    modelos_disponibles = {}
    errores = []
    
    modelos_base = {
        'Stacking Diverse': {
            '19': 'stacking_diverse_trained.pkl',
            '7': 'stacking_diverse_trained_7.pkl'
        },
        'Logistic Regression': {
            '19': 'Single Classifier (Logistic Regression)_trained.pkl',
            '7': 'Single Classifier_7.pkl'
        },
        'Voting Classifier': {
            '19': 'Voting Classifier (Soft)_trained.pkl',
            '7': 'Voting Classifier (Soft)_trained_7.pkl'
        }
    }
    
    for modelo_base, archivos in modelos_base.items():
        modelos_disponibles[modelo_base] = {}
        for num_features, archivo in archivos.items():
            try:
                if os.path.exists(archivo):
                    modelos_disponibles[modelo_base][num_features] = archivo
                else:
                    errores.append(f"❌ No se encontró: {archivo}")
            except Exception as e:
                errores.append(f"❌ Error con {archivo}: {str(e)}")
    
    if errores:
        st.warning("Algunos archivos de modelos no se encontraron:")
        for error in errores:
            st.write(error)
    
    return modelos_disponibles

def cargar_modelo_especifico(modelo_base, num_features, modelos_disponibles):

    try:
        if modelo_base in modelos_disponibles and num_features in modelos_disponibles[modelo_base]:
            archivo = modelos_disponibles[modelo_base][num_features]
            modelo = joblib.load(archivo)
            return modelo, archivo
        else:
            return None, None
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None


def obtener_peso_modelo(modelo, nombre_archivo):

    try:
        tamaño_bytes = os.path.getsize(nombre_archivo)
        
        tamaño_mb = tamaño_bytes / (1024 * 1024)
        
        return tamaño_mb
    except:
        return 0.0


def medir_tiempo_prediccion(modelo, datos_prueba, repeticiones=100):
    """
    Función para medir cuánto tiempo tarda el modelo en hacer predicciones
    """
    try:
        tiempos = []
        
        for i in range(repeticiones):
            inicio = time.time()
            modelo.predict(datos_prueba)
            fin = time.time()
            tiempos.append(fin - inicio)
        
        tiempo_promedio_ms = np.mean(tiempos) * 1000
        
        return tiempo_promedio_ms
    except:
        return 0.0


def procesar_datos_cliente(datos_cliente, usar_7_features=False):
    """
    Función simple para convertir los datos del cliente en números
    que el modelo pueda entender
    """
    if usar_7_features:
        datos_procesados = []
        
        datos_procesados.append(float(datos_cliente.get('TotalCharges', 0)))
        
        datos_procesados.append(float(datos_cliente.get('MonthlyCharges', 0)))
        
        datos_procesados.append(int(datos_cliente.get('tenure', 0)))
        
        internet = datos_cliente.get('InternetService', 'DSL')
        datos_procesados.append(1 if internet == 'Fiber optic' else 0)
        
        pago = datos_cliente.get('PaymentMethod', 'Electronic check')
        datos_procesados.append(1 if pago == 'Electronic check' else 0)
        
        contrato = datos_cliente.get('Contract', 'Month-to-month')
        datos_procesados.append(1 if contrato == 'Two year' else 0)
        
        genero = datos_cliente.get('gender', 'Male')
        datos_procesados.append(1 if genero == 'Male' else 0)
        
        return np.array(datos_procesados).reshape(1, -1)
    
    else:
  
        datos_procesados = []
        
        datos_procesados.append(int(datos_cliente.get('SeniorCitizen', 0)))
        datos_procesados.append(int(datos_cliente.get('tenure', 0)))
        datos_procesados.append(float(datos_cliente.get('MonthlyCharges', 0)))
        datos_procesados.append(float(datos_cliente.get('TotalCharges', 0)))
        
        datos_procesados.append(1 if datos_cliente.get('gender', 'Male') == 'Male' else 0)
        datos_procesados.append(1 if datos_cliente.get('Partner', 'No') == 'Yes' else 0)
        datos_procesados.append(1 if datos_cliente.get('Dependents', 'No') == 'Yes' else 0)
        datos_procesados.append(1 if datos_cliente.get('PhoneService', 'No') == 'Yes' else 0)
        datos_procesados.append(1 if datos_cliente.get('MultipleLines', 'No') == 'Yes' else 0)
        
        internet = datos_cliente.get('InternetService', 'DSL')
        if internet == 'DSL':
            datos_procesados.append(0)
        elif internet == 'Fiber optic':
            datos_procesados.append(1)
        else:
            datos_procesados.append(2)
        
        datos_procesados.append(1 if datos_cliente.get('OnlineSecurity', 'No') == 'Yes' else 0)
        datos_procesados.append(1 if datos_cliente.get('OnlineBackup', 'No') == 'Yes' else 0)
        datos_procesados.append(1 if datos_cliente.get('DeviceProtection', 'No') == 'Yes' else 0)
        datos_procesados.append(1 if datos_cliente.get('TechSupport', 'No') == 'Yes' else 0)
        datos_procesados.append(1 if datos_cliente.get('StreamingTV', 'No') == 'Yes' else 0)
        datos_procesados.append(1 if datos_cliente.get('StreamingMovies', 'No') == 'Yes' else 0)
        
        
        contrato = datos_cliente.get('Contract', 'Month-to-month')
        if contrato == 'Month-to-month':
            datos_procesados.append(0)
        elif contrato == 'One year':
            datos_procesados.append(1)
        else:
            datos_procesados.append(2)
   
        datos_procesados.append(1 if datos_cliente.get('PaperlessBilling', 'No') == 'Yes' else 0)
        datos_procesados.append(1 if datos_cliente.get('PaymentMethod', 'Electronic check') == 'Electronic check' else 0)
        
        return np.array(datos_procesados).reshape(1, -1)


# Cargar el dataset
with st.spinner("Cargando dataset..."):
    dataset_original = cargar_dataset()

# Limpiar los datos
if dataset_original is not None:
    with st.spinner("Limpiando datos..."):
        X_limpio, y_limpio = limpiar_datos(dataset_original)
else:
    X_limpio, y_limpio = None, None

# Cargar los modelos
with st.spinner("Cargando modelos de machine learning..."):
    modelos_disponibles = cargar_modelos()


total_modelos = sum(len(variantes) for variantes in modelos_disponibles.values())
if dataset_original is not None or total_modelos > 0:
    

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Predicción", 
        "EDA", 
        "Datos Limpios", 
        "Métricas y Rendimiento", 
        "Dashboard"
    ])

    # PESTAÑA 1: PREDICCIÓN

    with tab1:
        
        if total_modelos == 0:
            st.error("❌ No hay modelos disponibles para hacer predicciones")
            st.info("Por favor, asegúrate de que los archivos .pkl de los modelos estén en el directorio")
        else:
            st.subheader("Configuración del Modelo")
            
            col_config1, col_config2 = st.columns(2)
            
            with col_config1:
                modelos_base_disponibles = [modelo for modelo in modelos_disponibles.keys() 
                                          if len(modelos_disponibles[modelo]) > 0]
                
                if len(modelos_base_disponibles) == 0:
                    st.error("No hay modelos base disponibles")
                    modelo_seleccionado = None
                else:
                    modelo_seleccionado = st.selectbox(
                        "Selecciona el Modelo:",
                        modelos_base_disponibles,
                    )
            
            with col_config2:
                if modelo_seleccionado:
                    variantes_disponibles = list(modelos_disponibles[modelo_seleccionado].keys())
                    
                    opciones_features = []
                    if '19' in variantes_disponibles:
                        opciones_features.append("Todas las características (19)")
                    if '7' in variantes_disponibles:
                        opciones_features.append("Solo las 7 más importantes")
                    
                    if len(opciones_features) == 0:
                        st.error("No hay variantes disponibles para este modelo")
                        tipo_features = None
                    else:
                        tipo_features = st.selectbox(
                            "Número de Características:",
                            opciones_features,
                            
                        )
                else:
                    tipo_features = None
            
            if tipo_features:
                usar_7_features = "7 más importantes" in tipo_features
                num_features_str = '7' if usar_7_features else '19'
                
                    
            st.markdown("---")  
            
            col_formulario, col_resultado = st.columns([2, 1])
            
            with col_formulario:
                st.subheader("Datos del Cliente")
                
                with st.form("formulario_cliente"):
                    
                    if usar_7_features:
                        # FORMULARIO SIMPLIFICADO - SOLO 7 CARACTERÍSTICAS
                        
                        st.markdown("** 7 Características Principales**")
                        
                        gender = st.selectbox("Género", ["Male", "Female"])
                        
                        tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)
                        
                        col_cargos1, col_cargos2 = st.columns(2)
                        with col_cargos1:
                            MonthlyCharges = st.number_input("MonthlyCharges ($)", min_value=0.0, value=50.0)
                        with col_cargos2:
                            TotalCharges = st.number_input("TotalCharges ($)", min_value=0.0, value=1000.0)
                        
                        InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
                        
                        PaymentMethod = st.selectbox("PaymentMethod", 
                            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                        
                        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                        
                        SeniorCitizen = 0
                        Partner = "No"
                        Dependents = "No"
                        PhoneService = "Yes"
                        MultipleLines = "No"
                        OnlineSecurity = "No"
                        OnlineBackup = "No"
                        DeviceProtection = "No"
                        TechSupport = "No"
                        StreamingTV = "No"
                        StreamingMovies = "No"
                        PaperlessBilling = "Yes"
                        
                    else:
                        # FORMULARIO COMPLETO - TODAS LAS 19 CARACTERÍSTICAS
                        
                        st.markdown("** Todas las Características**")
                        
                        st.markdown("**👤 Información Personal**")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            gender = st.selectbox("Género", ["Male", "Female"])
                            SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                            Partner = st.selectbox("Partner", ["No", "Yes"])
                        
                        with col2:
                            Dependents = st.selectbox("Dependientes", ["No", "Yes"])
                            tenure = st.number_input("Meses como Cliente(tenure)", min_value=0, max_value=100, value=12)
                        
                        # Servicios
                        st.markdown("**📞 Servicios**")
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
                            MultipleLines = st.selectbox("MultipleLines", ["No", "Yes"])
                            InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
                        
                        with col4:
                            OnlineSecurity = st.selectbox("OnlineSecurity", ["No", "Yes"])
                            OnlineBackup = st.selectbox("OnlineBackup", ["No", "Yes"])
                            DeviceProtection = st.selectbox("DeviceProtection", ["No", "Yes"])
                        
                        col5, col6 = st.columns(2)
                        
                        with col5:
                            TechSupport = st.selectbox("TechSupport", ["No", "Yes"])
                            StreamingTV = st.selectbox("StreamingTV", ["No", "Yes"])
                        
                        with col6:
                            StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
                        
                        st.markdown("**💳 Contrato y Pagos**")
                        col7, col8 = st.columns(2)
                        
                        with col7:
                            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                            PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
                        
                        with col8:
                            PaymentMethod = st.selectbox("PaymentMethodo", 
                                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                        
                        col9, col10 = st.columns(2)
                        
                        with col9:
                            MonthlyCharges = st.number_input("MonthlyCharges ($)", min_value=0.0, value=50.0)
                        
                        with col10:
                            TotalCharges = st.number_input("TotalCharges ($)", min_value=0.0, value=1000.0)
                    
                    boton_predecir = st.form_submit_button("Predicción", type="primary")
            
            with col_resultado:
                if boton_predecir and modelo_seleccionado and tipo_features:
                    st.subheader("Resultado")
                    
                    datos_cliente = {
                        'SeniorCitizen': SeniorCitizen, 'tenure': tenure, 'MonthlyCharges': MonthlyCharges,
                        'TotalCharges': TotalCharges, 'gender': gender, 'Partner': Partner,
                        'Dependents': Dependents, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
                        'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
                        'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport, 'StreamingTV': StreamingTV,
                        'StreamingMovies': StreamingMovies, 'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
                        'PaymentMethod': PaymentMethod
                    }
                    
                    try:
                        modelo, archivo_modelo = cargar_modelo_especifico(modelo_seleccionado, num_features_str, modelos_disponibles)
                        
                        if modelo is None:
                            st.error(f"❌ No se pudo cargar el modelo {modelo_seleccionado} con {num_features_str} características")
                        else:
                            datos_procesados = procesar_datos_cliente(datos_cliente, usar_7_features)
                            
                            prediccion = modelo.predict(datos_procesados)[0]
                            probabilidades = modelo.predict_proba(datos_procesados)[0]
                            
                            if prediccion == 1:
                                st.error("**RIESGO ALTO**")
                                st.error("El cliente probablemente abandonará")
                            else:
                                st.success("**RIESGO BAJO**")
                                st.success("El cliente probablemente se quedará")
                            
                            st.write("**Probabilidades:**")
                            st.write(f"- No Churn: {probabilidades[0]:.1%}")
                            st.write(f"- Churn: {probabilidades[1]:.1%}")
                            
                            fig = go.Figure(data=[
                                go.Bar(x=['No Churn', 'Churn'], 
                                      y=[probabilidades[0], probabilidades[1]],
                                      marker_color=['green', 'red'])
                            ])
                            fig.update_layout(title="Probabilidades", height=300)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.info(f"**Modelo usado:** {modelo_seleccionado}")
                            st.info(f"**Features usadas:** {num_features_str}")
                        
                            
                    except Exception as e:
                        st.error(f"Error en la predicción: {e}")
                        st.error("Verifica que el modelo y las características sean compatibles")
                else:
                    if not modelo_seleccionado:
                        st.warning("⚠️ Selecciona un modelo")
                    elif not tipo_features:
                        st.warning("⚠️ Selecciona el tipo de características")
                    elif modelo_seleccionado and num_features_str not in modelos_disponibles[modelo_seleccionado]:
                        st.error(f"❌ El modelo {modelo_seleccionado} no está disponible con {num_features_str} características")
                

    # ============================================================================
    # PESTAÑA 2: EDA SIMPLE
    # ============================================================================

    with tab2:
        st.header("Exploración de Datos")
        
        if dataset_original is None:
            st.error("❌ No hay dataset disponible para explorar")
            st.info("Por favor, asegúrate de que el archivo 'WA_Fn-UseC_-Telco-Customer-Churn.csv' esté en el directorio")
        else:
            # Información básica del dataset
            st.subheader("📋 Información Básica")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Filas", len(dataset_original))
            
            with col2:
                st.metric("Total Columnas", len(dataset_original.columns))
            
            with col3:
                churn_rate = (dataset_original['Churn'] == 'Yes').mean()
                st.metric("Tasa de Churn", f"{churn_rate:.1%}")
            
            with col4:
                st.metric("Clientes que se van", (dataset_original['Churn'] == 'Yes').sum())
            
            # Gráficos simples
            st.subheader("📈 Gráficos")
            
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                # Gráfico de Churn
                fig1 = px.pie(dataset_original, names='Churn', title="Distribución de Churn")
                st.plotly_chart(fig1, use_container_width=True)
                
                # Gráfico por género
                fig3 = px.histogram(dataset_original, x='gender', color='Churn', 
                                   title="Churn por Género", barmode='group')
                st.plotly_chart(fig3, use_container_width=True)
            
            with col_graf2:
                # Gráfico por contrato
                fig2 = px.histogram(dataset_original, x='Contract', color='Churn', 
                                   title="Churn por Tipo de Contrato", barmode='group')
                st.plotly_chart(fig2, use_container_width=True)
                
                # Gráfico de cargos mensuales
                fig4 = px.box(dataset_original, x='Churn', y='MonthlyCharges', 
                             title="Cargos Mensuales por Churn")
                st.plotly_chart(fig4, use_container_width=True)
            
            # Mostrar los primeros datos
            st.subheader("Primeras 10 Filas del Dataset")
            st.dataframe(dataset_original.head(10), use_container_width=True)

    # PESTAÑA 3: DATOS LIMPIOS

    with tab3:
        st.header("Limpieza de Datos")
        
        if dataset_original is None:
            st.error("❌ No hay dataset disponible para mostrar el proceso de limpieza")
        else:
 
            # Gráfico de correlación con Churn
            st.subheader("Correlación con Churn")
            
            # Preparar datos para correlación
            df_corr = dataset_original.copy()
            
            # Convertir variables categóricas a numéricas para correlación
            df_corr['Churn_num'] = df_corr['Churn'].map({'No': 0, 'Yes': 1})
            df_corr['gender_num'] = df_corr['gender'].map({'Male': 1, 'Female': 0})
            df_corr['Partner_num'] = df_corr['Partner'].map({'Yes': 1, 'No': 0})
            df_corr['Dependents_num'] = df_corr['Dependents'].map({'Yes': 1, 'No': 0})
            df_corr['PhoneService_num'] = df_corr['PhoneService'].map({'Yes': 1, 'No': 0})
            df_corr['PaperlessBilling_num'] = df_corr['PaperlessBilling'].map({'Yes': 1, 'No': 0})
            
            # Convertir TotalCharges a numérico
            df_corr['TotalCharges'] = pd.to_numeric(df_corr['TotalCharges'], errors='coerce').fillna(0)
            
            # Seleccionar solo columnas numéricas para correlación
            columnas_numericas = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                                 'Churn_num', 'gender_num', 'Partner_num', 'Dependents_num', 
                                 'PhoneService_num', 'PaperlessBilling_num']
            
            # Calcular matriz de correlación
            correlacion = df_corr[columnas_numericas].corr()
            
            # Crear heatmap con plotly
            fig_heatmap = px.imshow(
                correlacion,
                text_auto=True,
                color_continuous_scale='RdBu',
                aspect='auto',
                title='Matriz de Correlación'
            )
            
            fig_heatmap.update_layout(
                width=800,
                height=600,
                xaxis_title="Variables",
                yaxis_title="Variables"
            )
            
            # Rotar etiquetas del eje x
            fig_heatmap.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            # Mostrar resultados de la limpieza
            st.subheader("Resultados de la Limpieza")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Datos Originales:**")
                st.write(f"- Filas: {len(dataset_original)}")
                st.write(f"- Columnas: {len(dataset_original.columns)}")
                st.write(f"- Incluye customerID: {'Sí' if 'customerID' in dataset_original.columns else 'No'}")
            
            with col2:
                if X_limpio is not None and y_limpio is not None:
                    st.write("**Datos Limpios:**")
                    st.write(f"- Filas en X: {len(X_limpio)}")
                    st.write(f"- Columnas en X: {len(X_limpio.columns)}")
                    st.write(f"- Filas en y: {len(y_limpio)}")
                else:
                    st.write("**Datos Limpios:**")
                    st.write("- Error en la limpieza")
            
            # Mostrar datos limpios
            if X_limpio is not None and y_limpio is not None:
                col_x, col_y = st.columns(2)
                
                with col_x:
                    st.subheader("Características (X)")
                    st.dataframe(X_limpio.head(10), use_container_width=True)
                
                with col_y:
                    st.subheader("Variable Objetivo (y)")
                    df_y = pd.DataFrame({
                        'y (limpio)': y_limpio.head(10),
                        'Churn (original)': dataset_original['Churn'].head(10)
                    })
                    st.dataframe(df_y, use_container_width=True)
#-----------------------------------------------
    # ============================================================================
    # PESTAÑA 4: MÉTRICAS Y RENDIMIENTO
    # ============================================================================

    # ============================================================================
    # PESTAÑA 4: MÉTRICAS Y RENDIMIENTO CON VALORES PREDEFINIDOS
    # ============================================================================

    with tab4:
        st.header("📈 Métricas y Rendimiento de Modelos")
        
        if total_modelos == 0:
            st.error("❌ No hay modelos disponibles para analizar")
            st.info("Por favor, asegúrate de que los archivos .pkl de los modelos estén en el directorio")
        else:
            # Datos predefinidos basados en las imágenes
            metricas_7_features = {
                'Stacking Diverse': {
                    'Accuracy': 0.795,
                    'AUC': 0.84,
                    'F1-Score': 0.575,
                    'Velocidad_ms': 12.5
                },
                'Logistic Regression': {
                    'Accuracy': 0.775,
                    'AUC': 0.825,
                    'F1-Score': 0.585,
                    'Velocidad_ms': 2.1
                },
                'Voting Classifier': {
                    'Accuracy': 0.785,
                    'AUC': 0.84,
                    'F1-Score': 0.570,
                    'Velocidad_ms': 8.7
                }
            }
            
            metricas_19_features = {
                'Stacking Diverse': {
                    'Accuracy': 0.805,
                    'AUC': 0.845,
                    'F1-Score': 0.60,
                    'Velocidad_ms': 15.8
                },
                'Logistic Regression': {
                    'Accuracy': 0.74,
                    'AUC': 0.765,
                    'F1-Score': 0.52,
                    'Velocidad_ms': 3.2
                },
                'Voting Classifier': {
                    'Accuracy': 0.79,
                    'AUC': 0.84,
                    'F1-Score': 0.58,
                    'Velocidad_ms': 11.4
                }
            }
            
            # Selector de tipo de características
            st.subheader("🔧 Configuración de Análisis")
            tipo_analisis = st.selectbox(
                "Selecciona el tipo de características:",
                ["7 características más importantes", "Todas las características (19)"]
            )
            
            usar_7_features = "7 características" in tipo_analisis
            metricas_actuales = metricas_7_features if usar_7_features else metricas_19_features
            num_features = "7" if usar_7_features else "19"
            
            st.info(f"📊 Mostrando métricas para modelos con **{num_features} características**")
            
            # Tabla de métricas comparativa
            st.subheader("📊 Tabla Comparativa de Métricas")
            
            # Preparar datos para la tabla
            datos_tabla = []
            for modelo, metricas in metricas_actuales.items():
                datos_tabla.append({
                    'Modelo': modelo,
                    'Accuracy': f"{metricas['Accuracy']:.1%}",
                    'AUC': f"{metricas['AUC']:.1%}",
                    'F1-Score': f"{metricas['F1-Score']:.1%}",
                    'Velocidad (ms)': f"{metricas['Velocidad_ms']:.1f}",
                    'Features': num_features
                })
            
            df_metricas = pd.DataFrame(datos_tabla)
            st.dataframe(df_metricas, use_container_width=True)
            
            # KPIs destacados
            st.subheader("🏆 Métricas Destacadas")
            
            # Encontrar el mejor modelo por métrica
            mejor_accuracy = max(metricas_actuales.items(), key=lambda x: x[1]['Accuracy'])
            mejor_auc = max(metricas_actuales.items(), key=lambda x: x[1]['AUC'])
            mejor_f1 = max(metricas_actuales.items(), key=lambda x: x[1]['F1-Score'])
            mas_rapido = min(metricas_actuales.items(), key=lambda x: x[1]['Velocidad_ms'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🎯 Mejor Accuracy", 
                    f"{mejor_accuracy[1]['Accuracy']:.1%}",
                    delta=f"{mejor_accuracy[0]}"
                )
            
            with col2:
                st.metric(
                    "📊 Mejor AUC", 
                    f"{mejor_auc[1]['AUC']:.1%}",
                    delta=f"{mejor_auc[0]}"
                )
            
            with col3:
                st.metric(
                    "⚖️ Mejor F1-Score", 
                    f"{mejor_f1[1]['F1-Score']:.1%}",
                    delta=f"{mejor_f1[0]}"
                )
            
            with col4:
                st.metric(
                    "⚡ Más Rápido", 
                    f"{mas_rapido[1]['Velocidad_ms']:.1f} ms",
                    delta=f"{mas_rapido[0]}"
                )
            
            # Gráficos comparativos
            st.subheader("📈 Gráficos Comparativos")
            
            # Preparar datos para gráficos
            modelos = list(metricas_actuales.keys())
            accuracy_vals = [metricas_actuales[m]['Accuracy'] for m in modelos]
            auc_vals = [metricas_actuales[m]['AUC'] for m in modelos]
            f1_vals = [metricas_actuales[m]['F1-Score'] for m in modelos]
            velocidad_vals = [metricas_actuales[m]['Velocidad_ms'] for m in modelos]
            
            # Gráfico de barras principal (como en la imagen)
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                fig_metricas = go.Figure()
                
                fig_metricas.add_trace(go.Bar(
                    name='Accuracy',
                    x=modelos,
                    y=accuracy_vals,
                    marker_color='steelblue'
                ))
                
                fig_metricas.add_trace(go.Bar(
                    name='AUC',
                    x=modelos,
                    y=auc_vals,
                    marker_color='orange'
                ))
                
                fig_metricas.add_trace(go.Bar(
                    name='F1-Score',
                    x=modelos,
                    y=f1_vals,
                    marker_color='green'
                ))
                
                fig_metricas.update_layout(
                    title=f'Comparación de Métricas ({num_features} características)',
                    xaxis_title='Modelos',
                    yaxis_title='Valor de Métrica',
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig_metricas, use_container_width=True)
            
            with col_graf2:
                # Gráfico de velocidad
                fig_velocidad = px.bar(
                    x=modelos,
                    y=velocidad_vals,
                    title=f'Velocidad de Ejecución ({num_features} características)',
                    labels={'x': 'Modelos', 'y': 'Tiempo (ms)'},
                    color=velocidad_vals,
                    color_continuous_scale='Reds'
                )
                
                fig_velocidad.update_layout(height=500)
                st.plotly_chart(fig_velocidad, use_container_width=True)
            
            # Gráfico radar para comparación integral
            st.subheader("🕸️ Comparación Integral (Gráfico Radar)")
            
            # Normalizar métricas para el radar (0-1)
            accuracy_norm = accuracy_vals
            auc_norm = auc_vals
            f1_norm = f1_vals
            # Invertir velocidad (menor es mejor) y normalizar
            velocidad_norm = [1 - (v - min(velocidad_vals)) / (max(velocidad_vals) - min(velocidad_vals)) for v in velocidad_vals]
            
            fig_radar = go.Figure()
            
            categorias = ['Accuracy', 'AUC', 'F1-Score', 'Velocidad']
            
            for i, modelo in enumerate(modelos):
                valores = [accuracy_norm[i], auc_norm[i], f1_norm[i], velocidad_norm[i]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=valores,
                    theta=categorias,
                    fill='toself',
                    name=modelo
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title=f"Comparación Integral de Modelos ({num_features} características)"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Análisis y recomendaciones
            st.subheader("💡 Análisis y Recomendaciones")
            
            col_analisis1, col_analisis2 = st.columns(2)
            
            with col_analisis1:
                st.success(f"""
                **🏆 Modelo Recomendado: {mejor_accuracy[0]}**
                
                **Fortalezas:**
                - Accuracy: {mejor_accuracy[1]['Accuracy']:.1%}
                - AUC: {mejor_accuracy[1]['AUC']:.1%}
                - F1-Score: {mejor_accuracy[1]['F1-Score']:.1%}
                - Velocidad: {mejor_accuracy[1]['Velocidad_ms']:.1f} ms
                
                **Mejor para:** Predicciones en producción donde la precisión es crítica
                """)
            
            with col_analisis2:
                st.info(f"""
                **⚡ Alternativa Rápida: {mas_rapido[0]}**
                
                **Fortalezas:**
                - Velocidad: {mas_rapido[1]['Velocidad_ms']:.1f} ms (más rápido)
                - Accuracy: {mas_rapido[1]['Accuracy']:.1%}
                - Simplicidad de implementación
                
                **Mejor para:** Aplicaciones en tiempo real con muchas predicciones
                """)
            
            # Comparación entre 7 vs 19 características
            st.subheader("🔄 Comparación: 7 vs 19 Características")
            
            if st.button("Ver comparación detallada 7 vs 19 características"):
                # Crear tabla comparativa
                comparacion_data = []
                
                for modelo in ['Stacking Diverse', 'Logistic Regression', 'Voting Classifier']:
                    if modelo in metricas_7_features and modelo in metricas_19_features:
                        comparacion_data.append({
                            'Modelo': modelo,
                            'Accuracy (7)': f"{metricas_7_features[modelo]['Accuracy']:.1%}",
                            'Accuracy (19)': f"{metricas_19_features[modelo]['Accuracy']:.1%}",
                            'AUC (7)': f"{metricas_7_features[modelo]['AUC']:.1%}",
                            'AUC (19)': f"{metricas_19_features[modelo]['AUC']:.1%}",
                            'Velocidad (7)': f"{metricas_7_features[modelo]['Velocidad_ms']:.1f} ms",
                            'Velocidad (19)': f"{metricas_19_features[modelo]['Velocidad_ms']:.1f} ms"
                        })
                
                df_comparacion = pd.DataFrame(comparacion_data)
                st.dataframe(df_comparacion, use_container_width=True)
                
                st.markdown("""
                **📋 Conclusiones:**
                - **19 características:** Mejor accuracy general pero más lento
                - **7 características:** Más rápido y eficiente con accuracy comparable
                - **Recomendación:** Usar 7 características para producción, 19 para análisis detallado
                """)
    
    # ============================================================================
    # PESTAÑA 5: DASHBOARD SIMPLE
    # ============================================================================

    with tab5:
        st.header("💡 Dashboard Ejecutivo Simple")
        
        if dataset_original is None:
            st.error("❌ No hay dataset disponible para el dashboard")
            st.info("Por favor, asegúrate de que el archivo 'WA_Fn-UseC_-Telco-Customer-Churn.csv' esté en el directorio")
        else:
            # KPIs principales
            st.subheader("📊 Indicadores Clave")
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_clientes = len(dataset_original)
            churn_rate = (dataset_original['Churn'] == 'Yes').mean()
            clientes_churn = (dataset_original['Churn'] == 'Yes').sum()
            ingreso_promedio = dataset_original['MonthlyCharges'].mean()
            
            with col1:
                st.metric("👥 Total Clientes", f"{total_clientes:,}")
            
            with col2:
                st.metric("📈 Tasa de Churn", f"{churn_rate:.1%}")
            
            with col3:
                st.metric("💰 Ingreso Mensual Promedio", f"${ingreso_promedio:.2f}")
            
            with col4:
                if total_modelos > 0:
                    st.metric("⭐ Modelos Disponibles", total_modelos)
                else:
                    st.metric("⭐ Modelos Disponibles", "0")
            
            # Recomendaciones simples
            st.subheader("💡 Recomendaciones Principales")
            
            col_rec1, col_rec2 = st.columns(2)
            
            # Calcular estadísticas reales del dataset
            churn_by_contract = dataset_original.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
            churn_by_internet = dataset_original.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean())
            
            with col_rec1:
                st.info(f"""
                **🔴 Clientes de Alto Riesgo:**
                
                - Contratos mes-a-mes ({churn_by_contract.get('Month-to-month', 0):.1%} abandono)
                - Servicios de fibra óptica ({churn_by_internet.get('Fiber optic', 0):.1%} abandono)  
                - Clientes nuevos (<12 meses)
                - Pago con cheque electrónico
                
                **Acción:** Contactar inmediatamente y ofrecer incentivos
                """)
            
            with col_rec2:
                st.success(f"""
                **🟢 Estrategias de Retención:**
                
                - Promover contratos de 2 años (solo {churn_by_contract.get('Two year', 0):.1%} abandono)
                - Ofrecer servicios de seguridad online
                - Programas de fidelidad para nuevos clientes
                - Mejorar métodos de pago
                
                **Resultado:** Reducción estimada del 15% en churn
                """)
            
            # Gráfico simple de insights
            st.subheader("📈 Insights Principales")
            
            # Datos reales para el gráfico
            datos_insight = pd.DataFrame({
                'Tipo de Contrato': churn_by_contract.index,
                'Tasa de Churn': churn_by_contract.values * 100,
                'Número de Clientes': dataset_original['Contract'].value_counts()[churn_by_contract.index].values
            })
            
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                fig_insight1 = px.bar(datos_insight, x='Tipo de Contrato', y='Tasa de Churn',
                                     title="Tasa de Churn por Tipo de Contrato (%)",
                                     color='Tasa de Churn', color_continuous_scale='reds')
                st.plotly_chart(fig_insight1, use_container_width=True)
            
            with col_graf2:
                fig_insight2 = px.pie(datos_insight, values='Número de Clientes', names='Tipo de Contrato',
                                     title="Distribución de Clientes por Contrato")
                st.plotly_chart(fig_insight2, use_container_width=True)
            
            # Resumen final
            st.subheader("📋 Resumen Ejecutivo")
            
            st.markdown(f"""
            **🎯 Situación Actual:**
            - {churn_rate:.1%} de clientes abandonan ({clientes_churn:,} de {total_clientes:,})
            - Mayor riesgo en contratos mes-a-mes y fibra óptica
            - {"Modelos ML disponibles para predicciones" if total_modelos > 0 else "Sin modelos ML disponibles"}
            
            **🚀 Oportunidades:**
            - Migrar clientes a contratos largos puede reducir churn significativamente
            - Programas de retención pueden generar grandes ahorros
            - {"Predicción temprana permite intervención proactiva" if total_modelos > 0 else "Cargar modelos ML para habilitar predicciones"}
            
            **✅ Próximos Pasos:**
            1. {"Implementar alertas automáticas para clientes de alto riesgo" if total_modelos > 0 else "Cargar modelos ML para habilitar predicciones automáticas"}
            2. Diseñar campañas específicas por segmento de cliente
            3. Monitorear métricas semanalmente para ajustar estrategias
            """)

    # ============================================================================
    # INFORMACIÓN ADICIONAL AL FINAL
    # ============================================================================

# Información del desarrollador
st.markdown("---")
st.markdown("**Realizado por Diego Aleman**")
st.markdown("**Docente: PhD Juan Astudillo**")
st.markdown("**Maestria en Ciencia de Datos**")
