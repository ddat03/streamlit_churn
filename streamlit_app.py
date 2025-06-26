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

    with tab4:
        st.header("📈 Métricas y Rendimiento de Modelos")
        
        if total_modelos == 0:
            st.error("❌ No hay modelos disponibles para analizar")
            st.info("Por favor, asegúrate de que los archivos .pkl de los modelos estén en el directorio")
        elif X_limpio is None or y_limpio is None:
            st.error("❌ No hay datos limpios disponibles para calcular métricas")
            st.info("Se necesita el dataset para calcular métricas reales")
        else:
            # Dividir datos en train/test
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
            from sklearn.preprocessing import LabelEncoder
            
            # Preparar los datos exactamente como fueron entrenados los modelos
            X_procesado = X_limpio.copy()
            
            # Mostrar información de debug del dataset
            st.write("**🔍 Información del Dataset:**")
            st.write(f"- Shape original: {X_procesado.shape}")
            st.write(f"- Columnas: {list(X_procesado.columns)}")
            
            # Codificar variables categóricas
            le_dict = {}
            categorical_columns = X_procesado.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                le = LabelEncoder()
                X_procesado[col] = le.fit_transform(X_procesado[col].astype(str))
                le_dict[col] = le
            
            st.write(f"- Variables categóricas codificadas: {list(categorical_columns)}")
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                X_procesado, y_limpio, test_size=0.2, random_state=42, stratify=y_limpio
            )
            
            st.info(f"📊 Datos divididos: {len(X_train)} entrenamiento, {len(X_test)} prueba")
            st.write(f"**Distribución y_test:** No Churn: {(y_test == 0).sum()}, Churn: {(y_test == 1).sum()}")
            
            # Crear lista de combinaciones disponibles
            combinaciones_disponibles = []
            for modelo_base, variantes in modelos_disponibles.items():
                for num_features in variantes.keys():
                    combinaciones_disponibles.append(f"{modelo_base} ({num_features} características)")
            
            if len(combinaciones_disponibles) > 0:
                modelo_analizar = st.selectbox("Selecciona un modelo para analizar:", 
                                              combinaciones_disponibles)
                
                # Extraer información del modelo
                if " (7 características)" in modelo_analizar:
                    modelo_base = modelo_analizar.replace(" (7 características)", "")
                    num_features_str = '7'
                    usar_7_features = True
                else:
                    modelo_base = modelo_analizar.replace(" (19 características)", "")
                    num_features_str = '19'
                    usar_7_features = False
                
                # Cargar el modelo específico
                modelo, archivo_modelo = cargar_modelo_especifico(modelo_base, num_features_str, modelos_disponibles)
                
                if modelo is not None:
                    st.success(f"✅ Modelo cargado: {archivo_modelo}")
                    
                    try:
                        # Preparar datos según el número de features
                        if usar_7_features:
                            # Para 7 features: usar las columnas más importantes que existan
                            important_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 
                                                'PaymentMethod', 'InternetService', 'gender']
                            
                            # Verificar qué columnas están disponibles
                            available_features = [col for col in important_features if col in X_test.columns]
                            st.write(f"**🔍 Features para modelo de 7:** {available_features}")
                            
                            if len(available_features) >= 7:
                                X_test_modelo = X_test[available_features[:7]].values
                            else:
                                # Usar las primeras 7 columnas numéricas
                                numeric_cols = X_test.select_dtypes(include=[np.number]).columns[:7]
                                X_test_modelo = X_test[numeric_cols].values
                                st.warning(f"⚠️ Usando columnas numéricas: {list(numeric_cols)}")
                        else:
                            # Para 19 features: usar todas las columnas
                            if X_test.shape[1] >= 19:
                                X_test_modelo = X_test.iloc[:, :19].values
                            else:
                                X_test_modelo = X_test.values
                                st.warning(f"⚠️ Solo {X_test.shape[1]} columnas disponibles, esperadas 19")
                        
                        st.write(f"**🔍 Shape datos para modelo:** {X_test_modelo.shape}")
                        st.write(f"**🔍 Modelo esperaba:** {num_features_str} características")
                        
                        # Hacer predicciones reales
                        y_pred = modelo.predict(X_test_modelo)
                        y_pred_proba = modelo.predict_proba(X_test_modelo)[:, 1]
                        
                        # Debug de predicciones
                        st.write(f"**🔍 Predicciones:** No Churn: {(y_pred == 0).sum()}, Churn: {(y_pred == 1).sum()}")
                        st.write(f"**🔍 Rango probabilidades:** {y_pred_proba.min():.3f} - {y_pred_proba.max():.3f}")
                        
                        # Calcular métricas reales con manejo de errores
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # F1-Score con manejo de división por cero
                        try:
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            if f1 == 0:
                                st.warning("⚠️ F1-Score es 0 - el modelo predice solo una clase")
                                # Verificar si el modelo está prediciendo solo una clase
                                unique_preds = np.unique(y_pred)
                                st.write(f"**🔍 Clases predichas:** {unique_preds}")
                        except Exception as e:
                            st.error(f"Error calculando F1-Score: {e}")
                            f1 = 0.0
                        
                        # AUC con manejo de errores
                        try:
                            auc = roc_auc_score(y_test, y_pred_proba)
                        except Exception as e:
                            st.error(f"Error calculando AUC: {e}")
                            auc = 0.0
                        
                        # Métricas de rendimiento
                        st.subheader("⚡ Métricas de Rendimiento")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            datos_prueba = np.random.random((1, X_test_modelo.shape[1]))
                            tiempo_ms = medir_tiempo_prediccion(modelo, datos_prueba, repeticiones=50)
                            st.metric("🕐 Tiempo de Predicción", f"{tiempo_ms:.2f} ms")
                        
                        with col2:
                            st.metric("🔢 Número de Features", X_test_modelo.shape[1])
                        
                        # Métricas de precisión REALES
                        st.subheader("🎯 Métricas de Precisión (REALES)")
                        
                        col4, col5, col6 = st.columns(3)
                        
                        with col4:
                            st.metric("🎯 Accuracy", f"{accuracy:.1%}")
                            if accuracy < 0.5:
                                st.warning("⚠️ Accuracy muy baja - revisar datos")
                        
                        with col5:
                            st.metric("⚖️ F1-Score", f"{f1:.1%}")
                            if f1 == 0:
                                st.error("❌ F1-Score = 0 - Problema con las predicciones")
                        
                        with col6:
                            st.metric("📊 AUC", f"{auc:.1%}")
                            if auc < 0.5:
                                st.warning("⚠️ AUC < 0.5 - Modelo peor que aleatorio")
                        
                        # Matriz de confusión
                        st.subheader("📊 Matriz de Confusión")
                        
                        cm = confusion_matrix(y_test, y_pred)
                        
                        # Mostrar matriz de confusión en números
                        st.write("**Valores de la matriz:**")
                        cm_df = pd.DataFrame(cm, 
                                           index=['Real: No Churn', 'Real: Churn'],
                                           columns=['Pred: No Churn', 'Pred: Churn'])
                        st.dataframe(cm_df)
                        
                        # Gráfico de matriz de confusión
                        fig_cm = px.imshow(
                            cm,
                            text_auto=True,
                            color_continuous_scale='Blues',
                            title='Matriz de Confusión',
                            labels=dict(x="Predicción", y="Real")
                        )
                        
                        fig_cm.update_xaxes(tickvals=[0, 1], ticktext=['No Churn', 'Churn'])
                        fig_cm.update_yaxes(tickvals=[0, 1], ticktext=['No Churn', 'Churn'])
                        
                        st.plotly_chart(fig_cm, use_container_width=True)
                        
                        # Reporte de clasificación
                        st.subheader("📋 Reporte de Clasificación")
                        
                        try:
                            report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], output_dict=True)
                            df_report = pd.DataFrame(report).transpose()
                            df_report = df_report.round(3)
                            st.dataframe(df_report, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generando reporte: {e}")
                        
                        # Comparación de todos los modelos
                        st.subheader("📊 Comparación de Todos los Modelos (Métricas Reales)")
                        
                        datos_comparacion = []
                        
                        with st.spinner("Calculando métricas reales para todos los modelos..."):
                            for modelo_combo in combinaciones_disponibles:
                                try:
                                    # Extraer información del modelo
                                    if " (7 características)" in modelo_combo:
                                        modelo_base_comp = modelo_combo.replace(" (7 características)", "")
                                        num_features_comp = '7'
                                        usar_7_features_comp = True
                                    else:
                                        modelo_base_comp = modelo_combo.replace(" (19 características)", "")
                                        num_features_comp = '19'
                                        usar_7_features_comp = False
                                    
                                    # Cargar modelo
                                    modelo_comp, archivo_comp = cargar_modelo_especifico(modelo_base_comp, num_features_comp, modelos_disponibles)
                                    
                                    if modelo_comp is not None:
                                        # Preparar datos específicos para este modelo
                                        if usar_7_features_comp:
                                            if len(available_features) >= 7:
                                                X_test_comp = X_test[available_features[:7]].values
                                            else:
                                                numeric_cols = X_test.select_dtypes(include=[np.number]).columns[:7]
                                                X_test_comp = X_test[numeric_cols].values
                                        else:
                                            if X_test.shape[1] >= 19:
                                                X_test_comp = X_test.iloc[:, :19].values
                                            else:
                                                X_test_comp = X_test.values
                                        
                                        # Calcular métricas
                                        y_pred_comp = modelo_comp.predict(X_test_comp)
                                        y_pred_proba_comp = modelo_comp.predict_proba(X_test_comp)[:, 1]
                                        
                                        accuracy_comp = accuracy_score(y_test, y_pred_comp)
                                        f1_comp = f1_score(y_test, y_pred_comp, zero_division=0)
                                        
                                        try:
                                            auc_comp = roc_auc_score(y_test, y_pred_proba_comp)
                                        except:
                                            auc_comp = 0.0
                                        
                                        # Métricas de rendimiento
                                        datos_prueba_comp = np.random.random((1, X_test_comp.shape[1]))
                                        tiempo_comp = medir_tiempo_prediccion(modelo_comp, datos_prueba_comp, repeticiones=20)
                                        
                                        datos_comparacion.append({
                                            'Modelo': modelo_combo,
                                            'Accuracy': f"{accuracy_comp:.1%}",
                                            'F1-Score': f"{f1_comp:.1%}",
                                            'AUC': f"{auc_comp:.1%}",
                                            'Tiempo (ms)': f"{tiempo_comp:.2f}",
                                            'Features': num_features_comp,
                                            'Accuracy_num': accuracy_comp * 100
                                        })
                                except Exception as e:
                                    st.warning(f"Error con {modelo_combo}: {str(e)}")
                                    # Agregar fila con errores para mantener la tabla completa
                                    datos_comparacion.append({
                                        'Modelo': modelo_combo,
                                        'Accuracy': "Error",
                                        'F1-Score': "Error", 
                                        'AUC': "Error",
                                        'Tiempo (ms)': "Error",
                                        'Features': num_features_comp,
                                        'Accuracy_num': 0
                                    })
                        
                        if len(datos_comparacion) > 0:
                            df_comparacion = pd.DataFrame(datos_comparacion)
                            st.dataframe(df_comparacion.drop('Accuracy_num', axis=1), use_container_width=True)
                            
                            # Gráfico comparativo (solo modelos sin error)
                            df_grafico = df_comparacion[df_comparacion['Accuracy'] != 'Error'].copy()
                            if len(df_grafico) > 1:
                                df_grafico['Accuracy_num'] = df_grafico['Accuracy_num'].astype(float)
                                fig_comp = px.bar(df_grafico, x='Modelo', y='Accuracy_num', 
                                                 title="Comparación de Accuracy REAL (%)",
                                                 color='Features')
                                fig_comp.update_layout(xaxis_tickangle=45)
                                st.plotly_chart(fig_comp, use_container_width=True)
                        else:
                            st.warning("No se pudieron calcular métricas para ningún modelo")
                    
                    except Exception as e:
                        st.error(f"❌ Error calculando métricas: {str(e)}")
                        st.write("**🔍 Información de debug completa:**")
                        st.write(f"- Shape de X_test: {X_test.shape}")
                        st.write(f"- Columnas de X_test: {list(X_test.columns)}")
                        st.write(f"- Tipo de y_test: {type(y_test)}")
                        st.write(f"- Valores únicos en y_test: {np.unique(y_test)}")
                        st.write(f"- Modelo: {modelo_base} ({num_features_str} características)")
                        st.write(f"- Archivo: {archivo_modelo}")
                        st.write(f"- Archivo existe: {os.path.exists(archivo_modelo)}")
                        
                else:
                    st.error(f"❌ No se pudo cargar el modelo {modelo_base} con {num_features_str} características")
                    st.write(f"**🔍 Archivo esperado:** {modelos_disponibles.get(modelo_base, {}).get(num_features_str, 'No encontrado')}")
            else:
                st.warning("No hay modelos disponibles para analizar")
    
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
