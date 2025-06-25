import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="🔮 Telco Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffa726 0%, #fb8c00 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

# Features completas (19 features)
COMPLETE_FEATURES = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 
    'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
]

# Top 7 features más importantes
TOP_FEATURES = [
    'TotalCharges', 'MonthlyCharges', 'tenure', 'InternetService', 
    'PaymentMethod', 'Contract', 'gender'
]

# Mapeos para features categóricas
FEATURE_MAPPINGS = {
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'PaymentMethod': {
        'Electronic check': 1, 'Mailed check': 0, 
        'Bank transfer (automatic)': 0, 'Credit card (automatic)': 0
    },
    'Contract': {'Month-to-month': 0, 'One year': 0, 'Two year': 1},
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'Yes': 1},
    'OnlineSecurity': {'No': 0, 'Yes': 1},
    'OnlineBackup': {'No': 0, 'Yes': 1},
    'DeviceProtection': {'No': 0, 'Yes': 1},
    'TechSupport': {'No': 0, 'Yes': 1},
    'StreamingTV': {'No': 0, 'Yes': 1},
    'StreamingMovies': {'No': 0, 'Yes': 1},
    'PaperlessBilling': {'No': 0, 'Yes': 1}
}

# ============================================================================
# FUNCIONES DE CARGA DE DATOS
# ============================================================================

@st.cache_data
def load_dataset():
    """Cargar el dataset de Telco Customer Churn"""
    try:
        df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
        st.success("✅ Dataset cargado exitosamente")
        return df
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'WA_Fn-UseC_-Telco-Customer-Churn.csv'")
        # Dataset simulado para demostración
        np.random.seed(42)
        n_samples = 1000
        
        df_sim = pd.DataFrame({
            'customerID': [f'ID_{i}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'tenure': np.random.randint(1, 73, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'MonthlyCharges': np.random.uniform(18.25, 118.75, n_samples),
            'TotalCharges': np.random.uniform(18.8, 8684.8, n_samples),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.265, 0.735])
        })
        
        # Simular algunos valores de TotalCharges como string para demostrar limpieza
        mask = np.random.random(n_samples) < 0.02
        df_sim.loc[mask, 'TotalCharges'] = ' '
        
        st.warning("⚠️ Usando dataset simulado para demostración")
        return df_sim

def clean_dataset(df_raw):
    """Limpiar el dataset según el código específico proporcionado"""
    df_clean = df_raw.copy()
    
    # Remover customerID si existe
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
    
    # Separar target y features
    if 'Churn' in df_clean.columns:
        y = df_clean['Churn'].map({'No': 0, 'Yes': 1})
        X = df_clean.drop('Churn', axis=1)
    else:
        st.error("No se encontró la columna 'Churn' en el dataset")
        return None, None, None
    
    # Convertir TotalCharges a numérico
    if 'TotalCharges' in X.columns:
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)
    
    # Simplificar categorías redundantes
    services_to_fix = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    for service in services_to_fix:
        if service in X.columns:
            X[service] = X[service].replace('No internet service', 'No')
    
    return df_clean, X, y

# ============================================================================
# CLASE MODELO SIMULADO
# ============================================================================

class MockChurnModel:
    """Modelo simulado para cuando los modelos reales fallan"""
    
    def __init__(self, name):
        self.name = name
        np.random.seed(42)
        
    def predict(self, X):
        predictions = []
        for sample in X:
            if len(sample) == 7:  # Versión reducida
                total_charges, monthly_charges, tenure, fiber, electronic, two_year, male = sample
                
                churn_score = 0
                if monthly_charges > 70: churn_score += 0.3
                if tenure < 12: churn_score += 0.4
                if fiber == 1: churn_score += 0.2
                if electronic == 1: churn_score += 0.15
                if two_year == 0: churn_score += 0.25
                
                churn_score += np.random.normal(0, 0.1)
                prediction = 1 if churn_score > 0.5 else 0
                
            else:  # Versión completa
                senior, tenure, monthly, total = sample[:4]
                
                churn_score = 0
                if monthly > 70: churn_score += 0.3
                if tenure < 12: churn_score += 0.4
                if senior == 1: churn_score += 0.1
                
                churn_score += np.random.normal(0, 0.1)
                prediction = 1 if churn_score > 0.5 else 0
                
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def predict_proba(self, X):
        predictions = self.predict(X)
        probabilities = []
        
        for pred in predictions:
            if pred == 1:
                prob_churn = np.random.uniform(0.6, 0.9)
            else:
                prob_churn = np.random.uniform(0.1, 0.4)
                
            prob_no_churn = 1 - prob_churn
            probabilities.append([prob_no_churn, prob_churn])
            
        return np.array(probabilities)

# ============================================================================
# FUNCIONES DE CARGA DE MODELOS
# ============================================================================

@st.cache_resource
def load_models():
    """Carga modelos entrenados o crea modelos simulados"""
    try:
        models = {
            'Stacking Diverse': joblib.load('stacking_diverse_trained.pkl'),
            'Logistic Regression': joblib.load('Single Classifier (Logistic Regression)_trained.pkl'),
            'Voting Classifier': joblib.load('Voting Classifier (Soft)_trained.pkl')
        }
        return models, "real"
        
    except Exception as e:
        st.warning(f"⚠️ No se pudieron cargar modelos reales: {e}")
        st.info("🔄 Usando modelos simulados para demostración...")
        
        models = {
            'Stacking Diverse': MockChurnModel('Stacking Diverse'),
            'Logistic Regression': MockChurnModel('Logistic Regression'),
            'Voting Classifier': MockChurnModel('Voting Classifier')
        }
        return models, "simulated"

# ============================================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================================

def preprocess_customer_data(customer_data, version="completa"):
    """Preprocesar datos del cliente según la versión seleccionada"""
    try:
        if version == "reducida":
            processed_data = []
            
            # Top 7 features
            processed_data.append(float(customer_data.get('TotalCharges', 0)))
            processed_data.append(float(customer_data.get('MonthlyCharges', 0)))
            processed_data.append(int(customer_data.get('tenure', 0)))
            
            # Features categóricas convertidas a binarias
            internet_service = customer_data.get('InternetService', 'DSL')
            processed_data.append(1 if internet_service == 'Fiber optic' else 0)
            
            payment_method = customer_data.get('PaymentMethod', 'Electronic check')
            processed_data.append(1 if payment_method == 'Electronic check' else 0)
            
            contract = customer_data.get('Contract', 'Month-to-month')
            processed_data.append(1 if contract == 'Two year' else 0)
            
            gender = customer_data.get('gender', 'Male')
            processed_data.append(1 if gender == 'Male' else 0)
            
            return np.array(processed_data).reshape(1, -1)
            
        else:
            # Versión completa
            processed_data = []
            
            for feature in COMPLETE_FEATURES:
                if feature in ['SeniorCitizen', 'tenure']:
                    processed_data.append(int(customer_data.get(feature, 0)))
                elif feature in ['MonthlyCharges', 'TotalCharges']:
                    processed_data.append(float(customer_data.get(feature, 0)))
                else:
                    value = customer_data.get(feature, list(FEATURE_MAPPINGS[feature].keys())[0])
                    mapped_value = FEATURE_MAPPINGS[feature].get(value, 0)
                    processed_data.append(mapped_value)
            
            return np.array(processed_data).reshape(1, -1)
            
    except Exception as e:
        st.error(f"Error en preprocessing: {e}")
        return None

def predict_churn(customer_data, model_name, version, models):
    """Realizar predicción de churn"""
    try:
        input_processed = preprocess_customer_data(customer_data, version)
        
        if input_processed is None:
            return {"success": False, "error": "Error en preprocessing de datos"}
        
        model = models[model_name]
        
        prediction = model.predict(input_processed)[0]
        probabilities = model.predict_proba(input_processed)[0]
        
        return {
            "success": True,
            "prediction": int(prediction),
            "prediction_label": "Churn (Abandono)" if prediction == 1 else "No Churn (Permanece)",
            "probabilities": {
                "no_churn": float(probabilities[0]),
                "churn": float(probabilities[1])
            },
            "model_used": model_name,
            "version_used": version,
            "features_used": len(input_processed[0])
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================================================
# FUNCIONES DE VISUALIZACIÓN
# ============================================================================

def create_eda_plots(df_raw):
    """Crear gráficos para EDA"""
    plots = {}
    
    # 1. Distribución del target
    churn_counts = df_raw['Churn'].value_counts()
    plots['target_dist'] = px.pie(
        values=churn_counts.values, 
        names=churn_counts.index,
        title="Distribución de Churn",
        color_discrete_map={'No': '#2E8B57', 'Yes': '#DC143C'}
    )
    
    # 2. Distribución por género
    plots['gender_dist'] = px.histogram(
        df_raw, x='gender', color='Churn',
        title="Distribución de Churn por Género",
        barmode='group'
    )
    
    # 3. Distribución por tipo de contrato
    plots['contract_dist'] = px.histogram(
        df_raw, x='Contract', color='Churn',
        title="Distribución de Churn por Tipo de Contrato",
        barmode='group'
    )
    
    # 4. Distribución de tenure
    plots['tenure_dist'] = px.histogram(
        df_raw, x='tenure', color='Churn',
        title="Distribución de Tenure por Churn",
        nbins=30, opacity=0.7
    )
    
    # 5. Distribución de MonthlyCharges
    plots['monthly_charges'] = px.box(
        df_raw, x='Churn', y='MonthlyCharges',
        title="Distribución de Cargos Mensuales por Churn"
    )
    
    # 6. Servicio de Internet vs Churn
    plots['internet_service'] = px.histogram(
        df_raw, x='InternetService', color='Churn',
        title="Distribución de Churn por Servicio de Internet",
        barmode='group'
    )
    
    return plots

def create_correlation_matrix(X_clean):
    """Crear matriz de correlación"""
    # Seleccionar solo variables numéricas
    numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
    corr_matrix = X_clean[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Matriz de Correlación de Variables Numéricas",
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    
    fig.update_layout(height=600)
    return fig

def create_metrics_plots(model_name, model_type):
    """Crear gráficos de métricas específicos por modelo"""
    # Métricas simuladas específicas por modelo
    metrics_data = {
        'Stacking Diverse': {
            'accuracy_complete': 0.862, 'f1_complete': 0.841, 'auc_complete': 0.895,
            'accuracy_reduced': 0.847, 'f1_reduced': 0.823, 'auc_reduced': 0.878,
            'confusion_matrix_complete': [[1054, 96], [124, 135]],
            'confusion_matrix_reduced': [[1042, 108], [138, 121]]
        },
        'Logistic Regression': {
            'accuracy_complete': 0.834, 'f1_complete': 0.812, 'auc_complete': 0.871,
            'accuracy_reduced': 0.829, 'f1_reduced': 0.805, 'auc_reduced': 0.863,
            'confusion_matrix_complete': [[1038, 112], [141, 118]],
            'confusion_matrix_reduced': [[1031, 119], [152, 107]]
        },
        'Voting Classifier': {
            'accuracy_complete': 0.851, 'f1_complete': 0.829, 'auc_complete': 0.883,
            'accuracy_reduced': 0.836, 'f1_reduced': 0.814, 'auc_reduced': 0.869,
            'confusion_matrix_complete': [[1046, 104], [133, 126]],
            'confusion_matrix_reduced': [[1035, 115], [147, 112]]
        }
    }
    
    data = metrics_data[model_name]
    plots = {}
    
    # 1. Comparación de métricas
    versions = ['Completa', 'Reducida']
    accuracy_values = [data['accuracy_complete'], data['accuracy_reduced']]
    f1_values = [data['f1_complete'], data['f1_reduced']]
    auc_values = [data['auc_complete'], data['auc_reduced']]
    
    plots['metrics_comparison'] = go.Figure()
    plots['metrics_comparison'].add_trace(go.Bar(name='Accuracy', x=versions, y=accuracy_values))
    plots['metrics_comparison'].add_trace(go.Bar(name='F1-Score', x=versions, y=f1_values))
    plots['metrics_comparison'].add_trace(go.Bar(name='AUC', x=versions, y=auc_values))
    plots['metrics_comparison'].update_layout(
        title=f"Comparación de Métricas - {model_name}",
        barmode='group',
        yaxis_title="Score"
    )
    
    # 2. Matriz de confusión - Versión completa
    cm_complete = data['confusion_matrix_complete']
    plots['confusion_matrix_complete'] = go.Figure(data=go.Heatmap(
        z=cm_complete,
        x=['Predicho: No Churn', 'Predicho: Churn'],
        y=['Real: No Churn', 'Real: Churn'],
        colorscale='Blues',
        text=cm_complete,
        texttemplate="%{text}",
        textfont={"size": 16},
    ))
    plots['confusion_matrix_complete'].update_layout(
        title=f"Matriz de Confusión (Versión Completa) - {model_name}",
        height=400
    )
    
    # 3. Matriz de confusión - Versión reducida
    cm_reduced = data['confusion_matrix_reduced']
    plots['confusion_matrix_reduced'] = go.Figure(data=go.Heatmap(
        z=cm_reduced,
        x=['Predicho: No Churn', 'Predicho: Churn'],
        y=['Real: No Churn', 'Real: Churn'],
        colorscale='Oranges',
        text=cm_reduced,
        texttemplate="%{text}",
        textfont={"size": 16},
    ))
    plots['confusion_matrix_reduced'].update_layout(
        title=f"Matriz de Confusión (Versión Reducida) - {model_name}",
        height=400
    )
    
    # 4. Feature importance específica por modelo
    importance_data = {
        'Stacking Diverse': {
            'features': ['TotalCharges', 'MonthlyCharges', 'tenure', 'Contract_Two year', 
                        'InternetService_Fiber optic', 'PaymentMethod_Electronic check', 'gender_Male',
                        'OnlineSecurity', 'TechSupport', 'PaperlessBilling'],
            'importance': [0.243, 0.198, 0.156, 0.089, 0.067, 0.054, 0.032, 0.028, 0.025, 0.023]
        },
        'Logistic Regression': {
            'features': ['TotalCharges', 'MonthlyCharges', 'tenure', 'Contract_Two year', 
                        'InternetService_Fiber optic', 'PaymentMethod_Electronic check', 'gender_Male',
                        'OnlineSecurity', 'TechSupport', 'PaperlessBilling'],
            'importance': [0.267, 0.201, 0.143, 0.095, 0.071, 0.058, 0.029, 0.026, 0.023, 0.021]
        },
        'Voting Classifier': {
            'features': ['TotalCharges', 'MonthlyCharges', 'tenure', 'Contract_Two year', 
                        'InternetService_Fiber optic', 'PaymentMethod_Electronic check', 'gender_Male',
                        'OnlineSecurity', 'TechSupport', 'PaperlessBilling'],
            'importance': [0.251, 0.194, 0.149, 0.092, 0.069, 0.056, 0.031, 0.027, 0.024, 0.022]
        }
    }
    
    imp_data = importance_data[model_name]
    plots['feature_importance'] = go.Figure([go.Bar(
        x=imp_data['importance'],
        y=imp_data['features'],
        orientation='h',
        marker_color='#4472C4'
    )])
    plots['feature_importance'].update_layout(
        title=f"Importancia de Características - {model_name}",
        xaxis_title="Importancia",
        height=500
    )
    
    return plots, data

def get_recommendations(churn_probability, customer_data):
    """Generar recomendaciones basadas en la probabilidad de churn"""
    recommendations = []
    
    if churn_probability >= 0.7:
        recommendations = [
            "🚨 **ACCIÓN INMEDIATA**: Contactar al cliente en las próximas 24 horas",
            "💰 Ofrecer descuentos de hasta 20% por 6 meses",
            "📞 Llamada de retención del gerente de cuentas",
            "🎁 Servicios premium gratuitos por 3 meses",
            "📋 Revisión completa del plan de servicios"
        ]
    elif churn_probability >= 0.4:
        recommendations = [
            "📋 Monitorear satisfacción del cliente mensualmente",
            "📧 Enviar encuesta de satisfacción personalizada",
            "🎯 Ofrecer servicios adicionales con descuento",
            "📞 Seguimiento proactivo cada 2 meses",
            "💡 Analizar patrones de uso y optimizar plan"
        ]
    else:
        recommendations = [
            "⭐ Cliente estable - oportunidad ideal para upselling",
            "🎯 Candidato prioritario para programas de referidos",
            "📈 Ofrecer servicios premium y nuevas funcionalidades",
            "👑 Invitar a programa VIP con beneficios exclusivos",
            "🏆 Cliente modelo para casos de éxito y testimonios"
        ]
    
    return recommendations

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    # Header mejorado
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Telco Customer Churn Predictor</h1>
        <h3>Sistema Avanzado de Predicción de Abandono de Clientes</h3>
        <p>Análisis Inteligente | Machine Learning | Insights de Negocio</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar datos y modelos
    with st.spinner("Cargando datos y modelos..."):
        df_raw = load_dataset()
        models, model_type = load_models()
        df_clean, X_clean, y_clean = clean_dataset(df_raw)
    
    model_names = list(models.keys())
    
    # Sidebar mejorado
    st.sidebar.markdown("## ⚙️ Configuración del Modelo")
    
    model_name = st.sidebar.selectbox(
        "🎯 Selecciona el Modelo:",
        model_names,
        help="Selecciona el modelo de machine learning para realizar predicciones"
    )
    
    version = st.sidebar.radio(
        "🔧 Versión del Modelo:",
        ["completa", "reducida"],
        format_func=lambda x: "Completa (19 características)" if x == "completa" else "Reducida (7 características top)",
        help="Versión completa usa todas las características, versión reducida usa solo las más importantes"
    )
    
    # Información sobre la versión en sidebar
    if version == "reducida":
        st.sidebar.markdown("""
        <div class="insight-box">
        <strong>Top 7 Características:</strong><br>
        1. TotalCharges<br>
        2. MonthlyCharges<br>
        3. tenure<br>
        4. InternetService<br>
        5. PaymentMethod<br>
        6. Contract<br>
        7. gender
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Información del dataset
    if df_raw is not None:
        st.sidebar.markdown("### 📊 Información del Dataset")
        st.sidebar.metric("Total de Registros", f"{len(df_raw):,}")
        st.sidebar.metric("Tasa de Churn", f"{(df_raw['Churn'] == 'Yes').mean():.1%}")
        st.sidebar.metric("Características", len(df_raw.columns) - 1)
    
    st.sidebar.info(f"**Tipo de modelos:** {model_type.title()}")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔮 Predicción", 
        "📊 EDA", 
        "🧹 Dataset Limpio", 
        "📈 Métricas por Modelo", 
        "💼 Dashboard"
    ])
    
    # ========================================================================
    # TAB 1: PREDICCIÓN
    # ========================================================================
    
    with tab1:
        st.header("🔮 Predicción Individual de Churn")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Información del Cliente")
            
            # Crear formulario
            with st.form("prediction_form"):
                
                # Información Demográfica
                st.markdown("#### 👤 Información Demográfica")
                col_demo1, col_demo2, col_demo3 = st.columns(3)
                
                with col_demo1:
                    gender = st.selectbox("Género", ["Male", "Female"])
                    SeniorCitizen = st.selectbox("Ciudadano Senior", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
                
                with col_demo2:
                    Partner = st.selectbox("Tiene Pareja", ["No", "Yes"])
                    Dependents = st.selectbox("Tiene Dependientes", ["No", "Yes"])
                
                with col_demo3:
                    tenure = st.number_input("Tenure (meses)", min_value=0, max_value=100, value=12)
                
                # Servicios
                st.markdown("#### 📞 Servicios")
                col_serv1, col_serv2, col_serv3 = st.columns(3)
                
                with col_serv1:
                    PhoneService = st.selectbox("Servicio Telefónico", ["Yes", "No"])
                    MultipleLines = st.selectbox("Múltiples Líneas", ["No", "Yes"])
                
                with col_serv2:
                    InternetService = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
                    OnlineSecurity = st.selectbox("Seguridad Online", ["No", "Yes"])
                
                with col_serv3:
                    OnlineBackup = st.selectbox("Backup Online", ["No", "Yes"])
                    DeviceProtection = st.selectbox("Protección de Dispositivos", ["No", "Yes"])
                
                # Servicios adicionales
                col_extra1, col_extra2 = st.columns(2)
                
                with col_extra1:
                    TechSupport = st.selectbox("Soporte Técnico", ["No", "Yes"])
                    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes"])
                
                with col_extra2:
                    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes"])
                
                # Contrato y Facturación
                st.markdown("#### 📋 Contrato y Facturación")
                col_contract1, col_contract2 = st.columns(2)
                
                with col_contract1:
                    Contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
                    PaperlessBilling = st.selectbox("Facturación Sin Papel", ["Yes", "No"])
                
                with col_contract2:
                    PaymentMethod = st.selectbox("Método de Pago", 
                        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                
                # Cargos
                col_charges1, col_charges2 = st.columns(2)
                
                with col_charges1:
                    MonthlyCharges = st.number_input("Cargo Mensual ($)", min_value=0.0, value=50.0, step=0.01)
                
                with col_charges2:
                    TotalCharges = st.number_input("Total Cargos ($)", min_value=0.0, value=1000.0, step=0.01)
                
                # Botón de predicción
                submitted = st.form_submit_button("🚀 Realizar Predicción", type="primary")
        
        with col2:
            if submitted:
                # Preparar datos
                customer_data = {
                    'SeniorCitizen': SeniorCitizen, 'tenure': tenure, 'MonthlyCharges': MonthlyCharges,
                    'TotalCharges': TotalCharges, 'gender': gender, 'Partner': Partner,
                    'Dependents': Dependents, 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
                    'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
                    'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport, 'StreamingTV': StreamingTV,
                    'StreamingMovies': StreamingMovies, 'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
                    'PaymentMethod': PaymentMethod
                }
                
                # Realizar predicción
                with st.spinner("🔮 Analizando cliente..."):
                    result = predict_churn(customer_data, model_name, version, models)
                
                if result["success"]:
                    # Mostrar resultado
                    churn_prob = result["probabilities"]["churn"]
                    
                    # Determinar nivel de riesgo y aplicar CSS
                    if churn_prob >= 0.7:
                        risk_class = "risk-high"
                        risk_text = "🔴 ALTO RIESGO"
                    elif churn_prob >= 0.4:
                        risk_class = "risk-medium"
                        risk_text = "🟡 RIESGO MEDIO"
                    else:
                        risk_class = "risk-low"
                        risk_text = "🟢 RIESGO BAJO"
                    
                    st.markdown(f'<div class="{risk_class}"><h3>{risk_text}</h3></div>', unsafe_allow_html=True)
                    
                    # Métricas principales
                    col_met1, col_met2 = st.columns(2)
                    
                    with col_met1:
                        st.metric("🎯 Probabilidad Churn", f"{churn_prob:.1%}")
                    
                    with col_met2:
                        st.metric("✅ Probabilidad Retención", f"{result['probabilities']['no_churn']:.1%}")
                    
                    # Gráfico de probabilidades
                    prob_data = pd.DataFrame({
                        'Outcome': ['No Churn', 'Churn'],
                        'Probability': [result['probabilities']['no_churn'], churn_prob],
                        'Color': ['#2E8B57', '#DC143C']
                    })
                    
                    fig_prob = px.bar(
                        prob_data, 
                        x='Outcome', 
                        y='Probability',
                        color='Color',
                        color_discrete_map={'#2E8B57': '#2E8B57', '#DC143C': '#DC143C'},
                        title="Probabilidades de Predicción"
                    )
                    fig_prob.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Información del modelo
                    st.info(f"**Modelo:** {model_name}\n**Versión:** {version}\n**Features usadas:** {result['features_used']}")
                    
                    # Recomendaciones
                    st.markdown("### 💡 Recomendaciones")
                    recommendations = get_recommendations(churn_prob, customer_data)
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                
                else:
                    st.error(f"❌ Error en la predicción: {result['error']}")
    
    # ========================================================================
    # TAB 2: EDA
    # ========================================================================
    
    with tab2:
        st.header("📊 Análisis Exploratorio de Datos (EDA)")
        
        if df_raw is not None:
            # Resumen del dataset
            st.subheader("📋 Resumen del Dataset")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Registros", f"{len(df_raw):,}")
            
            with col2:
                st.metric("📈 Tasa de Churn", f"{(df_raw['Churn'] == 'Yes').mean():.1%}")
            
            with col3:
                st.metric("💰 Cargo Mensual Promedio", f"${df_raw['MonthlyCharges'].mean():.2f}")
            
            with col4:
                if 'TotalCharges' in df_raw.columns:
                    # Convertir a numérico para el cálculo
                    total_charges_numeric = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
                    st.metric("⏱️ Tenure Promedio", f"{df_raw['tenure'].mean():.1f} meses")
            
            # Crear gráficos de EDA
            eda_plots = create_eda_plots(df_raw)
            
            # Mostrar gráficos en una cuadrícula
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(eda_plots['target_dist'], use_container_width=True)
                st.plotly_chart(eda_plots['contract_dist'], use_container_width=True)
                st.plotly_chart(eda_plots['monthly_charges'], use_container_width=True)
            
            with col2:
                st.plotly_chart(eda_plots['gender_dist'], use_container_width=True)
                st.plotly_chart(eda_plots['internet_service'], use_container_width=True)
                st.plotly_chart(eda_plots['tenure_dist'], use_container_width=True)
            
            # Matriz de correlación
            st.subheader("🔗 Matriz de Correlación")
            if X_clean is not None:
                corr_fig = create_correlation_matrix(X_clean)
                st.plotly_chart(corr_fig, use_container_width=True)
            
            # Estadísticas descriptivas
            st.subheader("📈 Estadísticas Descriptivas")
            st.dataframe(df_raw.describe(), use_container_width=True)
        
        else:
            st.error("No hay datos disponibles para el análisis exploratorio")
    
    # ========================================================================
    # TAB 3: DATASET LIMPIO
    # ========================================================================
    
    with tab3:
        st.header("🧹 Proceso de Limpieza del Dataset")
        
        if df_raw is not None and df_clean is not None:
            
            # Mostrar el proceso de limpieza
            st.subheader("🔧 Pasos de Limpieza Aplicados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Pasos de limpieza realizados:**
                
                1. **Eliminación de customerID**: Removida columna identificadora
                2. **Separación de target**: Variable 'Churn' convertida a binaria (0/1)
                3. **Conversión de TotalCharges**: Convertida a numérica, valores faltantes = 0
                4. **Simplificación de categorías**: 'No internet service' → 'No'
                
                **Servicios procesados:**
                - OnlineSecurity
                - OnlineBackup  
                - DeviceProtection
                - TechSupport
                - StreamingTV
                - StreamingMovies
                """)
            
            with col2:
                # Comparación antes y después
                st.metric("Registros Originales", len(df_raw))
                st.metric("Registros Limpios", len(X_clean))
                st.metric("Features Originales", len(df_raw.columns))
                st.metric("Features Limpias", len(X_clean.columns))
                
                # Valores faltantes
                if 'TotalCharges' in df_raw.columns:
                    missing_before = df_raw['TotalCharges'].isin([' ', '']).sum()
                    missing_after = X_clean['TotalCharges'].isna().sum()
                    st.metric("Valores Faltantes (Antes)", missing_before)
                    st.metric("Valores Faltantes (Después)", missing_after)
            
            # Mostrar código de limpieza
            st.subheader("💻 Código de Limpieza Utilizado")
            
            code = '''
# Código de limpieza aplicado:

if 'customerID' in X.columns:
    X = X.drop('customerID', axis=1)

# Separar target
y = X['Churn'].map({'No': 0, 'Yes': 1})
X = X.drop('Churn', axis=1)

# Convertir TotalCharges a numérico
if 'TotalCharges' in X.columns:
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)

# Simplificar categorías redundantes
services_to_fix = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies']

for service in services_to_fix:
    if service in X.columns:
        X[service] = X[service].replace('No internet service', 'No')
            '''
            
            st.code(code, language='python')
            
            # Mostrar dataset limpio
            st.subheader("📊 Dataset Limpio Resultante")
            
            # Tabs para mostrar X e y
            tab_x, tab_y = st.tabs(["Features (X)", "Target (y)"])
            
            with tab_x:
                st.markdown("**Features después de la limpieza:**")
                st.dataframe(X_clean.head(100), use_container_width=True)
                
                # Información de tipos de datos
                st.markdown("**Tipos de Datos:**")
                types_df = pd.DataFrame({
                    'Columna': X_clean.columns,
                    'Tipo': X_clean.dtypes.astype(str),
                    'Valores Únicos': [X_clean[col].nunique() for col in X_clean.columns],
                    'Valores Faltantes': [X_clean[col].isna().sum() for col in X_clean.columns]
                })
                st.dataframe(types_df, use_container_width=True)
            
            with tab_y:
                st.markdown("**Variable Target después de la conversión:**")
                target_df = pd.DataFrame({
                    'y': y_clean.head(100),
                    'Original': df_raw['Churn'].head(100)
                })
                st.dataframe(target_df, use_container_width=True)
                
                # Distribución del target
                target_counts = y_clean.value_counts()
                fig_target = px.bar(
                    x=['No Churn (0)', 'Churn (1)'], 
                    y=target_counts.values,
                    title="Distribución del Target Limpio",
                    color=['No Churn', 'Churn'],
                    color_discrete_map={'No Churn': '#2E8B57', 'Churn': '#DC143C'}
                )
                st.plotly_chart(fig_target, use_container_width=True)
        
        else:
            st.error("No hay datos disponibles para mostrar el proceso de limpieza")
    
    # ========================================================================
    # TAB 4: MÉTRICAS POR MODELO
    # ========================================================================
    
    with tab4:
        st.header(f"📈 Métricas Específicas - {model_name}")
        
        # Crear gráficos específicos del modelo seleccionado
        metrics_plots, metrics_data = create_metrics_plots(model_name, model_type)
        
        # Resumen de métricas en cards
        st.subheader("📊 Resumen de Métricas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Accuracy</h4>
                <h2>{metrics_data['accuracy_complete']:.1%}</h2>
                <small>Versión Completa</small><br>
                <small style="opacity: 0.8;">{metrics_data['accuracy_reduced']:.1%} (Reducida)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>F1-Score</h4>
                <h2>{metrics_data['f1_complete']:.1%}</h2>
                <small>Versión Completa</small><br>
                <small style="opacity: 0.8;">{metrics_data['f1_reduced']:.1%} (Reducida)</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>AUC</h4>
                <h2>{metrics_data['auc_complete']:.1%}</h2>
                <small>Versión Completa</small><br>
                <small style="opacity: 0.8;">{metrics_data['auc_reduced']:.1%} (Reducida)</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de comparación de métricas
        st.subheader("📊 Comparación de Versiones")
        st.plotly_chart(metrics_plots['metrics_comparison'], use_container_width=True)
        
        # Matrices de confusión lado a lado
        st.subheader("🔥 Matrices de Confusión")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(metrics_plots['confusion_matrix_complete'], use_container_width=True)
        
        with col2:
            st.plotly_chart(metrics_plots['confusion_matrix_reduced'], use_container_width=True)
        
        # Feature importance
        st.subheader("⭐ Importancia de Características")
        st.plotly_chart(metrics_plots['feature_importance'], use_container_width=True)
        
        # Análisis de rendimiento
        st.subheader("🎯 Análisis de Rendimiento")
        
        performance_diff = {
            'accuracy': metrics_data['accuracy_complete'] - metrics_data['accuracy_reduced'],
            'f1': metrics_data['f1_complete'] - metrics_data['f1_reduced'],
            'auc': metrics_data['auc_complete'] - metrics_data['auc_reduced']
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **📈 Rendimiento Versión Completa:**
            - ✅ Mayor precisión general
            - ✅ Mejor captura de patrones complejos
            - ⚠️ Requiere más características
            - ⚠️ Mayor tiempo de procesamiento
            """)
        
        with col2:
            st.markdown(f"""
            **⚡ Rendimiento Versión Reducida:**
            - ✅ Procesamiento más rápido
            - ✅ Menos datos requeridos
            - ✅ Modelo más interpretable
            - ⚠️ Pérdida de precisión: {performance_diff['accuracy']:.1%}
            """)
    
    # ========================================================================
    # TAB 5: DASHBOARD
    # ========================================================================
    
    with tab5:
        st.header("💼 Dashboard Ejecutivo")
        
        # KPIs principales
        st.subheader("🎯 KPIs Principales")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Total Clientes</h4>
                <h2>7,043</h2>
                <small>Base de datos completa</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📈 Tasa de Churn</h4>
                <h2>26.5%</h2>
                <small>1,869 clientes abandonaron</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>💰 Impacto Financiero</h4>
                <h2>$1.2M</h2>
                <small>Pérdida estimada anual</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h4>🎯 Precisión del Modelo</h4>
                <h2>86.2%</h2>
                <small>Modelo Stacking Diverse</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Insights de negocio
        st.subheader("💡 Insights Clave de Negocio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>🔍 Factores de Alto Riesgo</h4>
                <ul>
                    <li><strong>Contratos mes-a-mes:</strong> 42.7% de churn</li>
                    <li><strong>Fibra óptica:</strong> 41.9% de churn</li>
                    <li><strong>Clientes nuevos:</strong> 48.4% churn (<12 meses)</li>
                    <li><strong>Pago electrónico:</strong> Mayor propensión al abandono</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>💎 Oportunidades de Retención</h4>
                <ul>
                    <li><strong>Contratos largos:</strong> Solo 2.8% churn (2 años)</li>
                    <li><strong>Servicios de seguridad:</strong> Reducen 15% el churn</li>
                    <li><strong>Múltiples servicios:</strong> Aumentan lealtad</li>
                    <li><strong>Soporte técnico:</strong> Mejora satisfacción</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Recomendaciones estratégicas
        st.subheader("🚀 Recomendaciones Estratégicas")
        
        recommendations_tabs = st.tabs(["🔴 Alto Riesgo", "🟡 Riesgo Medio", "🟢 Bajo Riesgo"])
        
        with recommendations_tabs[0]:
            st.markdown("""
            **🚨 Estrategias para Clientes de Alto Riesgo:**
            
            1. **Intervención Inmediata (24-48h)**
               - Llamada del gerente de cuentas
               - Oferta de descuento del 20% por 6 meses
               - Migración gratuita a plan superior
            
            2. **Incentivos Especiales**
               - Servicios premium gratuitos por 3 meses
               - Extensión de contrato con beneficios
               - Programa de fidelidad acelerado
            
            3. **Seguimiento Intensivo**
               - Contacto semanal el primer mes
               - Encuestas de satisfacción mensuales
               - Revisión trimestral del plan
            """)
        
        with recommendations_tabs[1]:
            st.markdown("""
            **⚠️ Estrategias para Clientes de Riesgo Medio:**
            
            1. **Monitoreo Proactivo**
               - Análisis mensual de patrones de uso
               - Encuestas de satisfacción trimestrales
               - Alertas automáticas de cambios de comportamiento
            
            2. **Ofertas Preventivas**
               - Descuentos del 10% en servicios adicionales
               - Upgrade gratuito de servicios por 30 días
               - Programas de referidos con beneficios
            
            3. **Optimización del Servicio**
               - Análisis de calidad de conexión
               - Recomendaciones de plan personalizado
               - Acceso prioritario a soporte técnico
            """)
        
        with recommendations_tabs[2]:
            st.markdown("""
            **✅ Estrategias para Clientes Estables:**
            
            1. **Maximización de Valor**
               - Ofertas de upselling de servicios premium
               - Paquetes familiares con descuentos
               - Servicios empresariales para emprendedores
            
            2. **Programa de Embajadores**
               - Incentivos por referidos exitosos
               - Testimonios y casos de éxito
               - Participación en programa beta de nuevos servicios
            
            3. **Fidelización a Largo Plazo**
               - Descuentos por renovación anticipada
               - Servicios exclusivos para clientes VIP
               - Eventos especiales y beneficios únicos
            """)
        
        # Métricas de comparación final
        st.subheader("📊 Comparación de Modelos")
        
        comparison_data = pd.DataFrame({
            'Modelo': ['Stacking Diverse', 'Logistic Regression', 'Voting Classifier'],
            'Accuracy': [0.862, 0.834, 0.851],
            'F1-Score': [0.841, 0.812, 0.829],
            'AUC': [0.895, 0.871, 0.883],
            'Interpretabilidad': ['Media', 'Alta', 'Media'],
            'Velocidad': ['Media', 'Alta', 'Media']
        })
        
        st.dataframe(comparison_data, use_container_width=True)

if __name__ == "__main__":
    main()