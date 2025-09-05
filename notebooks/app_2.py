import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuración básica
st.set_page_config(
    page_title="Barcelona Accident Predictor",
    page_icon="🚗",
    layout="wide"
)

# Funciones
@st.cache_data
def load_predictions():
    try:
        df = pd.read_csv(r"C:\Users\emili\sp-ml-17-final-project-g3\notebooks\predicciones_test.csv")
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return pd.DataFrame()

def get_risk_color(risk_level):
    if risk_level >= 0.7:
        return "#dc3545", "🔴 Alto"
    elif risk_level >= 0.4:
        return "#ffc107", "🟡 Medio"
    else:
        return "#28a745", "🟢 Bajo"

# Cargar datos
df_preds = load_predictions()

if df_preds.empty:
    st.error("❌ No se pudieron cargar los datos. Verifica la ruta del archivo.")
    st.stop()

# HEADER
st.title("🚗 Predictor de Accidentes Barcelona")
st.markdown("**Visualización interactiva del riesgo de accidentes viales por horas**")
st.markdown("---")

# CONTROLES EN LA SIDEBAR
with st.sidebar:
    st.header("🎛️ Controles")
    
    # Fecha
    fecha_min = df_preds["Fecha"].dt.date.min()
    fecha_max = df_preds["Fecha"].dt.date.max()
    
    fecha = st.date_input(
        "📅 Fecha:",
        value=fecha_min,
        min_value=fecha_min,
        max_value=fecha_max
    )
    
    # Hora
    hora = st.slider("🕐 Hora:", 0, 23, 12)
    st.write(f"**Hora seleccionada:** {hora:02d}:00")
    
    # Filtros
    st.subheader("🔍 Filtros")
    
    risk_filter = st.selectbox(
        "Nivel de riesgo:",
        ["Todos", "Solo alto riesgo (>70%)", "Solo medio-alto (>40%)", "Solo bajo (<40%)"]
    )
    
    map_style = st.radio(
        "Estilo de mapa:",
        ["Mapa de calor", "Puntos", "Ambos"]
    )

    choose_model = st.selectbox(
        "Modelo predictivo:",
        ["Emiliano (LGBM)", "Pablo (LGBM)", "Simon (RANDOMFOREST)"]
    )

# FILTRAR DATOS
df_dia = df_preds[df_preds["Fecha"].dt.date == fecha]
df_filtrado = df_dia[df_dia["hora"] == hora]

# Aplicar filtro de riesgo
if risk_filter == "Solo alto riesgo (>70%)":
    df_filtrado = df_filtrado[df_filtrado['y_proba'] >= 0.7]
elif risk_filter == "Solo medio-alto (>40%)":
    df_filtrado = df_filtrado[df_filtrado['y_proba'] >= 0.4]
elif risk_filter == "Solo bajo (<40%)":
    df_filtrado = df_filtrado[df_filtrado['y_proba'] < 0.4]

# MÉTRICAS
if not df_filtrado.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    total_points = len(df_filtrado)
    avg_risk = df_filtrado['y_proba'].mean()
    high_risk_points = len(df_filtrado[df_filtrado['y_proba'] >= 0.7])
    max_risk = df_filtrado['y_proba'].max()
    
    with col1:
        st.metric("📍 Puntos", f"{total_points:,}")
    
    with col2:
        st.metric("📊 Riesgo Promedio", f"{avg_risk:.1%}")
    
    with col3:
        st.metric("🚨 Alto Riesgo", f"{high_risk_points:,}")
    
    with col4:
        st.metric("⚠️ Riesgo Máximo", f"{max_risk:.1%}")

st.markdown("---")

# LAYOUT PRINCIPAL
col_mapa, col_info = st.columns([2, 1])

with col_mapa:
    st.subheader("🗺️ Mapa de Riesgo")
    
    if not df_filtrado.empty:
        # Crear mapa
        m = folium.Map(
            location=[41.3851, 2.1734],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        if map_style in ["Mapa de calor", "Ambos"]:
            # Mapa de calor
            heat_data = df_filtrado[["lat", "lon", "y_proba"]].values.tolist()
            plugins.HeatMap(
                heat_data,
                radius=20,
                blur=15,
                min_opacity=0.3,
                gradient={0.0: '#28a745', 0.4: '#ffc107', 0.7: '#fd7e14', 1.0: '#dc3545'}
            ).add_to(m)
        
        if map_style in ["Puntos", "Ambos"]:
            # Puntos coloreados
            for _, row in df_filtrado.iterrows():
                color, level_text = get_risk_color(row['y_proba'])
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=max(3, row['y_proba'] * 15),
                    popup=f"""
                    <b>Riesgo:</b> {row['y_proba']:.1%}<br>
                    <b>Nivel:</b> {level_text}<br>
                    <b>Hora:</b> {hora:02d}:00<br>
                    <b>Coordenadas:</b> {row['lat']:.4f}, {row['lon']:.4f}
                    """,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
        
        # Mostrar mapa
        st_folium(m, width=700, height=500)
        
    else:
        st.warning(f"⚠️ No hay datos para {fecha} a las {hora:02d}:00")
        # Mapa vacío
        m = folium.Map(location=[41.3851, 2.1734], zoom_start=12)
        st_folium(m, width=700, height=500)

with col_info:
    st.subheader("📊 Información")
    
    if not df_filtrado.empty:
        # Información básica
        st.info(f"""
        **Análisis para {fecha} - {hora:02d}:00**
        
        📍 **Total de puntos:** {len(df_filtrado):,}
        
        📊 **Riesgo promedio:** {df_filtrado['y_proba'].mean():.1%}
        
        🚨 **Puntos alto riesgo:** {len(df_filtrado[df_filtrado['y_proba'] >= 0.7]):,}
        
        ⚠️ **Riesgo máximo:** {df_filtrado['y_proba'].max():.1%}
        """)
        
        # Top 5 zonas peligrosas
        st.subheader("🎯 Top 5 Zonas Peligrosas")
        top_risk = df_filtrado.nlargest(5, 'y_proba')
        for i, (_, row) in enumerate(top_risk.iterrows(), 1):
            color, level = get_risk_color(row['y_proba'])
            st.write(f"**{i}.** Riesgo: **{row['y_proba']:.1%}** {level}")
            st.write(f"   📍 {row['lat']:.4f}, {row['lon']:.4f}")
        
        # Histograma simple
        st.subheader("📈 Distribución de Riesgo")
        fig = px.histogram(df_filtrado, x='y_proba', nbins=15, title="Distribución del Riesgo")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

# GRÁFICOS ADICIONALES
if not df_dia.empty:
    st.markdown("---")
    st.subheader("📈 Análisis Temporal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Riesgo por hora
        hourly_risk = df_dia.groupby('hora')['y_proba'].mean().reset_index()
        fig = px.line(hourly_risk, x='hora', y='y_proba', 
                     title=f"Riesgo Promedio por Hora - {fecha}")
        fig.add_vline(x=hora, line_dash="dot", annotation_text=f"Actual: {hora:02d}:00")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Puntos de alto riesgo por hora
        high_risk_hourly = df_dia[df_dia['y_proba'] >= 0.7].groupby('hora').size().reset_index()
        high_risk_hourly.columns = ['hora', 'count']
        fig = px.bar(high_risk_hourly, x='hora', y='count', 
                    title=f"Puntos Alto Riesgo por Hora - {fecha}")
        st.plotly_chart(fig, use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown("🚗 **Barcelona Accident Risk Predictor** | Desarrollado con Streamlit")