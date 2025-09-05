import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ------------------------
# 1. Cargar predicciones
# ------------------------
@st.cache_data
def load_predictions():
    df = pd.read_csv(r"C:\Users\emili\sp-ml-17-final-project-g3\notebooks\predicciones_test.csv")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df

df_preds = load_predictions()

# ------------------------
# 2. Interfaz
# ------------------------
st.title("Mapa Predicho de Riesgo de Accidentes en Barcelona 🚦")

# Selector de fecha
fecha = st.date_input(
    "Selecciona una fecha:",
    value=df_preds["Fecha"].min(),
    min_value=df_preds["Fecha"].min(),
    max_value=df_preds["Fecha"].max()
)

# Slider de horas
hora = st.slider("Selecciona una hora:", 0, 23, 12)

# ------------------------
# 3. Filtrado
# ------------------------
df_dia = df_preds[df_preds["Fecha"].dt.date == fecha]
df_filtrado = df_dia[df_dia["hora"] == hora]

# ------------------------
# 4. Crear mapa
# ------------------------
m = folium.Map(location=[41.3851, 2.1734], zoom_start=12)

if not df_filtrado.empty:
    heat_data = df_filtrado[["lat", "lon", "y_proba"]].values.tolist()
    HeatMap(
        heat_data,
        radius=15,
        blur=20,
        min_opacity=0.4,
        gradient={0.2: "yellow", 0.5: "orange", 0.8: "red"}
    ).add_to(m)
else:
    st.warning("⚠️ No hay datos para esta fecha y hora.")

# ------------------------
# 5. Mostrar en Streamlit
# ------------------------
st_data = st_folium(m, width=700, height=500)
