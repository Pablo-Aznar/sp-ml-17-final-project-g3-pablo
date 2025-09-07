import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')
import json
import hashlib
from urllib.parse import quote
import pytz  # NUEVO: Para zona horaria de Barcelona

# === NUEVOS IMPORTS para ruteo vial y ETA ===
import os
import re
import networkx as nx
from math import radians, sin, cos, asin, sqrt
from shapely.geometry import LineString, Point
from shapely.ops import linemerge
import geopandas as gpd

# === Open-Meteo ===
import requests

# === Geocoding ===
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    st.warning("⚠️ Instala geopy para búsqueda de direcciones: pip install geopy")

try:
    import osmnx as ox
    OSMNX_AVAILABLE = True
except Exception:
    OSMNX_AVAILABLE = False

# === KD-Tree opcional para acelerar búsqueda espacial (mejora ruteo) ===
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# === Shapely para geometría avanzada ===
try:
    from shapely.geometry import LineString, Point
    from shapely.ops import linemerge
    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False
    st.warning("⚠️ Instala shapely para geometría avanzada: pip install shapely")

# === ZONA HORARIA DE BARCELONA ===
BARCELONA_TZ = pytz.timezone('Europe/Madrid')

# === DISTRITOS DE BARCELONA ===
BARCELONA_DISTRICTS = [
    'Ciutat Vella', 'Eixample', 'Sants-Montjuïc', 'Les Corts', 'Sarrià-Sant Gervasi',
    'Gràcia', 'Horta-Guinardó', 'Nou Barris', 'Sant Andreu', 'Sant Martí'
]

# === TIPOS DE CARRETERA PARA COCHES (Optimizado para Barcelona) ===
ALLOWED_HIGHWAY_TYPES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 
    'unclassified', 'residential', 'motorway_link', 'trunk_link', 
    'primary_link', 'secondary_link', 'tertiary_link'
}

# === FILTROS ESPECÍFICOS BARCELONA ===
EXCLUDED_HIGHWAY_TYPES = {
    'cycleway', 'footway', 'pedestrian', 'path', 'steps', 
    'track', 'service', 'living_street', 'construction'
}

# === FESTIVOS DE BARCELONA 2025 ===
FESTIVOS_BARCELONA_2025 = {
    (2025, 1, 1): "Año Nuevo",
    (2025, 1, 6): "Reyes",
    (2025, 4, 18): "Viernes Santo",
    (2025, 4, 21): "Lunes de Pascua",
    (2025, 5, 1): "Día del Trabajo",
    (2025, 6, 24): "San Juan",
    (2025, 8, 15): "Asunción",
    (2025, 9, 11): "Diada Nacional de Catalunya",
    (2025, 9, 24): "La Mercè",
    (2025, 10, 12): "Fiesta Nacional España",
    (2025, 11, 1): "Todos los Santos",
    (2025, 12, 6): "Día de la Constitución",
    (2025, 12, 8): "Inmaculada Concepción",
    (2025, 12, 25): "Navidad",
    (2025, 12, 26): "San Esteban",
}

# === LUGARES POPULARES DE BARCELONA ===
POPULAR_PLACES = {
    "🏛️ Sagrada Familia": (41.4036, 2.1744),
    "🌳 Park Güell": (41.4145, 2.1527),
    "⚽ Camp Nou": (41.3809, 2.1228),
    "🏥 Hospital Clínic": (41.3889, 2.1505),
    "📍 Plaça Catalunya": (41.3870, 2.1701),
    "✈️ Aeropuerto T1": (41.2974, 2.0833),
    "🚂 Estación Sants": (41.3792, 2.1402),
    "⛵ Port Olímpic": (41.3874, 2.1963),
    "🏥 Hospital Sant Pau": (41.4144, 2.1774),
    "🎓 Universidad BCN": (41.3866, 2.1639),
    "🏖️ Barceloneta": (41.3805, 2.1893),
    "🛍️ La Rambla": (41.3810, 2.1730),
    "🏛️ Casa Batlló": (41.3916, 2.1650),
    "🎨 MACBA": (41.3833, 2.1667),
    "🏟️ Palau Sant Jordi": (41.3643, 2.1527),
}

# === RUTAS FAVORITAS (Persistencia) ===
FAVORITES_FILE = "favorite_routes.json"

def load_favorite_routes():
    """Cargar rutas favoritas desde archivo JSON"""
    if os.path.exists(FAVORITES_FILE):
        try:
            with open(FAVORITES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_favorite_routes(favorites):
    """Guardar rutas favoritas en archivo JSON"""
    try:
        with open(FAVORITES_FILE, 'w', encoding='utf-8') as f:
            json.dump(favorites, f, ensure_ascii=False, indent=2)
        return True
    except:
        return False

def is_festivo(date_obj):
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    if year == 2025:
        return (year, month, day) in FESTIVOS_BARCELONA_2025
    return False

def get_festivo_name(date_obj):
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    if year == 2025:
        return FESTIVOS_BARCELONA_2025.get((year, month, day), None)
    return None

# Configuración de página
st.set_page_config(
    page_title="Predicción Accidentes Barcelona",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS profesionales
st.markdown("""
<style>
    /* Variables de color profesionales */
    :root {
        --primary-blue: #2E5BBA;
        --secondary-blue: #8BB8E8;
        --accent-green: #00A86B;
        --warning-orange: #FF8C42;
        --danger-red: #DC3545;
        --neutral-gray: #6C757D;
        --light-gray: #F8F9FA;
        --dark-gray: #212529;
        --white: #FFFFFF;
    }
    
    /* Estilo de la aplicación principal */
    .stApp {
        background-color: var(--light-gray);
    }
    
    /* Tarjetas de ruta profesionales */
    .route-card {
        background: linear-gradient(145deg, var(--white) 0%, #F1F3F5 100%);
        border: 1px solid #DEE2E6;
        border-radius: 12px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        transition: all 0.3s ease;
        position: relative;
    }
    
    .route-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    .route-card h3 {
        color: var(--primary-blue);
        font-weight: 600;
        margin-bottom: 20px;
        font-size: 1.3em;
        border-bottom: 2px solid var(--secondary-blue);
        padding-bottom: 8px;
    }
    
    /* Badges de seguridad profesionales */
    .safety-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        margin: 4px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .high-risk { 
        background: linear-gradient(135deg, var(--danger-red), #FF6B6B);
        color: white;
    }
    
    .medium-risk { 
        background: linear-gradient(135deg, var(--warning-orange), #FFB347);
        color: white;
    }
    
    .low-risk { 
        background: linear-gradient(135deg, var(--accent-green), #4CAF50);
        color: white;
    }
    
    /* Información de tráfico elegante */
    .traffic-info {
        background: var(--white);
        border: 1px solid #E9ECEF;
        border-left: 4px solid var(--primary-blue);
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .traffic-info b {
        color: var(--primary-blue);
        font-weight: 600;
    }
    
    /* Métricas profesionales */
    .metric-container {
        background: var(--white);
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .metric-container:hover {
        border-color: var(--secondary-blue);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Botones profesionales */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Sidebar profesional */
    .css-1d391kg {
        background-color: var(--white);
        border-right: 1px solid #E9ECEF;
    }
    
    /* Headers mejorados */
    h1, h2, h3 {
        color: var(--dark-gray);
        font-weight: 600;
    }
    
    h1 {
        border-bottom: 3px solid var(--primary-blue);
        padding-bottom: 10px;
    }
    
    h2 {
        border-bottom: 2px solid var(--secondary-blue);
        padding-bottom: 8px;
    }
    
    /* Inputs más elegantes */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #CED4DA;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-blue);
        box-shadow: 0 0 0 0.2rem rgba(46, 91, 186, 0.25);
    }
    
    .stSelectbox > div > div > div {
        border-radius: 8px;
    }
    
    /* Alertas profesionales */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Métricas del dashboard */
    div[data-testid="metric-container"] {
        background: var(--white);
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Tabs más elegantes */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--light-gray);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        border: 1px solid #E9ECEF;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--white);
        border-color: var(--primary-blue);
        color: var(--primary-blue);
    }
    
    /* Mapas con bordes elegantes */
    iframe {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Código más elegante */
    .stCode {
        border-radius: 8px;
        border: 1px solid #E9ECEF;
    }
    
    /* Separadores elegantes */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--secondary-blue), transparent);
        margin: 24px 0;
    }
    
    /* Tooltips y popups */
    .stTooltipIcon {
        color: var(--neutral-gray);
    }
    
    /* Animaciones suaves */
    * {
        transition: all 0.2s ease;
    }
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-gray);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--neutral-gray);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-blue);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
#   HORA LOCAL DE BARCELONA
# -----------------------------
def get_barcelona_time():
    """Obtener hora actual de Barcelona"""
    return datetime.now(BARCELONA_TZ)

def get_next_hour_barcelona():
    """
    Obtener la siguiente hora completa en Barcelona (minutos = 0)
    Maneja el cambio de día automáticamente
    """
    now = get_barcelona_time()
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    
    # Convertir a fecha y hora locales para el widget
    next_date = next_hour.date()
    next_hour_value = next_hour.hour
    
    return next_date, next_hour_value

# -----------------------------
#   Geocoding y búsqueda de direcciones - BARCELONA ESPECÍFICO
# -----------------------------
@st.cache_data(ttl=3600)
def geocode_address(address):
    """Geocodificar una dirección usando Nominatim - Solo Barcelona"""
    if not GEOPY_AVAILABLE:
        return None
    
    try:
        geolocator = Nominatim(user_agent="barcelona_accidents_app")
        # Forzar búsqueda específica en Barcelona
        search_address = f"{address}, Barcelona, España"
        
        location = geolocator.geocode(search_address, timeout=10)
        if location:
            return {
                'address': location.address,
                'lat': location.latitude,
                'lon': location.longitude
            }
    except GeocoderTimedOut:
        st.error("Timeout al buscar la dirección. Intenta de nuevo.")
    except Exception as e:
        st.error(f"Error geocoding: {e}")
    
    return None

def is_in_barcelona_city(address):
    """Verificar si una dirección está específicamente en Barcelona ciudad"""
    address_lower = address.lower()
    
    # Debe contener Barcelona
    if 'barcelona' not in address_lower:
        return False
    
    # No debe ser área metropolitana
    excluded_areas = ['hospitalet', 'badalona', 'santa coloma', 'cornellà', 'esplugues', 
                      'sant boi', 'viladecans', 'castelldefels', 'gavà', 'el prat']
    
    for excluded in excluded_areas:
        if excluded in address_lower:
            return False
    
    # Preferir direcciones con código postal de Barcelona (080xx)
    if any(code in address for code in ['080', '08001', '08002', '08003', '08004', '08005']):
        return True
    
    # Verificar distritos de Barcelona
    for district in BARCELONA_DISTRICTS:
        if district.lower() in address_lower:
            return True
    
    # Si contiene Barcelona y no está excluida, probablemente sea válida
    return True

@st.cache_data(ttl=3600)
def search_addresses(query, limit=5):
    """Buscar direcciones que coincidan con la consulta - SOLO BARCELONA CIUDAD"""
    if not GEOPY_AVAILABLE:
        return []
    
    try:
        geolocator = Nominatim(user_agent="barcelona_accidents_app")
        # Búsqueda más específica en Barcelona ciudad
        search_query = f"{query}, Barcelona, España"
        
        locations = geolocator.geocode(
            search_query,
            exactly_one=False,
            limit=limit*2,  # Buscar más para poder filtrar
            timeout=10
        )
        
        if locations:
            results = []
            for loc in locations:
                # Filtro estricto para Barcelona ciudad
                if is_in_barcelona_city(loc.address):
                    # Verificar coordenadas están dentro de Barcelona aproximadamente
                    lat, lon = loc.latitude, loc.longitude
                    if (41.32 <= lat <= 41.47) and (2.05 <= lon <= 2.25):
                        results.append({
                            'address': loc.address,
                            'lat': lat,
                            'lon': lon
                        })
                        
                        if len(results) >= limit:
                            break
            
            return results
    except:
        pass
    
    return []

@st.cache_data(ttl=300)  # Cache de 5 minutos para evitar muchas consultas
def get_address_suggestions(query_text):
    """
    Obtener sugerencias de direcciones para autocompletado - SOLO BARCELONA CIUDAD
    """
    if len(query_text.strip()) < 3:
        return []
    
    suggestions = search_addresses(query_text, limit=5)
    # Formatear las sugerencias para el dropdown
    options = []
    for suggestion in suggestions:
        # Limpiar y acortar el texto mostrado
        display_text = suggestion['address']
        # Remover "España" del final si existe
        if display_text.endswith(", España"):
            display_text = display_text[:-9]
        
        if len(display_text) > 80:
            display_text = display_text[:77] + "..."
        options.append({
            'display': display_text,
            'coords': (suggestion['lat'], suggestion['lon']),
            'full_address': suggestion['address']
        })
    
    return options

def create_address_autocomplete(label, key_prefix, placeholder_text):
    """
    Crear un campo de autocompletado para direcciones
    Devuelve las coordenadas seleccionadas o None
    """
    # Campo de texto para escribir
    query = st.text_input(
        f"Escribe para buscar {label.lower()}",
        placeholder=placeholder_text,
        key=f"{key_prefix}_query"
    )
    
    selected_coords = None
    
    # Solo mostrar sugerencias si hay 3+ caracteres
    if len(query.strip()) >= 3:
        with st.spinner("Buscando direcciones en Barcelona..."):
            suggestions = get_address_suggestions(query)
        
        if suggestions:
            # Crear opciones para el selectbox
            options = ["Selecciona una dirección..."] + [s['display'] for s in suggestions]
            
            selected = st.selectbox(
                f"Sugerencias para {label.lower()}:",
                options=options,
                key=f"{key_prefix}_select"
            )
            
            # Si se selecciona una opción válida
            if selected != "Selecciona una dirección..." and selected in [s['display'] for s in suggestions]:
                # Encontrar las coordenadas correspondientes
                for suggestion in suggestions:
                    if suggestion['display'] == selected:
                        selected_coords = suggestion['coords']
                        st.success(f"✅ {label} seleccionado: {suggestion['full_address'][:50]}...")
                        break
        else:
            st.info("No se encontraron direcciones en Barcelona. Prueba con otros términos.")
    elif len(query.strip()) > 0:
        st.info("Escribe al menos 3 caracteres para buscar direcciones")
    
    return selected_coords

# -----------------------------
#   Simulación de tráfico en tiempo real
# -----------------------------
@st.cache_data(ttl=300)  # Cache de 5 minutos
def get_traffic_conditions(lat1, lon1, lat2, lon2, current_hour=None):
    """
    Simular condiciones de tráfico basadas en hora del día
    En producción, usar API de TomTom o Google Maps
    """
    if current_hour is None:
        current_hour = get_barcelona_time().hour
    
    # Factor de tráfico según hora del día
    traffic_factors = {
        # Madrugada: tráfico muy ligero
        0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.6,
        # Mañana: hora punta
        6: 0.8, 7: 1.3, 8: 1.5, 9: 1.4, 10: 1.1, 11: 0.9,
        # Mediodía
        12: 1.0, 13: 1.2, 14: 1.3, 15: 1.0,
        # Tarde: hora punta
        16: 1.1, 17: 1.3, 18: 1.5, 19: 1.4, 20: 1.2,
        # Noche
        21: 0.9, 22: 0.7, 23: 0.6
    }
    
    base_factor = traffic_factors.get(current_hour, 1.0)
    
    # Añadir algo de aleatoriedad para simular variaciones
    random_factor = np.random.uniform(0.9, 1.1)
    traffic_multiplier = base_factor * random_factor
    
    # Determinar nivel de tráfico
    if traffic_multiplier < 0.7:
        traffic_level = "🟢 Tráfico ligero"
        color = "green"
    elif traffic_multiplier < 1.2:
        traffic_level = "🟡 Tráfico moderado"
        color = "yellow"
    else:
        traffic_level = "🔴 Tráfico denso"
        color = "red"
    
    return {
        'multiplier': traffic_multiplier,
        'level': traffic_level,
        'color': color,
        'description': f"Factor de tráfico: {traffic_multiplier:.1f}x"
    }

def generate_google_maps_url(origin, destination, waypoints=None):
    """
    Generar URL de Google Maps para navegación
    """
    base_url = "https://www.google.com/maps/dir/"
    
    # Formatear origen y destino
    origin_str = f"{origin[0]},{origin[1]}"
    dest_str = f"{destination[0]},{destination[1]}"
    
    # Construir URL
    url = f"{base_url}{origin_str}/{dest_str}"
    
    # Añadir parámetros
    params = "?travelmode=driving"
    
    # Si hay waypoints, añadirlos
    if waypoints and len(waypoints) > 0:
        waypoints_str = "|".join([f"{wp[0]},{wp[1]}" for wp in waypoints[:8]])  # Max 8 waypoints
        params += f"&waypoints={quote(waypoints_str)}"
    
    return url + params

def generate_waze_url(origin, destination):
    """
    Generar URL de Waze para navegación
    """
    return f"https://www.waze.com/ul?ll={destination[0]},{destination[1]}&navigate=yes&from={origin[0]},{origin[1]}"

# -----------------------------
#   Carga de modelo
# -----------------------------
@st.cache_data
def load_model():
    try:
        model_data = joblib.load("./models/barcelona_accident_model_enhanced.joblib")
        return model_data
    except FileNotFoundError:
        st.error("Modelo mejorado no encontrado. Ejecuta el script 2 mejorado primero.")
        return None

# -----------------------------
#   Predicción por clúster
# -----------------------------
def create_prediction_function(_model_data):
    feature_cols = _model_data['feature_cols']

    def predict_risk(cluster_id, hour, temperature, precipitation, wind_speed, date_obj):
        # Calendario
        day_of_week = date_obj.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        dia_festivo = 1 if is_festivo(date_obj) else 0
        fiesta = dia_festivo

        fd = {}
        # Meteo directas
        fd['temperature_2m (°C)'] = float(temperature)
        fd['wind_speed_10m (km/h)'] = float(wind_speed)

        # Calendario/tiempo
        fd['Fiesta'] = int(fiesta)
        fd['dia_festivo'] = int(dia_festivo)
        fd['hour'] = int(hour)
        fd['day_of_week'] = int(day_of_week)
        fd['is_weekend'] = int(is_weekend)

        # Cíclicas
        fd['hour_sin'] = float(np.sin(2 * np.pi * hour / 24))
        fd['hour_cos'] = float(np.cos(2 * np.pi * hour / 24))
        fd['dow_sin']  = float(np.sin(2 * np.pi * day_of_week / 7))
        fd['dow_cos']  = float(np.cos(2 * np.pi * day_of_week / 7))

        # Binarias derivadas
        fd['is_rush_hour'] = int(hour in [7, 8, 9, 17, 18, 19])
        fd['is_night'] = int((hour >= 22) or (hour <= 5))

        # Meteo derivadas
        fd['temp_cold'] = int(temperature < 10)
        fd['temp_hot']  = int(temperature > 25)
        has_rain = int(precipitation > 0.1)

        # Interacciones
        fd['weekend_night'] = int(is_weekend * fd['is_night'])
        fd['rush_rain']     = int(fd['is_rush_hour'] * has_rain)

        # Cluster histórico
        historical_data = _model_data.get('historical_data', {})
        accidents_by_cluster = historical_data.get('accidents_by_cluster', {})
        if cluster_id in accidents_by_cluster and len(accidents_by_cluster) > 0:
            cluster_accidents = accidents_by_cluster[cluster_id]
            max_accidents = max(accidents_by_cluster.values())
            max_accidents = max(max_accidents, 1)
            fd['cluster_acc_rate'] = float(min(0.15, cluster_accidents / max_accidents * 0.15))
            fd['cluster_count']    = float(cluster_accidents)
        else:
            fd['cluster_acc_rate'] = 0.02
            fd['cluster_count']    = 100.0

        fd['cluster_rain_avg'] = float(precipitation)

        feature_array = np.array([fd.get(col, 0.0) for col in feature_cols], dtype=float).reshape(1, -1)
        X_sel = _model_data['selector'].transform(feature_array) if _model_data.get('selector') is not None else feature_array
        X_scl = _model_data['scaler'].transform(X_sel) if _model_data.get('scaler') is not None else X_sel

        prob = float(_model_data['model'].predict_proba(X_scl)[0, 1])
        # suavizado contextual opcional (ligero)
        if fd['is_rush_hour'] and is_weekend:
            prob *= 1.2
        elif fd['is_rush_hour']:
            prob *= 1.1
        elif fd['is_night'] and is_weekend:
            prob *= 1.15
        prob = min(0.99, prob)

        return {
            'probability': prob,
            'prediction': int(prob >= _model_data['optimal_threshold']),
            'risk_level': 'Alto' if prob >= 0.10 else ('Medio' if prob >= 0.07 else 'Bajo')
        }

    return predict_risk

def get_risk_color(probability):
    if probability >= 0.10:
        return '#e74c3c'
    elif probability >= 0.07:
        return '#f39c12'
    else:
        return '#27ae60'

def get_gradient_color(risk_value):
    """Obtener color gradiente según el riesgo (verde->amarillo->rojo)"""
    if risk_value <= 0.05:
        # Verde a amarillo
        ratio = risk_value / 0.05
        r = int(39 + (243 - 39) * ratio)
        g = int(174 + (243 - 174) * ratio)
        b = int(96 + (18 - 96) * ratio)
    else:
        # Amarillo a rojo
        ratio = min(1.0, (risk_value - 0.05) / 0.05)
        r = int(243 + (231 - 243) * ratio)
        g = int(156 + (76 - 156) * ratio)
        b = int(18 + (60 - 18) * ratio)
    return f'#{r:02x}{g:02x}{b:02x}'

# -----------------------------
#   Geodésicos
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2.0)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2.0)**2
    return 2 * R * asin(sqrt(a))

# -----------------------------
#   Red vial OSMnx OPTIMIZADA PARA BARCELONA
# -----------------------------
GRAPH_DIR = "./data/graph"
GRAPH_PATH = os.path.join(GRAPH_DIR, "barcelona_drive_detailed.graphml")
GRAPH_GEOMETRY_PATH = os.path.join(GRAPH_DIR, "barcelona_geometry.json")

def filter_graph_for_cars(G):
    """
    Filtrar el grafo para quedarse solo con carreteras transitables por coches
    """
    if G is None:
        return None
    
    # Crear copia del grafo
    G_filtered = G.copy()
    
    # Lista de aristas a eliminar
    edges_to_remove = []
    
    for u, v, k, data in G_filtered.edges(keys=True, data=True):
        highway = data.get('highway', '')
        
        # Convertir a lista si es string
        if isinstance(highway, str):
            highway_types = [highway]
        else:
            highway_types = highway if isinstance(highway, list) else [str(highway)]
        
        # Verificar si algún tipo de carretera está excluido
        should_remove = False
        for hw_type in highway_types:
            if hw_type in EXCLUDED_HIGHWAY_TYPES:
                should_remove = True
                break
            # También verificar que esté en tipos permitidos
            if hw_type not in ALLOWED_HIGHWAY_TYPES:
                should_remove = True
                break
        
        # Verificar restricciones de acceso
        access = data.get('access', '')
        motor_vehicle = data.get('motor_vehicle', '')
        
        if access in ['no', 'private', 'customers'] or motor_vehicle == 'no':
            should_remove = True
        
        if should_remove:
            edges_to_remove.append((u, v, k))
    
    # Eliminar aristas filtradas
    G_filtered.remove_edges_from(edges_to_remove)
    
    # Eliminar nodos aislados
    isolated_nodes = list(nx.isolates(G_filtered))
    G_filtered.remove_nodes_from(isolated_nodes)
    
    return G_filtered

def extract_edge_geometry(G):
    """
    Extraer y guardar la geometría real de cada arista
    """
    edge_geometries = {}
    
    for u, v, k, data in G.edges(keys=True, data=True):
        edge_id = f"{u}_{v}_{k}"
        
        # Obtener coordenadas de nodos
        u_coords = (G.nodes[u]['y'], G.nodes[u]['x'])
        v_coords = (G.nodes[v]['y'], G.nodes[v]['x'])
        
        # Verificar si hay geometría en los datos
        geometry = data.get('geometry', None)
        
        if geometry is not None and SHAPELY_AVAILABLE:
            # Extraer coordenadas de la geometría
            if hasattr(geometry, 'coords'):
                coords = [(lat, lon) for lon, lat in geometry.coords]
            else:
                coords = [u_coords, v_coords]
        else:
            # Usar solo puntos de inicio y fin
            coords = [u_coords, v_coords]
        
        edge_geometries[edge_id] = {
            'coords': coords,
            'length': data.get('length', 0),
            'highway': data.get('highway', 'unknown')
        }
    
    return edge_geometries

def interpolate_edge_coordinates(coords, max_segment_length_m=50):
    """
    Interpolar puntos adicionales en segmentos largos para visualización suave
    """
    if not SHAPELY_AVAILABLE or len(coords) < 2:
        return coords
    
    try:
        # Crear LineString
        line = LineString([(lon, lat) for lat, lon in coords])
        
        # Calcular longitud total en metros (aproximado)
        total_length_deg = line.length
        total_length_m = total_length_deg * 111000  # Conversión aprox de grados a metros
        
        # Si el segmento es corto, no interpolar
        if total_length_m < max_segment_length_m:
            return coords
        
        # Calcular número de puntos a interpolar
        num_points = int(total_length_m / max_segment_length_m)
        num_points = min(num_points, 50)  # Límite máximo
        
        # Generar puntos interpolados
        interpolated_coords = []
        for i in range(num_points + 1):
            fraction = i / num_points
            point = line.interpolate(fraction, normalized=True)
            interpolated_coords.append((point.y, point.x))  # lat, lon
        
        return interpolated_coords
    
    except Exception:
        return coords

@st.cache_data(show_spinner=True)
def load_or_build_graph():
    """
    Cargar o construir grafo detallado optimizado para Barcelona
    """
    if not OSMNX_AVAILABLE:
        return None, None, "OSMnx no está instalado. Ejecuta: pip install osmnx"
    
    os.makedirs(GRAPH_DIR, exist_ok=True)
    
    # Intentar cargar grafo existente
    if os.path.exists(GRAPH_PATH) and os.path.exists(GRAPH_GEOMETRY_PATH):
        try:
            st.info("📂 Cargando grafo detallado desde caché...")
            G = ox.load_graphml(GRAPH_PATH)
            
            with open(GRAPH_GEOMETRY_PATH, 'r') as f:
                edge_geometries = json.load(f)
            
            st.success("✅ Grafo detallado cargado correctamente")
            return G, edge_geometries, None
            
        except Exception as e:
            st.warning(f"⚠️ Error cargando caché, descargando de nuevo: {e}")
    
    # Descargar y procesar grafo
    try:
        st.info("🌍 Descargando red vial detallada de Barcelona...")
        
        # Configurar OSMnx para mayor detalle
        ox.settings.use_cache = True
        ox.settings.log_console = False
        
        # Descargar grafo SIN simplificar para mayor precisión
        G = ox.graph_from_place(
            "Barcelona, Spain", 
            network_type="drive", 
            simplify=False,  # CLAVE: No simplificar para mantener geometría
            retain_all=False,
            truncate_by_edge=True,
            custom_filter=None  # Usar filtro por defecto para carreteras
        )
        
        # IMPORTANTE: Asegurar que el grafo respeta direcciones de circulación
        # OSMnx por defecto crea grafos dirigidos, pero lo hacemos explícito
        if not G.is_directed():
            st.warning("⚠️ Convirtiendo a grafo dirigido para respetar direcciones de tráfico")
            G = G.to_directed()
        
        st.info("🚗 Filtrando carreteras para coches...")
        G_filtered = filter_graph_for_cars(G)
        
        if G_filtered is None or len(G_filtered.edges()) == 0:
            return None, None, "Error: Grafo filtrado está vacío"
        
        st.info("📐 Extrayendo geometría de carreteras...")
        edge_geometries = extract_edge_geometry(G_filtered)
        
        # Guardar en caché
        st.info("💾 Guardando en caché...")
        ox.save_graphml(G_filtered, GRAPH_PATH)
        
        with open(GRAPH_GEOMETRY_PATH, 'w') as f:
            json.dump(edge_geometries, f, indent=2)
        
        st.success(f"✅ Grafo detallado creado: {len(G_filtered.nodes())} nodos, {len(G_filtered.edges())} aristas")
        return G_filtered, edge_geometries, None
        
    except Exception as e:
        return None, None, f"No se pudo descargar la red de Barcelona: {e}"

def parse_maxspeed(value):
    if value is None:
        return None
    if isinstance(value, list):
        speeds = [parse_maxspeed(v) for v in value]
        speeds = [s for s in speeds if s is not None]
        return max(speeds) if speeds else None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).lower()
    m = re.search(r"(\d+(\.\d+)?)", s)
    if m:
        return float(m.group(1))
    return None

DEFAULT_SPEEDS = {
    "motorway": 80, "trunk": 60, "primary": 50,
    "secondary": 50, "tertiary": 40, "unclassified": 30,
    "residential": 30, "service": 20, "living_street": 10,
    "motorway_link": 60, "trunk_link": 50, "primary_link": 40,
    "secondary_link": 40, "tertiary_link": 30
}

VEHICLE_SPEED_FACTORS = {
    "🚗 Coche": 0.3,  # Mucho más lento en ciudad por tráfico real
    "🏍️ Moto": 0.4,  # Algo más rápido que coches en tráfico
    "🚲 Bicicleta": 0.08,  # Mucho más lento
    "🚚 Camión": 0.25,  # Más lento que coches
}

VEHICLE_RISK_FACTORS = {
    "🚗 Coche": 1.0,
    "🏍️ Moto": 1.3,  # Mayor riesgo para motos
    "🚲 Bicicleta": 1.5,  # Mucho más vulnerable
    "🚚 Camión": 0.9,  # Ligeramente más seguro
}

def infer_speed_kmh(edge_data, vehicle_type="🚗 Coche"):
    mx = parse_maxspeed(edge_data.get("maxspeed"))
    if mx is not None and mx > 0:
        base_speed = float(mx)
    else:
        hw = edge_data.get("highway")
        if isinstance(hw, list):
            candidates = [DEFAULT_SPEEDS.get(h, 30) for h in hw]
            base_speed = float(max(candidates) if candidates else 30.0)
        else:
            base_speed = float(DEFAULT_SPEEDS.get(hw, 30.0))
    
    # Ajustar por tipo de vehículo
    return base_speed * VEHICLE_SPEED_FACTORS.get(vehicle_type, 1.0)

def nearest_osm_node(G, lat, lon):
    try:
        return ox.distance.nearest_nodes(G, lon, lat)
    except Exception:
        min_d, min_n = 1e18, None
        for n, d in G.nodes(data=True):
            d_km = haversine(lat, lon, d.get('y'), d.get('x'))
            if d_km < min_d:
                min_d, min_n = d_km, n
        return min_n

# ====== Riesgo → pesos de ruta (MEJORADO) ======
def risk_factor_from_prob(prob, vehicle_type="🚗 Coche"):
    """Penalización: más fuerte para notar el cambio con el slider."""
    base_factor = 1.0
    if prob >= 0.10:   # alto
        base_factor = 7.0
    elif prob >= 0.07: # medio
        base_factor = 3.5
    else:
        base_factor = 1.0
    
    # Ajustar por tipo de vehículo
    vehicle_factor = VEHICLE_RISK_FACTORS.get(vehicle_type, 1.0)
    return base_factor * vehicle_factor

def build_cluster_kdtree(cluster_geometries, predictions_data):
    """Prepara KD-Tree (si SciPy disponible) con lat/lon y probs de clusters."""
    pts, probs, ids = [], [], []
    for cid, geo in cluster_geometries.items():
        if cid in predictions_data:
            pts.append([geo['lat'], geo['lon']])
            probs.append(predictions_data[cid]['probability'])
            ids.append(cid)
    if len(pts) == 0:
        return None, None, None
    pts = np.array(pts)
    if SCIPY_AVAILABLE:
        tree = cKDTree(pts)
    else:
        tree = None
    return tree, pts, np.array(probs)

def precompute_edge_costs(G, predictions_data, cluster_geometries, buffer_m=200.0, 
                         objective="balanced", vehicle_type="🚗 Coche", traffic_multiplier=1.0):
    """
    Para cada arista con factor de tráfico añadido - OPTIMIZADO
    """
    if G is None or predictions_data is None or cluster_geometries is None:
        return 0

    tree, pts, probs = build_cluster_kdtree(cluster_geometries, predictions_data)

    penalized = 0
    radius_deg = buffer_m / 111000.0

    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = float(data.get('length', 0.0))
        length_km = max(1e-4, length_m / 1000.0)
        speed_kmh = max(5.0, infer_speed_kmh(data, vehicle_type))
        
        # Aplicar factor de tráfico
        effective_speed = speed_kmh / traffic_multiplier
        time_min = (length_km / effective_speed) * 60.0

        yu, xu = G.nodes[u].get('y'), G.nodes[u].get('x')
        yv, xv = G.nodes[v].get('y'), G.nodes[v].get('x')
        y_mid, x_mid = (yu + yv) / 2.0, (xu + xv) / 2.0

        # prob de riesgo (máx en el radio)
        risk_prob = 0.0
        if tree is not None:
            idxs = tree.query_ball_point([y_mid, x_mid], r=radius_deg)
            if idxs:
                risk_prob = float(np.max(probs[idxs]))
        else:
            for cid, geo in cluster_geometries.items():
                if cid not in predictions_data:
                    continue
                d_km = haversine(y_mid, x_mid, geo['lat'], geo['lon'])
                if d_km * 1000.0 <= buffer_m:
                    risk_prob = max(risk_prob, predictions_data[cid]['probability'])

        rf = risk_factor_from_prob(risk_prob, vehicle_type)
        if rf > 1.0:
            penalized += 1

        if objective == "fastest":
            route_weight = time_min
        elif objective == "safest":
            route_weight = time_min * rf
        else:  # balanced
            route_weight = time_min * (1.0 + 0.5 * (rf - 1.0))

        data['length_km'] = length_km
        data['speed_kmh'] = speed_kmh
        data['time_min'] = time_min
        data['risk_prob'] = risk_prob
        data['route_weight'] = route_weight

    return penalized

def route_between_nodes(G, src, dst, edge_geometries=None):
    """
    Calcular ruta entre nodos con geometría real de carreteras
    """
    try:
        path = nx.shortest_path(G, src, dst, weight='route_weight')
    except nx.NetworkXNoPath:
        return None
    
    # Coordenadas básicas (nodos)
    basic_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
    
    # Coordenadas detalladas (con geometría real)
    detailed_coords = []
    total_km = 0.0
    total_time_min = 0.0
    risks = []
    
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        best = None
        best_w = 1e18
        best_key = None
        
        # Encontrar la mejor arista entre los nodos
        for k in G[u][v].keys():
            w = G[u][v][k].get('route_weight', None)
            if w is not None and w < best_w:
                best_w = w
                best = G[u][v][k]
                best_key = k
        
        if best is not None:
            total_km += best.get('length_km', 0.0)
            total_time_min += best.get('time_min', 0.0)
            risks.append(best.get('risk_prob', 0.0))
            
            # Añadir geometría detallada si está disponible
            if edge_geometries and best_key is not None:
                edge_id = f"{u}_{v}_{best_key}"
                if edge_id in edge_geometries:
                    edge_coords = edge_geometries[edge_id]['coords']
                    # Interpolar para suavizar curvas
                    interpolated = interpolate_edge_coordinates(edge_coords)
                    
                    # Evitar duplicar el último punto del segmento anterior
                    if detailed_coords and len(interpolated) > 0:
                        interpolated = interpolated[1:]  # Omitir primer punto
                    
                    detailed_coords.extend(interpolated)
                else:
                    # Fallback a coordenadas básicas
                    if i == 0:  # Primer segmento
                        detailed_coords.extend([basic_coords[i], basic_coords[i+1]])
                    else:
                        detailed_coords.append(basic_coords[i+1])
            else:
                # Fallback a coordenadas básicas
                if i == 0:  # Primer segmento
                    detailed_coords.extend([basic_coords[i], basic_coords[i+1]])
                else:
                    detailed_coords.append(basic_coords[i+1])
    
    # Si no se pudo generar coordenadas detalladas, usar básicas
    if not detailed_coords:
        detailed_coords = basic_coords
    
    return {
        "path": path, 
        "coords": detailed_coords,  # Coordenadas con geometría real
        "basic_coords": basic_coords,  # Coordenadas básicas de respaldo
        "km": total_km, 
        "min": total_time_min,
        "risks": risks,
        "avg_risk": np.mean(risks) if risks else 0.0
    }

def calculate_alternative_routes(G_full, edge_geometries, predictions_data, cluster_geometries,
                                origin, destination, vehicle_type="🚗 Coche",
                                traffic_multiplier=1.0):
    """Calcular 3 rutas alternativas con diferentes prioridades, tráfico y geometría real"""
    if G_full is None:
        return None, "No hay red vial cargada."
    
    routes = {}
    objectives = [
        ("🛡️ Más Segura", "safest"),
        ("⚡ Más Rápida", "fastest"),
        ("⚖️ Equilibrada", "balanced")
    ]
    
    for name, obj in objectives:
        # Recalcular pesos para cada objetivo con tráfico
        _ = precompute_edge_costs(
            G_full, predictions_data, cluster_geometries, 
            buffer_m=200, objective=obj, vehicle_type=vehicle_type,
            traffic_multiplier=traffic_multiplier
        )
        
        o_node = nearest_osm_node(G_full, origin[0], origin[1])
        d_node = nearest_osm_node(G_full, destination[0], destination[1])
        
        if o_node and d_node:
            route = route_between_nodes(G_full, o_node, d_node, edge_geometries)
            if route:
                # Ajustar tiempo con tráfico
                route['min'] = route['min'] * traffic_multiplier
                routes[name] = route
    
    return routes, None

def get_route_safety_score(route_data):
    """Calcular puntuación de seguridad de una ruta (0-100)"""
    avg_risk = route_data.get('avg_risk', 0.0)
    safety_score = max(0, 100 - (avg_risk * 1000))
    return safety_score

def get_safety_badge(risk_level):
    """Obtener badge de seguridad con estrellas"""
    if risk_level < 0.03:
        return "⭐⭐⭐⭐⭐ Excelente"
    elif risk_level < 0.05:
        return "⭐⭐⭐⭐ Muy Segura"
    elif risk_level < 0.08:
        return "⭐⭐⭐ Segura"
    elif risk_level < 0.10:
        return "⭐⭐ Precaución"
    else:
        return "⭐ Alto Riesgo"

def analyze_route_warnings(route_data, predictions_data, cluster_geometries):
    """Analizar la ruta y generar advertencias específicas"""
    warnings = []
    high_risk_zones = []
    
    coords = route_data.get('coords', [])
    for i, coord in enumerate(coords[::5]):  # Samplear cada 5 puntos
        for cluster_id, geo in cluster_geometries.items():
            if cluster_id not in predictions_data:
                continue
            
            dist_km = haversine(coord[0], coord[1], geo['lat'], geo['lon'])
            if dist_km < 0.5:  # Dentro de 500m
                risk = predictions_data[cluster_id]['probability']
                if risk >= 0.10:
                    high_risk_zones.append({
                        'cluster': cluster_id,
                        'risk': risk,
                        'position': i * 5 / len(coords) * 100  # % del recorrido
                    })
    
    if high_risk_zones:
        zones_by_position = sorted(high_risk_zones, key=lambda x: x['position'])
        for zone in zones_by_position[:3]:  # Máximo 3 advertencias
            if zone['position'] < 33:
                position_text = "al inicio del recorrido"
            elif zone['position'] < 66:
                position_text = "a mitad del recorrido"
            else:
                position_text = "cerca del destino"
            
            warnings.append({
                'type': 'high' if zone['risk'] >= 0.10 else 'medium',
                'message': f"Zona {zone['cluster']} con riesgo {zone['risk']*100:.1f}% {position_text}",
                'suggestion': "Conduce con precaución extra en esta zona"
            })
    
    return warnings

# -----------------------------
#   Mapas mejorados - CON Y SIN ZONAS DE RIESGO
# -----------------------------
def create_clean_barcelona_map():
    """Crear mapa limpio de Barcelona sin zonas de riesgo"""
    center_lat, center_lon = 41.3851, 2.1734
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
    
    # Solo añadir marcadores de lugares populares (opcional)
    for place_name, coords in list(POPULAR_PLACES.items())[:5]:  # Solo algunos lugares principales
        folium.Marker(
            location=[coords[0], coords[1]],
            popup=place_name,
            icon=folium.Icon(color='lightblue', icon='info-sign'),
            tooltip=place_name
        ).add_to(m)
    
    return m

def create_enhanced_route_map(predictions_data, cluster_geometries, route_data=None, 
                            origin=None, destination=None, show_gradient=True):
    """Mapa mejorado con gradiente de colores según riesgo y geometría real"""
    center_lat, center_lon = 41.3851, 2.1734
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB positron')
    
    # Añadir zonas de riesgo con transparencia
    for cluster_id, prediction_data in predictions_data.items():
        if cluster_id in cluster_geometries:
            coords = cluster_geometries[cluster_id]
            probability = prediction_data['probability']
            risk_level = prediction_data['risk_level']
            color = get_risk_color(probability)
            radius_meters = max(50, min(300, probability * 2000))
            
            folium.Circle(
                location=[coords['lat'], coords['lon']],
                radius=radius_meters,
                popup=folium.Popup(f"""
                    <div style='width: 200px'>
                    <b>🚦 Zona {cluster_id}</b><br>
                    <b>Probabilidad:</b> {probability*100:.2f}%<br>
                    <b>Nivel:</b> <span style='color: {color}'>{risk_level}</span><br>
                    <b>Estado:</b> {'⚠️ Accidente probable' if prediction_data['prediction'] else '✅ Sin accidente'}
                    </div>
                """, max_width=250),
                color=color, fillColor=color, fillOpacity=0.4, weight=2
            ).add_to(m)
    
    # Marcadores de origen y destino mejorados
    if origin:
        folium.Marker(
            location=[origin[0], origin[1]], 
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
            popup="🚀 Origen"
        ).add_to(m)
    
    if destination:
        folium.Marker(
            location=[destination[0], destination[1]], 
            icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa'),
            popup="🎯 Destino"
        ).add_to(m)
    
    # Dibujar ruta con geometría real y gradiente
    if route_data and 'coords' in route_data:
        coords = route_data['coords']
        risks = route_data.get('risks', [])
        
        if show_gradient and risks and len(coords) > 2:
            # Dibujar la ruta por segmentos con colores según riesgo
            # Dividir coordenadas en segmentos aproximados según los riesgos
            segments_per_risk = max(1, len(coords) // len(risks))
            
            for i, risk in enumerate(risks):
                start_idx = i * segments_per_risk
                end_idx = min((i + 1) * segments_per_risk, len(coords))
                
                if start_idx < len(coords) and end_idx <= len(coords) and start_idx < end_idx:
                    segment_coords = coords[start_idx:end_idx + 1]  # +1 para overlap
                    color = get_gradient_color(risk)
                    
                    folium.PolyLine(
                        segment_coords,
                        color=color,
                        weight=8,
                        opacity=0.8,
                        popup=f"Riesgo: {risk*100:.1f}%"
                    ).add_to(m)
        else:
            # Ruta simple sin gradiente pero con geometría real
            folium.PolyLine(
                coords, 
                weight=6, 
                opacity=0.9, 
                color='#2c3e50',
                popup="Ruta calculada"
            ).add_to(m)
    
    return m

def create_barcelona_map(predictions_data, cluster_geometries, route_coords=None, origin=None, destination=None):
    """Mapa con zonas de riesgo (cuando hay predicciones)"""
    center_lat, center_lon = 41.3851, 2.1734
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    for cluster_id, prediction_data in predictions_data.items():
        if cluster_id in cluster_geometries:
            coords = cluster_geometries[cluster_id]
            probability = prediction_data['probability']
            risk_level = prediction_data['risk_level']
            color = get_risk_color(probability)
            radius_meters = max(50, min(300, probability * 2000))
            folium.Circle(
                location=[coords['lat'], coords['lon']],
                radius=radius_meters,
                popup=folium.Popup(f"""
                    <b>Zona {cluster_id}</b><br>
                    <b>Probabilidad:</b> {probability*100:.2f}%<br>
                    <b>Nivel:</b> {risk_level}<br>
                    <b>Predicción:</b> {'Accidente probable' if prediction_data['prediction'] else 'Sin accidente'}
                """, max_width=250),
                color=color, fillColor=color, fillOpacity=0.7, weight=2
            ).add_to(m)

    if origin:
        folium.Marker(location=[origin[0], origin[1]], icon=folium.Icon(color='green', icon='play')).add_to(m)
    if destination:
        folium.Marker(location=[destination[0], destination[1]], icon=folium.Icon(color='red', icon='flag')).add_to(m)

    if route_coords and len(route_coords) >= 2:
        folium.PolyLine(route_coords, weight=6, opacity=0.9, color='#2c3e50').add_to(m)
    return m

# -----------------------------
#   Charts
# -----------------------------
def create_hourly_chart(historical_data):
    if 'accidents_by_hour' in historical_data:
        hours = list(range(24))
        accidents = [historical_data['accidents_by_hour'].get(h, 0) for h in hours]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours, y=accidents, mode='lines+markers',
            line=dict(color='#e74c3c', width=3), marker=dict(size=8),
            hovertemplate='Hora: %{x}:00<br>Accidentes: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title="Distribución de Accidentes por Hora del Día (2017-2024)",
            xaxis_title="Hora del Día", yaxis_title="Número de Accidentes (Total 8 años)",
            height=400, showlegend=False, xaxis=dict(tickmode='linear', tick0=0, dtick=2)
        )
        return fig
    return None

def create_monthly_chart(historical_data):
    if 'accidents_by_month' in historical_data:
        months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                  'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        accidents = [historical_data['accidents_by_month'].get(i+1, 0) for i in range(12)]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=months, y=accidents, marker_color='#3498db',
            hovertemplate='Mes: %{x}<br>Accidentes: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title="Distribución de Accidentes por Mes (2017-2024)",
            xaxis_title="Mes", yaxis_title="Número de Accidentes (Total 8 años)",
            height=400, showlegend=False
        )
        return fig
    return None

def create_yearly_trend_chart(historical_data):
    if 'accidents_by_year' in historical_data:
        years = sorted(historical_data['accidents_by_year'].keys())
        accidents = [historical_data['accidents_by_year'][year] for year in years]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=accidents, mode='lines+markers',
            line=dict(color='#9b59b6', width=3), marker=dict(size=10),
            hovertemplate='Año: %{x}<br>Accidentes: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title="Tendencia de Accidentes por Año", xaxis_title="Año",
            yaxis_title="Número de Accidentes", height=400, showlegend=False
        )
        return fig
    return None

def create_heatmap_hour_dow(historical_data):
    if 'hour_dow_matrix' not in historical_data:
        return None
    mat = historical_data['hour_dow_matrix']
    mat_norm = {}
    try:
        for k, inner in mat.items():
            ki = int(k)
            mat_norm[ki] = {}
            for kk, vv in (inner or {}).items():
                try:
                    mat_norm[ki][int(kk)] = int(vv)
                except Exception:
                    continue
    except Exception:
        return None

    hours = list(range(24))
    dows = list(range(7))
    outer_keys = set(mat_norm.keys())
    is_dow_outer = outer_keys.issubset(set(dows))

    z = []
    if is_dow_outer:
        for h in hours:
            row = []
            for d in dows:
                row.append(mat_norm.get(d, {}).get(h, 0))
            z.append(row)
    else:
        for h in hours:
            inner = mat_norm.get(h, {})
            row = [inner.get(d, 0) for d in dows]
            z.append(row)

    total = sum(sum(r) for r in z)
    if total == 0:
        return None

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'],
        y=[f"{h:02d}:00" for h in hours],
        colorscale='Reds',
        colorbar=dict(title='Accidentes'),
        hovertemplate='Día: %{x}<br>Hora: %{y}<br>Accidentes: %{z}<extra></extra>'
    ))
    fig.update_layout(
        title="Mapa de Calor: Accidentes por Hora y Día de la Semana",
        xaxis_title="Día de la Semana",
        yaxis_title="Hora del Día",
        height=600
    )
    return fig

# -----------------------------
#   Meteo (Open-Meteo) – Barcelona centro
# -----------------------------
BCN_LAT = 41.3874
BCN_LON = 2.1686
BCN_TZ  = "Europe/Madrid"

@st.cache_data(ttl=900, show_spinner=False)
def fetch_openmeteo_hourly(lat, lon, start_iso=None, days_ahead=14, past_days=7):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": BCN_TZ,
        "forecast_days": days_ahead,
        "past_days": past_days
    }
    try:
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        hourly = js.get("hourly", {})
        return {
            "time": hourly.get("time", []),
            "temp": hourly.get("temperature_2m", []),
            "precip": hourly.get("precipitation", []),
            "wind": hourly.get("wind_speed_10m", [])
        }
    except Exception:
        return {"time": [], "temp": [], "precip": [], "wind": []}

def get_weather_for_datetime(target_date: date, hour: int):
    data = fetch_openmeteo_hourly(BCN_LAT, BCN_LON)
    times, temps, precs, winds = data["time"], data["temp"], data["precip"], data["wind"]
    if not times:
        return 15.0, 0.0, 10.0, "Meteo no disponible (fallback)"

    target_dt = datetime(target_date.year, target_date.month, target_date.day, hour)
    target_str = target_dt.strftime("%Y-%m-%dT%H:00")

    if target_str in times:
        idx = times.index(target_str)
    else:
        def parse_t(s):
            for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(s, fmt)
                except:
                    pass
            return None
        target = parse_t(target_str + ":00")
        diffs = []
        for i, s in enumerate(times):
            dt = parse_t(s)
            diffs.append((abs((dt - target).total_seconds()) if dt else 1e12, i))
        idx = min(diffs)[1]

    t = float(temps[idx]) if idx < len(temps) else 15.0
    p = float(precs[idx]) if idx < len(precs) else 0.0
    w = float(winds[idx]) if idx < len(winds) else 10.0
    info = f"{times[idx]} → {t:.1f}°C, {p:.1f} mm, {w:.0f} km/h"
    return t, p, w, info

# ==========================
# Main App
# ==========================
def main():
    st.title("🚗 Sistema de Predicción de Accidentes de Tráfico - Barcelona")
    st.markdown("**Modelo de IA para predicción de riesgo vial en tiempo real**")
    st.markdown(f"*Datos históricos: 2017-2024 | Accidentes analizados: 67,424*")
    
    model_data = load_model()
    if model_data is None:
        st.stop()
    
    predict_risk = create_prediction_function(model_data)
    
    # Sidebar
    st.sidebar.header("🎛️ Parámetros de Predicción")
    st.sidebar.subheader("📅 Fecha y Hora")
    
    # === MODIFICACIÓN: Usar hora de Barcelona +1 ===
    today = datetime.now().date()
    max_date = today + timedelta(days=30)
    
    # Calcular fecha y hora siguiente en Barcelona
    next_date, next_hour = get_next_hour_barcelona()
    
    prediction_date = st.sidebar.date_input(
        "Fecha (máx. 30 días)", 
        value=next_date,
        min_value=today, 
        max_value=max_date
    )
    
    prediction_hour = st.sidebar.selectbox(
        "Hora", 
        options=list(range(24)), 
        index=next_hour,
        format_func=lambda x: f"{x:02d}:00"
    )
    
    day_of_week = prediction_date.weekday()
    day_names = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
    is_weekend = day_of_week >= 5
    is_holiday = is_festivo(prediction_date)
    holiday_name = get_festivo_name(prediction_date)
    
    # Botón de predicción en sidebar
    if st.sidebar.button("🔮 Actualizar Predicciones", type="primary"):
        st.session_state.update_predictions = True
    
    st.sidebar.subheader("📆 Tipo de día")
    if is_holiday and holiday_name:
        st.sidebar.info(f"🎉 Festivo: {holiday_name}")
    elif is_weekend:
        st.sidebar.info("📅 Fin de semana")
    else:
        st.sidebar.success("💼 Día laborable")

    # === METEO vertical (Open-Meteo) - Solo mostrar si hay predicciones ===
    st.sidebar.subheader("🌤️ Meteorología (Open-Meteo)")
    if st.session_state.get('predictions_data') and len(st.session_state.predictions_data) > 0:
        # Mostrar datos meteorológicos guardados de la última predicción
        if st.session_state.get('prediction_params'):
            params = st.session_state.prediction_params
            st.sidebar.write(f"**Fecha:** {params['date'].strftime('%d/%m/%Y')}")
            st.sidebar.write(f"**Hora:** {params['hour']:02d}:00")
            st.sidebar.write(f"**Temperatura:** {params['temperature']:.1f} °C")
            st.sidebar.write(f"**Precipitación:** {params['precipitation']:.1f} mm")
            st.sidebar.write(f"**Viento:** {params['wind_speed']:.0f} km/h")
            st.sidebar.caption("Centro BCN • Datos Open-Meteo")
    else:
        st.sidebar.info("📋 Selecciona fecha y hora, luego pulsa 'Actualizar Predicciones'")
        st.sidebar.write("**Fecha:** No seleccionada")
        st.sidebar.write("**Hora:** No seleccionada")
        st.sidebar.write("**Temperatura:** -- °C")
        st.sidebar.write("**Precipitación:** -- mm")
        st.sidebar.write("**Viento:** -- km/h")
    
    # Históricos
    historical_data = model_data.get('historical_data', {})
    if historical_data:
        st.sidebar.subheader("📊 Estadísticas Históricas")
        st.sidebar.metric("Total Accidentes", f"{historical_data.get('total_accidents', 0):,}")
        st.sidebar.metric("Promedio Anual", f"{historical_data.get('yearly_trends', {}).get('avg_per_year', 0):,.0f}")
        if 'most_dangerous_clusters' in historical_data:
            st.sidebar.write("**Top 3 Zonas más Peligrosas:**")
            for i, (cluster_id, accidents) in enumerate(list(historical_data['most_dangerous_clusters'].items())[:3], 1):
                st.sidebar.write(f"{i}. Zona {cluster_id}: {accidents:,}")
    
    # State - INICIALIZACIÓN LIMPIA
    if 'predictions_data' not in st.session_state:
        st.session_state.predictions_data = {}
        st.session_state.update_predictions = False
        st.session_state.prediction_params = None
        st.session_state.predictions_calculated = False
    
    # Inicializar rutas favoritas
    if 'favorite_routes' not in st.session_state:
        st.session_state.favorite_routes = load_favorite_routes()
    
    # Predicciones - SOLO cuando el usuario pulse el botón
    if st.session_state.get('update_predictions', False):
        # Obtener datos meteorológicos actuales para la predicción
        temperature, precipitation, wind_speed, meteo_info = get_weather_for_datetime(prediction_date, prediction_hour)
        
        with st.spinner(f"Calculando predicciones para {prediction_date.strftime('%d/%m/%Y')} {prediction_hour:02d}:00..."):
            predictions_data = {}
            cluster_geometries = model_data.get('cluster_geometries', {})
            probabilities_debug = []
            accidents_by_cluster = historical_data.get('accidents_by_cluster', {})
            max_historical_accidents = max(accidents_by_cluster.values()) if accidents_by_cluster else 1000
            
            for cluster_id in cluster_geometries.keys():
                prediction = predict_risk(
                    cluster_id=cluster_id,
                    hour=prediction_hour,
                    temperature=temperature,
                    precipitation=precipitation,
                    wind_speed=wind_speed,
                    date_obj=prediction_date
                )
                # Ajuste suave por histórico del cluster
                if cluster_id in accidents_by_cluster:
                    historical_factor = accidents_by_cluster[cluster_id] / max_historical_accidents
                    prediction['probability'] *= (0.7 + historical_factor * 0.6)
                    prediction['probability'] = min(0.99, prediction['probability'])
                # Recalcular nivel
                prediction['risk_level'] = ('Alto' if prediction['probability'] >= 0.10
                                            else 'Medio' if prediction['probability'] >= 0.07 else 'Bajo')
                prediction['prediction'] = int(prediction['probability'] >= model_data['optimal_threshold'])
                predictions_data[cluster_id] = prediction
                probabilities_debug.append(prediction['probability'])
            
            # Guardar resultados y marcar como calculadas
            st.session_state.predictions_data = predictions_data
            st.session_state.prediction_params = {
                'date': prediction_date,
                'hour': prediction_hour,
                'temperature': temperature,
                'precipitation': precipitation,
                'wind_speed': wind_speed
            }
            st.session_state.predictions_calculated = True
            st.session_state.update_predictions = False
            
            # Debug info solo en sidebar cuando hay predicciones
            st.sidebar.write("📊 Debug - Probabilidades:")
            st.sidebar.write(f"Min: {min(probabilities_debug):.3f}")
            st.sidebar.write(f"Max: {max(probabilities_debug):.3f}")
            st.sidebar.write(f"Promedio: {np.mean(probabilities_debug):.3f}")
            st.sidebar.write(f"Desv. Est: {np.std(probabilities_debug):.3f}")
            
            # Mostrar mensaje de éxito
            st.success(f"✅ Predicciones actualizadas para {prediction_date.strftime('%d/%m/%Y')} {prediction_hour:02d}:00")
            st.rerun()
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("🗺️ Mapa de Riesgo en Tiempo Real")
        
        # === MODIFICACIÓN: Mostrar mapa limpio hasta que se actualicen predicciones ===
        if (st.session_state.get('predictions_data') and 
            len(st.session_state.predictions_data) > 0 and 
            st.session_state.get('predictions_calculated', False)):
            cluster_geometries = model_data.get('cluster_geometries', {})
            risk_map = create_barcelona_map(st.session_state.predictions_data, cluster_geometries)
            map_data = st_folium(risk_map, width=700, height=500)
            
            # Mostrar información de la predicción
            if st.session_state.get('prediction_params'):
                params = st.session_state.prediction_params
                st.info(f"📅 Predicciones para: {params['date'].strftime('%d/%m/%Y')} a las {params['hour']:02d}:00")
        else:
            # Mapa limpio sin zonas de riesgo
            clean_map = create_clean_barcelona_map()
            map_data = st_folium(clean_map, width=700, height=500)
            st.info("👆 Selecciona fecha y hora en el panel lateral, luego pulsa 'Actualizar Predicciones' para ver las zonas de riesgo")
            
    with col2:
        st.header("📊 Estadísticas en Tiempo Real")
        if (st.session_state.get('predictions_data') and 
            len(st.session_state.predictions_data) > 0 and 
            st.session_state.get('predictions_calculated', False)):
            predictions = list(st.session_state.predictions_data.values())
            probabilities = [p['probability'] for p in predictions]
            risk_levels = [p['risk_level'] for p in predictions]
            high_risk = sum(1 for r in risk_levels if r == 'Alto')
            medium_risk = sum(1 for r in risk_levels if r == 'Medio')
            low_risk = sum(1 for r in risk_levels if r == 'Bajo')
            avg_probability = np.mean(probabilities)
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Alto Riesgo", high_risk, delta=f"{high_risk/101*100:.1f}%")
                st.metric("Riesgo Medio", medium_risk, delta=f"{medium_risk/101*100:.1f}%")
            with col2b:
                st.metric("Bajo Riesgo", low_risk, delta=f"{low_risk/101*100:.1f}%")
                st.metric("Prob. Promedio", f"{avg_probability*100:.1f}%")
            st.subheader("🎨 Leyenda")
            st.markdown("🔴 **Alto Riesgo** (≥10%)")
            st.markdown("🟠 **Riesgo Medio** (7-10%)")
            st.markdown("🟢 **Bajo Riesgo** (<7%)")
            st.subheader("⚠️ Top Zonas de Riesgo Actual")
            sorted_areas = sorted(
                st.session_state.predictions_data.items(),
                key=lambda x: x[1]['probability'],
                reverse=True
            )[:5]
            for i, (area_id, data_) in enumerate(sorted_areas, 1):
                risk_emoji = "🔴" if data_['risk_level'] == 'Alto' else "🟠" if data_['risk_level'] == 'Medio' else "🟢"
                st.write(f"{risk_emoji} **{i}.** Zona {area_id}: {data_['probability']*100:.2f}%")
        else:
            st.info("📋 Actualiza las predicciones para ver estadísticas en tiempo real")
            st.write("**Estadísticas disponibles después de actualizar predicciones:**")
            st.write("- Distribución de zonas por nivel de riesgo")
            st.write("- Probabilidad promedio de accidentes")
            st.write("- Top 5 zonas más peligrosas")
            st.write("- Leyenda de colores del mapa")


    # ---------------------------
    # RECOMENDADOR DE RUTA OPTIMIZADO CON GEOMETRÍA REAL
    # ---------------------------
    st.header("🚦 Planificador de Ruta Segura - Geometría Real")
    st.markdown("Calcula la mejor ruta evitando zonas de alto riesgo con precisión en carreteras")
    
    # Inicializar estado
    if 'route_origin' not in st.session_state:
        st.session_state.route_origin = None
    if 'route_destination' not in st.session_state:
        st.session_state.route_destination = None
    if 'route_alternatives' not in st.session_state:
        st.session_state.route_alternatives = None
    if 'selected_route' not in st.session_state:
        st.session_state.selected_route = None
    
    # TABS PARA DIFERENTES MÉTODOS DE SELECCIÓN
    st.subheader("📍 Selección de Origen y Destino")
    tab_places, tab_address, tab_map, tab_favorites = st.tabs([
        "🏛️ Lugares Populares", 
        "🔍 Buscar Dirección",
        "🗺️ Clic en Mapa",
        "⭐ Rutas Favoritas"
    ])
    
    # TAB 1: LUGARES POPULARES
    with tab_places:
        col_places = st.columns(2)
        with col_places[0]:
            st.write("**🟢 Seleccionar Origen:**")
            place_cols = st.columns(3)
            for idx, (place, coords) in enumerate(list(POPULAR_PLACES.items())[:9]):
                col = place_cols[idx % 3]
                with col:
                    if st.button(place, key=f"pop_orig_{idx}", use_container_width=True):
                        st.session_state.route_origin = coords
                        st.success(f"✅ Origen: {place}")
                        st.session_state.route_alternatives = None
        
        with col_places[1]:
            st.write("**🔴 Seleccionar Destino:**")
            place_cols = st.columns(3)
            for idx, (place, coords) in enumerate(list(POPULAR_PLACES.items())[9:]):
                col = place_cols[idx % 3]
                with col:
                    if st.button(place, key=f"pop_dest_{idx}", use_container_width=True):
                        st.session_state.route_destination = coords
                        st.success(f"✅ Destino: {place}")
                        st.session_state.route_alternatives = None
    
    # TAB 2: BÚSQUEDA DE DIRECCIONES CON AUTOCOMPLETADO
    with tab_address:
        if GEOPY_AVAILABLE:
            st.markdown("### 🔍 Búsqueda Inteligente de Direcciones")
            st.caption("🏙️ Busca solo en Barcelona ciudad • Escribe al menos 3 caracteres")
            
            col_addr = st.columns(2)
            
            with col_addr[0]:
                st.write("**🟢 Buscar dirección de origen:**")
                origin_coords = create_address_autocomplete(
                    label="origen",
                    key_prefix="origin_addr",
                    placeholder_text="Ej: Carrer de Balmes 123"
                )
                
                # Botón para establecer origen
                if origin_coords and st.button("Establecer como Origen", key="set_origin", use_container_width=True):
                    st.session_state.route_origin = origin_coords
                    st.session_state.route_alternatives = None
                    st.success("✅ Origen establecido correctamente")
            
            with col_addr[1]:
                st.write("**🔴 Buscar dirección de destino:**")
                dest_coords = create_address_autocomplete(
                    label="destino", 
                    key_prefix="dest_addr",
                    placeholder_text="Ej: Passeig de Gràcia 45"
                )
                
                # Botón para establecer destino
                if dest_coords and st.button("Establecer como Destino", key="set_dest", use_container_width=True):
                    st.session_state.route_destination = dest_coords
                    st.session_state.route_alternatives = None
                    st.success("✅ Destino establecido correctamente")
        else:
            st.warning("Instala 'geopy' para habilitar la búsqueda de direcciones: pip install geopy")
    
    # TAB 3: SELECCIÓN EN MAPA
    with tab_map:
        st.info("💡 Haz clic en el mapa principal arriba para seleccionar puntos")
        
        # Detectar clics en el mapa
        last_click = None
        if 'map_data' in locals() and isinstance(map_data, dict) and 'last_clicked' in map_data and map_data['last_clicked'] is not None:
            last_click = (map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
        
        if last_click:
            st.write(f"**📍 Último punto seleccionado:** {last_click[0]:.4f}, {last_click[1]:.4f}")
            col_map_select = st.columns(2)
            with col_map_select[0]:
                if st.button("Usar como Origen", key="map_orig", use_container_width=True):
                    st.session_state.route_origin = last_click
                    st.session_state.route_alternatives = None
                    st.success("✅ Origen establecido")
            with col_map_select[1]:
                if st.button("Usar como Destino", key="map_dest", use_container_width=True):
                    st.session_state.route_destination = last_click
                    st.session_state.route_alternatives = None
                    st.success("✅ Destino establecido")
        else:
            st.info("Esperando clic en el mapa...")
    
    # TAB 4: RUTAS FAVORITAS
    with tab_favorites:
        if st.session_state.favorite_routes:
            st.write("**📌 Tus rutas guardadas:**")
            for route_id, route_info in st.session_state.favorite_routes.items():
                col_fav = st.columns([3, 1, 1])
                with col_fav[0]:
                    st.write(f"**{route_info['name']}**")
                    st.caption(f"Guardada: {route_info.get('saved_date', 'Desconocida')}")
                with col_fav[1]:
                    if st.button("Cargar", key=f"load_fav_{route_id}"):
                        st.session_state.route_origin = tuple(route_info['origin'])
                        st.session_state.route_destination = tuple(route_info['destination'])
                        st.session_state.route_alternatives = None
                        st.success(f"✅ Ruta '{route_info['name']}' cargada")
                with col_fav[2]:
                    if st.button("🗑️", key=f"del_fav_{route_id}"):
                        del st.session_state.favorite_routes[route_id]
                        save_favorite_routes(st.session_state.favorite_routes)
                        st.rerun()
        else:
            st.info("No tienes rutas favoritas guardadas aún")
    
    # Mostrar selección actual
    st.markdown("---")
    col_current = st.columns(2)
    with col_current[0]:
        if st.session_state.route_origin:
            st.success(f"**🟢 Origen:** {st.session_state.route_origin[0]:.4f}, {st.session_state.route_origin[1]:.4f}")
        else:
            st.warning("**🟢 Origen:** No seleccionado")
    
    with col_current[1]:
        if st.session_state.route_destination:
            st.success(f"**🔴 Destino:** {st.session_state.route_destination[0]:.4f}, {st.session_state.route_destination[1]:.4f}")
        else:
            st.warning("**🔴 Destino:** No seleccionado")
    
    # Configuración de ruta
    st.subheader("⚙️ Configuración de Ruta")
    col_config = st.columns(4)
    
    with col_config[0]:
        vehicle_type = st.selectbox(
            "Tipo de vehículo",
            options=list(VEHICLE_SPEED_FACTORS.keys()),
            index=0,
            help="El tipo de vehículo afecta la velocidad y el riesgo"
        )
    
    with col_config[1]:
        # Sincronizar con hora de predicción
        departure_hour = prediction_hour
        departure_time = datetime.strptime(f"{departure_hour:02d}:00", "%H:%M").time()
        st.write("**Hora de salida:**")
        st.info(f"🕐 {departure_time.strftime('%H:%M')} (sincronizada con predicción)")
    
    with col_config[2]:
        consider_traffic = st.checkbox(
            "Considerar tráfico",
            value=True,
            help="Ajustar tiempos según tráfico actual"
        )
    
    with col_config[3]:
        if st.button("🗑️ Limpiar Selección"):
            st.session_state.route_origin = None
            st.session_state.route_destination = None
            st.session_state.route_alternatives = None
            st.session_state.selected_route = None
            st.rerun()
    
    # Obtener condiciones de tráfico si está habilitado
    traffic_info = None
    traffic_multiplier = 1.0
    if consider_traffic and st.session_state.route_origin and st.session_state.route_destination:
        traffic_info = get_traffic_conditions(
            st.session_state.route_origin[0], st.session_state.route_origin[1],
            st.session_state.route_destination[0], st.session_state.route_destination[1],
            departure_hour
        )
        traffic_multiplier = traffic_info['multiplier']
        
        # Mostrar información de tráfico
        st.markdown(f"""
        <div class='traffic-info'>
            <b>🚦 Condiciones de Tráfico Actuales</b><br>
            {traffic_info['level']} - {traffic_info['description']}
        </div>
        """, unsafe_allow_html=True)
    
    # Cargar grafo solo cuando sea necesario
    can_route = (st.session_state.route_origin is not None) and (st.session_state.route_destination is not None)
    
    if can_route:
        # Botón para calcular rutas
        if st.button("🔍 Buscar Rutas Precisas", type="primary", use_container_width=True):
            with st.spinner("🗺️ Cargando red vial detallada de Barcelona..."):
                G, edge_geometries, g_msg = load_or_build_graph()
                if g_msg:
                    st.error(g_msg)
                elif G and edge_geometries:
                    with st.spinner("🧮 Calculando rutas con geometría real..."):
                        routes, error = calculate_alternative_routes(
                            G, edge_geometries,
                            st.session_state.predictions_data,
                            model_data.get('cluster_geometries', {}),
                            st.session_state.route_origin,
                            st.session_state.route_destination,
                            vehicle_type,
                            traffic_multiplier
                        )
                        if error:
                            st.error(error)
                        elif routes:
                            st.session_state.route_alternatives = routes
                            st.session_state.selected_route = None
                            
                            # Información sobre la precisión del routing
                            st.success("✅ Rutas calculadas con geometría real de carreteras")
                            st.info("🛣️ Las rutas siguen exactamente el trazado de las carreteras de Barcelona")
        
        # Mostrar alternativas si existen
        if st.session_state.route_alternatives:
            st.subheader("🔄 Comparación de Rutas Precisas")
            
            tabs = st.tabs(list(st.session_state.route_alternatives.keys()))
            
            for tab, (route_name, route_data) in zip(tabs, st.session_state.route_alternatives.items()):
                with tab:
                    # Tarjeta de resumen de ruta
                    avg_risk = route_data.get('avg_risk', 0.0)
                    safety_score = get_route_safety_score(route_data)
                    safety_badge = get_safety_badge(avg_risk)
                    
                    # Ajustar tiempo con tráfico
                    adjusted_time = route_data['min']
                    
                    st.markdown(f"""
                    <div class='route-card'>
                        <h3>{route_name} - Geometría Real</h3>
                        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;'>
                            <div style='text-align: center;'>
                                <h4>📏 Distancia</h4>
                                <p style='font-size: 24px; margin: 0;'>{route_data['km']:.1f} km</p>
                            </div>
                            <div style='text-align: center;'>
                                <h4>⏱️ Tiempo</h4>
                                <p style='font-size: 24px; margin: 0;'>{adjusted_time:.0f} min</p>
                            </div>
                            <div style='text-align: center;'>
                                <h4>🛡️ Seguridad</h4>
                                <p style='font-size: 20px; margin: 0;'>{safety_score:.0f}/100</p>
                            </div>
                            <div style='text-align: center;'>
                                <h4>⭐ Valoración</h4>
                                <p style='font-size: 18px; margin: 0;'>{safety_badge}</p>
                            </div>
                        </div>
                        <p style='margin-top: 15px; font-size: 14px;'>🛣️ Ruta calculada con geometría real de carreteras</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mapa de la ruta con geometría real
                    route_map = create_enhanced_route_map(
                        st.session_state.predictions_data,
                        model_data.get('cluster_geometries', {}),
                        route_data,
                        st.session_state.route_origin,
                        st.session_state.route_destination,
                        show_gradient=True
                    )
                    st_folium(route_map, height=400, key=f"map_{route_name}")
                    
                    # Botón de selección
                    col_actions = st.columns(2)
                    with col_actions[0]:
                        if st.button(f"✅ Seleccionar {route_name}", key=f"select_{route_name}", use_container_width=True):
                            st.session_state.selected_route = route_data
                            st.session_state.selected_route['name'] = route_name
                            st.success(f"✅ Has seleccionado la ruta {route_name}")
                    
                    with col_actions[1]:
                        # Exportar a Google Maps
                        google_url = generate_google_maps_url(
                            st.session_state.route_origin,
                            st.session_state.route_destination
                        )
                        st.markdown(f"[🗺️ Abrir en Google Maps]({google_url})", unsafe_allow_html=True)
            
            # Mostrar ruta seleccionada con más detalle
            if st.session_state.selected_route:
                st.markdown("---")
                st.subheader("📋 Ruta Seleccionada - Detalles Completos")
                
                selected = st.session_state.selected_route
                col_details = st.columns(5)
                
                with col_details[0]:
                    st.metric("Distancia Total", f"{selected['km']:.2f} km")
                
                with col_details[1]:
                    adjusted_time = selected['min']
                    st.metric("Tiempo Estimado", f"{adjusted_time:.0f} min")
                
                with col_details[2]:
                    avg_speed = (selected['km'] / adjusted_time) * 60 if adjusted_time > 0 else 0
                    st.metric("Velocidad Media", f"{avg_speed:.0f} km/h")
                
                with col_details[3]:
                    st.metric("Índice de Seguridad", f"{get_route_safety_score(selected):.0f}/100")
                
                with col_details[4]:
                    # Guardar como favorita
                    route_name = st.text_input("Nombre para guardar:", placeholder="Casa-Trabajo", key="save_route_name")
                    if st.button("⭐ Guardar", key="save_fav"):
                        if route_name:
                            route_id = hashlib.md5(f"{route_name}{datetime.now()}".encode()).hexdigest()[:8]
                            st.session_state.favorite_routes[route_id] = {
                                'name': route_name,
                                'origin': list(st.session_state.route_origin),
                                'destination': list(st.session_state.route_destination),
                                'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'vehicle': vehicle_type,
                                'safety_score': get_route_safety_score(selected)
                            }
                            save_favorite_routes(st.session_state.favorite_routes)
                            st.success(f"✅ Ruta '{route_name}' guardada en favoritos")
                
                # Información adicional basada en el horario
                arrival_time = (datetime.combine(date.today(), departure_time) + 
                              timedelta(minutes=int(adjusted_time))).time()
                st.info(f"🕐 **Salida:** {departure_time.strftime('%H:%M')} → **Llegada estimada:** {arrival_time.strftime('%H:%M')}")
                
                # Análisis de precisión de la ruta
                if 'coords' in selected and len(selected['coords']) > len(selected.get('basic_coords', [])):
                    improvement = len(selected['coords']) - len(selected.get('basic_coords', []))
                    st.success(f"🎯 **Geometría mejorada:** +{improvement} puntos adicionales para mayor precisión en curvas")
    
    else:
        st.info("👆 Selecciona un origen y destino usando cualquiera de los métodos disponibles")
    
    # Tabs para análisis histórico
    st.header("📈 Análisis Histórico Completo")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Por Hora", "📅 Por Mes", "📈 Tendencia Anual", "🔥 Mapa de Calor"])
    
    with tab1:
        hourly_chart = create_hourly_chart(historical_data)
        if hourly_chart:
            st.plotly_chart(hourly_chart, use_container_width=True)
            if 'peak_hours' in historical_data:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🔴 Horas Más Peligrosas")
                    for hour, accidents in list(historical_data['peak_hours'].items())[:5]:
                        st.write(f"**{hour}:00** - {accidents:,} accidentes")
                with col2:
                    st.subheader("🟢 Horas Más Seguras")
                    for hour, accidents in list(historical_data['safest_hours'].items())[:5]:
                        st.write(f"**{hour}:00** - {accidents:,} accidentes")
    
    with tab2:
        monthly_chart = create_monthly_chart(historical_data)
        if monthly_chart:
            st.plotly_chart(monthly_chart, use_container_width=True)
            if 'seasonal_patterns' in historical_data:
                st.subheader("🌍 Patrones Estacionales")
                seasons = historical_data['seasonal_patterns']
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("❄️ Invierno", f"{seasons.get('winter', 0):,}")
                with col2: st.metric("🌸 Primavera", f"{seasons.get('spring', 0):,}")
                with col3: st.metric("☀️ Verano", f"{seasons.get('summer', 0):,}")
                with col4: st.metric("🍂 Otoño", f"{seasons.get('autumn', 0):,}")
    
    with tab3:
        yearly_chart = create_yearly_trend_chart(historical_data)
        if yearly_chart:
            st.plotly_chart(yearly_chart, use_container_width=True)
            if 'yearly_trends' in historical_data:
                trends = historical_data['yearly_trends']
                slope = trends.get('trend_slope', 0)
                if slope > 0: trend_text = f"📈 Tendencia creciente: +{slope:.1f} accidentes/año"
                elif slope < 0: trend_text = f"📉 Tendencia decreciente: {slope:.1f} accidentes/año"
                else: trend_text = "➡️ Tendencia estable"
                st.markdown(f"**{trend_text}**")
    
    with tab4:
        heatmap = create_heatmap_hour_dow(historical_data)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
            st.markdown("**Interpretación:** Los colores más intensos indican más accidentes. "
                       "Permite identificar patrones como horas pico en días laborables.")

if __name__ == "__main__":
    main()