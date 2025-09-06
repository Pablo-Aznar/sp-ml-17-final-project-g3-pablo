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

# === NUEVOS IMPORTS para ruteo vial y ETA ===
import os
import re
import networkx as nx
from math import radians, sin, cos, asin, sqrt

# === Open-Meteo ===
import requests

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
    "🛍️ La Rambla": (41.3810, 2.1730)
}

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

# Estilos CSS personalizados
st.markdown("""
<style>
    .route-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .safety-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .high-risk { background-color: #e74c3c; }
    .medium-risk { background-color: #f39c12; }
    .low-risk { background-color: #27ae60; }
</style>
""", unsafe_allow_html=True)

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
            'risk_level': 'Alto' if prob >= 0.10 else ('Medio' if prob >= 0.05 else 'Bajo')
        }

    return predict_risk

def get_risk_color(probability):
    if probability >= 0.10:
        return '#e74c3c'
    elif probability >= 0.05:
        return '#f39c12'
    else:
        return '#27ae60'

def get_gradient_color(risk_value):
    """Obtener color gradiente según el riesgo (verde->amarillo->rojo)"""
    if risk_value <= 0.05:
        # Verde a amarillo
        ratio = risk_value / 0.05
        r = int(39 + (243 - 39) * ratio)
        g = int(174 + (156 - 174) * ratio)
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
#   Red vial OSMnx
# -----------------------------
GRAPH_DIR = "./data/graph"
GRAPH_PATH = os.path.join(GRAPH_DIR, "barcelona_drive.graphml")

@st.cache_data(show_spinner=True)
def load_or_build_graph():
    if not OSMNX_AVAILABLE:
        return None, "OSMnx no está instalado. Ejecuta: pip install osmnx"
    os.makedirs(GRAPH_DIR, exist_ok=True)
    if os.path.exists(GRAPH_PATH):
        try:
            G = ox.load_graphml(GRAPH_PATH)
            return G, None
        except Exception as e:
            return None, f"Error cargando graphml cacheado: {e}"
    try:
        G = ox.graph_from_place("Barcelona, Spain", network_type="drive", simplify=True)
        ox.save_graphml(G, GRAPH_PATH)
        return G, None
    except Exception as e:
        return None, f"No se pudo descargar la red de Barcelona: {e}"

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
    "residential": 30, "service": 20, "living_street": 10
}

VEHICLE_SPEED_FACTORS = {
    "🚗 Coche": 1.0,
    "🏍️ Moto": 1.1,  # Motos pueden ir un poco más rápido en tráfico
    "🚲 Bicicleta": 0.3,  # Mucho más lento
    "🚚 Camión": 0.85,  # Más lento que coches
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

# ====== EJE VIAL: subgrafo con grandes ejes ======
EJE_NAME_PATTERNS = [
    "gran via", "gran via de les corts", "granvia",
    "avinguda diagonal", "diagonal",
    "avinguda meridiana", "meridiana",
    "ronda del litoral", "ronda litoral", "b-10",
    "ronda de dalt", "ronda dalt", "b-20",
]
EJE_REF_PATTERNS = ["B-10", "B-20", "B-23", "C-31", "C31", "B10", "B20", "B23"]
EJE_HIGHWAY_TYPES = {"motorway", "trunk", "primary"}

def _edge_has_pattern(value, patterns):
    if value is None:
        return False
    vals = value if isinstance(value, list) else [value]
    for v in vals:
        s = str(v).lower()
        if any(pat in s for pat in [p.lower() for p in patterns]):
            return True
    return False

def is_axis_edge(data):
    if _edge_has_pattern(data.get("name"), EJE_NAME_PATTERNS):
        return True
    if _edge_has_pattern(data.get("ref"), EJE_REF_PATTERNS):
        return True
    hw = data.get("highway")
    if isinstance(hw, list):
        if any(h in EJE_HIGHWAY_TYPES for h in hw):
            return True
    else:
        if hw in EJE_HIGHWAY_TYPES:
            return True
    return False

def build_axis_subgraph(G):
    edges_to_keep = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if is_axis_edge(data):
            edges_to_keep.append((u, v, k))
    if not edges_to_keep:
        return None
    G_axis = G.edge_subgraph(edges_to_keep).copy()
    return G_axis

# ====== Riesgo → pesos de ruta (MEJORADO) ======
def risk_factor_from_prob(prob, vehicle_type="🚗 Coche"):
    """Penalización: más fuerte para notar el cambio con el slider."""
    base_factor = 1.0
    if prob >= 0.10:   # alto
        base_factor = 7.0
    elif prob >= 0.05: # medio
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

def precompute_edge_costs(G, predictions_data, cluster_geometries, buffer_m=200.0, objective="balanced", vehicle_type="🚗 Coche"):
    """
    Para cada arista:
      - tiempo base por longitud/velocidad.
      - busca TODOS los clusters dentro del buffer usando KD-Tree (si hay) y toma la prob MAX.
      - asigna weight según objetivo y factor de riesgo.
    Devuelve: número de aristas penalizadas (debug).
    """
    if G is None or predictions_data is None or cluster_geometries is None:
        return 0

    tree, pts, probs = build_cluster_kdtree(cluster_geometries, predictions_data)

    penalized = 0
    # m → grados lat (aprox)
    radius_deg = buffer_m / 111000.0

    for u, v, k, data in G.edges(keys=True, data=True):
        length_m = float(data.get('length', 0.0))
        length_km = max(1e-4, length_m / 1000.0)
        speed_kmh = max(5.0, infer_speed_kmh(data, vehicle_type))
        time_min = (length_km / speed_kmh) * 60.0

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
            # fallback lineal
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

def nearest_node_on_graph(G, lat, lon):
    return nearest_osm_node(G, lat, lon)

def route_between_nodes(G, src, dst):
    try:
        path = nx.shortest_path(G, src, dst, weight='route_weight')
    except nx.NetworkXNoPath:
        return None
    coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
    total_km = 0.0
    total_time_min = 0.0
    risks = []
    
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        best = None
        best_w = 1e18
        for k in G[u][v].keys():
            w = G[u][v][k].get('route_weight', None)
            if w is not None and w < best_w:
                best_w = w
                best = G[u][v][k]
        if best is not None:
            total_km += best.get('length_km', 0.0)
            total_time_min += best.get('time_min', 0.0)
            risks.append(best.get('risk_prob', 0.0))
    
    return {
        "path": path, 
        "coords": coords, 
        "km": total_km, 
        "min": total_time_min,
        "risks": risks,
        "avg_risk": np.mean(risks) if risks else 0.0
    }

def calculate_alternative_routes(G_full, predictions_data, cluster_geometries,
                                origin, destination, vehicle_type="🚗 Coche"):
    """Calcular 3 rutas alternativas con diferentes prioridades"""
    if G_full is None:
        return None, "No hay red vial cargada."
    
    routes = {}
    objectives = [
        ("🛡️ Más Segura", "safest"),
        ("⚡ Más Rápida", "fastest"),
        ("⚖️ Equilibrada", "balanced")
    ]
    
    for name, obj in objectives:
        # Recalcular pesos para cada objetivo
        _ = precompute_edge_costs(G_full, predictions_data, cluster_geometries, 
                                 buffer_m=200, objective=obj, vehicle_type=vehicle_type)
        
        o_node = nearest_node_on_graph(G_full, origin[0], origin[1])
        d_node = nearest_node_on_graph(G_full, destination[0], destination[1])
        
        if o_node and d_node:
            route = route_between_nodes(G_full, o_node, d_node)
            if route:
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
    
    # Analizar cada segmento de la ruta
    coords = route_data.get('coords', [])
    for i, coord in enumerate(coords[::5]):  # Samplear cada 5 puntos
        # Buscar clusters cercanos
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
    
    # Generar advertencias
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
#   Mapas mejorados
# -----------------------------
def create_enhanced_route_map(predictions_data, cluster_geometries, route_data=None, 
                            origin=None, destination=None, show_gradient=True):
    """Mapa mejorado con gradiente de colores según riesgo"""
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
    
    # Dibujar ruta con gradiente si está disponible
    if route_data and 'coords' in route_data:
        coords = route_data['coords']
        risks = route_data.get('risks', [])
        
        if show_gradient and risks:
            # Dibujar la ruta por segmentos con colores según riesgo
            for i in range(len(coords)-1):
                segment = [coords[i], coords[i+1]]
                risk = risks[i] if i < len(risks) else 0.0
                color = get_gradient_color(risk)
                
                folium.PolyLine(
                    segment,
                    color=color,
                    weight=8,
                    opacity=0.8
                ).add_to(m)
        else:
            # Ruta simple sin gradiente
            folium.PolyLine(
                coords, 
                weight=6, 
                opacity=0.9, 
                color='#2c3e50'
            ).add_to(m)
    
    return m

def create_barcelona_map(predictions_data, cluster_geometries, route_coords=None, origin=None, destination=None):
    """Mapa básico (compatibilidad hacia atrás)"""
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
    today = datetime.now().date()
    max_date = today + timedelta(days=30)
    prediction_date = st.sidebar.date_input(
        "Fecha (máx. 30 días)", value=today, min_value=today, max_value=max_date
    )
    prediction_hour = st.sidebar.selectbox(
        "Hora", options=list(range(24)), index=datetime.now().hour,
        format_func=lambda x: f"{x:02d}:00"
    )
    day_of_week = prediction_date.weekday()
    day_names = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']
    is_weekend = day_of_week >= 5
    is_holiday = is_festivo(prediction_date)
    holiday_name = get_festivo_name(prediction_date)

    st.sidebar.subheader("📆 Tipo de día")
    if is_holiday and holiday_name:
        st.sidebar.info(f"🎉 Festivo: {holiday_name}")
    elif is_weekend:
        st.sidebar.info("📅 Fin de semana")
    else:
        st.sidebar.success("💼 Día laborable")

    # === METEO vertical (Open-Meteo) ===
    st.sidebar.subheader("🌤️ Meteorología (Open-Meteo)")
    temperature, precipitation, wind_speed, meteo_info = get_weather_for_datetime(prediction_date, prediction_hour)
    st.sidebar.write(f"**Hora consultada:** {meteo_info.split(' → ')[0]}")
    st.sidebar.write(f"**Temperatura:** {temperature:.1f} °C")
    st.sidebar.write(f"**Precipitación:** {precipitation:.1f} mm")
    st.sidebar.write(f"**Viento:** {wind_speed:.0f} km/h")
    st.sidebar.caption("Centro BCN • Datos Open-Meteo")

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
    
    # Botón de predicción
    if st.sidebar.button("🔮 Actualizar Predicciones", type="primary"):
        st.session_state.update_predictions = True
        st.session_state.prediction_params = {
            'date': prediction_date,
            'hour': prediction_hour,
            'temperature': temperature,
            'precipitation': precipitation,
            'wind_speed': wind_speed
        }
    
    # State
    if 'predictions_data' not in st.session_state:
        st.session_state.predictions_data = {}
        st.session_state.update_predictions = True
        st.session_state.prediction_params = None
    
    current_params = {
        'date': prediction_date,
        'hour': prediction_hour,
        'temperature': temperature,
        'precipitation': precipitation,
        'wind_speed': wind_speed
    }
    if st.session_state.prediction_params != current_params:
        st.session_state.update_predictions = True
        st.session_state.prediction_params = current_params
    
    # Predicciones
    if st.session_state.get('update_predictions', False):
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
                                            else 'Medio' if prediction['probability'] >= 0.05 else 'Bajo')
                prediction['prediction'] = int(prediction['probability'] >= model_data['optimal_threshold'])
                predictions_data[cluster_id] = prediction
                probabilities_debug.append(prediction['probability'])
            
            st.session_state.predictions_data = predictions_data
            st.session_state.update_predictions = False
            
            st.sidebar.write("📊 Debug - Probabilidades:")
            st.sidebar.write(f"Min: {min(probabilities_debug):.3f}")
            st.sidebar.write(f"Max: {max(probabilities_debug):.3f}")
            st.sidebar.write(f"Promedio: {np.mean(probabilities_debug):.3f}")
            st.sidebar.write(f"Desv. Est: {np.std(probabilities_debug):.3f}")
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("🗺️ Mapa de Riesgo en Tiempo Real")
        if st.session_state.predictions_data:
            cluster_geometries = model_data.get('cluster_geometries', {})
            risk_map = create_barcelona_map(st.session_state.predictions_data, cluster_geometries)
            map_data = st_folium(risk_map, width=700, height=500)
            st.info(f"📅 Predicciones para: {prediction_date.strftime('%d/%m/%Y')} a las {prediction_hour:02d}:00")
        else:
            st.info("Haz clic en 'Actualizar Predicciones' para ver el mapa de riesgo")
    with col2:
        st.header("📊 Estadísticas en Tiempo Real")
        if st.session_state.predictions_data:
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
            st.markdown("🟠 **Riesgo Medio** (5-10%)")
            st.markdown("🟢 **Bajo Riesgo** (<5%)")
            st.subheader("⚠️ Top Zonas de Riesgo Actual")
            sorted_areas = sorted(
                st.session_state.predictions_data.items(),
                key=lambda x: x[1]['probability'],
                reverse=True
            )[:5]
            for i, (area_id, data_) in enumerate(sorted_areas, 1):
                risk_emoji = "🔴" if data_['risk_level'] == 'Alto' else "🟠" if data_['risk_level'] == 'Medio' else "🟢"
                st.write(f"{risk_emoji} **{i}.** Zona {area_id}: {data_['probability']*100:.2f}%")

    # ---------------------------
    # RECOMENDADOR DE RUTA MEJORADO
    # ---------------------------
    st.header("🚦 Planificador de Ruta Segura")
    st.markdown("Calcula la mejor ruta evitando zonas de alto riesgo de accidentes")
    
    # Inicializar estado
    if 'route_origin' not in st.session_state:
        st.session_state.route_origin = None
    if 'route_destination' not in st.session_state:
        st.session_state.route_destination = None
    if 'route_alternatives' not in st.session_state:
        st.session_state.route_alternatives = None
    if 'selected_route' not in st.session_state:
        st.session_state.selected_route = None
    
    # Lugares populares
    st.subheader("📍 Selección de Origen y Destino")
    
    col_places = st.columns(2)
    with col_places[0]:
        st.write("**🟢 Seleccionar Origen:**")
        place_cols = st.columns(3)
        for idx, (place, coords) in enumerate(list(POPULAR_PLACES.items())[:6]):
            col = place_cols[idx % 3]
            with col:
                if st.button(place, key=f"orig_{idx}", use_container_width=True):
                    st.session_state.route_origin = coords
                    st.success(f"✅ Origen: {place}")
                    st.session_state.route_alternatives = None
    
    with col_places[1]:
        st.write("**🔴 Seleccionar Destino:**")
        place_cols = st.columns(3)
        for idx, (place, coords) in enumerate(list(POPULAR_PLACES.items())[6:12]):
            col = place_cols[idx % 3]
            with col:
                if st.button(place, key=f"dest_{idx}", use_container_width=True):
                    st.session_state.route_destination = coords
                    st.success(f"✅ Destino: {place}")
                    st.session_state.route_alternatives = None
    
    # O usar el mapa
    st.info("💡 **Tip:** También puedes hacer clic en el mapa de arriba para seleccionar puntos personalizados")
    
    # Detectar clics en el mapa
    last_click = None
    if 'map_data' in locals() and isinstance(map_data, dict) and 'last_clicked' in map_data and map_data['last_clicked'] is not None:
        last_click = (map_data['last_clicked']['lat'], map_data['last_clicked']['lng'])
    
    if last_click:
        col_map_select = st.columns(3)
        with col_map_select[0]:
            st.write(f"**Punto seleccionado:** {last_click[0]:.4f}, {last_click[1]:.4f}")
        with col_map_select[1]:
            if st.button("Usar como Origen", key="map_orig"):
                st.session_state.route_origin = last_click
                st.session_state.route_alternatives = None
        with col_map_select[2]:
            if st.button("Usar como Destino", key="map_dest"):
                st.session_state.route_destination = last_click
                st.session_state.route_alternatives = None
    
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
    col_config = st.columns(3)
    
    with col_config[0]:
        vehicle_type = st.selectbox(
            "Tipo de vehículo",
            options=list(VEHICLE_SPEED_FACTORS.keys()),
            index=0,
            help="El tipo de vehículo afecta la velocidad y el riesgo"
        )
    
    with col_config[1]:
        departure_time = st.time_input(
            "Hora de salida",
            value=datetime.now().time(),
            help="La hora afecta el nivel de riesgo"
        )
    
    with col_config[2]:
        if st.button("🗑️ Limpiar Selección"):
            st.session_state.route_origin = None
            st.session_state.route_destination = None
            st.session_state.route_alternatives = None
            st.session_state.selected_route = None
            st.rerun()
    
    # Cargar grafo solo cuando sea necesario
    can_route = (st.session_state.route_origin is not None) and (st.session_state.route_destination is not None)
    
    if can_route:
        # Botón para calcular rutas
        if st.button("🔍 Buscar Rutas Alternativas", type="primary", use_container_width=True):
            with st.spinner("🗺️ Cargando red vial de Barcelona..."):
                G, g_msg = load_or_build_graph()
                if g_msg:
                    st.error(g_msg)
                elif G:
                    with st.spinner("🧮 Calculando rutas alternativas..."):
                        routes, error = calculate_alternative_routes(
                            G,
                            st.session_state.predictions_data,
                            model_data.get('cluster_geometries', {}),
                            st.session_state.route_origin,
                            st.session_state.route_destination,
                            vehicle_type
                        )
                        if error:
                            st.error(error)
                        elif routes:
                            st.session_state.route_alternatives = routes
                            st.session_state.selected_route = None
        
        # Mostrar alternativas si existen
        if st.session_state.route_alternatives:
            st.subheader("🔄 Comparación de Rutas")
            
            tabs = st.tabs(list(st.session_state.route_alternatives.keys()))
            
            for tab, (route_name, route_data) in zip(tabs, st.session_state.route_alternatives.items()):
                with tab:
                    # Tarjeta de resumen de ruta
                    avg_risk = route_data.get('avg_risk', 0.0)
                    safety_score = get_route_safety_score(route_data)
                    safety_badge = get_safety_badge(avg_risk)
                    
                    st.markdown(f"""
                    <div class='route-card'>
                        <h3>{route_name}</h3>
                        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 15px;'>
                            <div style='text-align: center;'>
                                <h4>📏 Distancia</h4>
                                <p style='font-size: 24px; margin: 0;'>{route_data['km']:.1f} km</p>
                            </div>
                            <div style='text-align: center;'>
                                <h4>⏱️ Tiempo</h4>
                                <p style='font-size: 24px; margin: 0;'>{route_data['min']:.0f} min</p>
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
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mapa de la ruta
                    route_map = create_enhanced_route_map(
                        st.session_state.predictions_data,
                        model_data.get('cluster_geometries', {}),
                        route_data,
                        st.session_state.route_origin,
                        st.session_state.route_destination,
                        show_gradient=True
                    )
                    st_folium(route_map, height=400, key=f"map_{route_name}")
                    
                    # Advertencias de la ruta
                    warnings = analyze_route_warnings(
                        route_data,
                        st.session_state.predictions_data,
                        model_data.get('cluster_geometries', {})
                    )
                    
                    if warnings:
                        with st.expander("⚠️ Avisos de Seguridad"):
                            for warning in warnings:
                                if warning['type'] == 'high':
                                    st.warning(f"**{warning['message']}**\n\n💡 {warning['suggestion']}")
                                else:
                                    st.info(f"{warning['message']}\n\n💡 {warning['suggestion']}")
                    
                    # Botón de selección
                    if st.button(f"✅ Seleccionar {route_name}", key=f"select_{route_name}", use_container_width=True):
                        st.session_state.selected_route = route_data
                        st.success(f"✅ Has seleccionado la ruta {route_name}")
            
            # Mostrar ruta seleccionada con más detalle
            if st.session_state.selected_route:
                st.markdown("---")
                st.subheader("📋 Ruta Seleccionada - Detalles")
                
                selected = st.session_state.selected_route
                col_details = st.columns(4)
                
                with col_details[0]:
                    st.metric("Distancia Total", f"{selected['km']:.2f} km")
                
                with col_details[1]:
                    st.metric("Tiempo Estimado", f"{selected['min']:.0f} min")
                
                with col_details[2]:
                    avg_speed = (selected['km'] / selected['min']) * 60 if selected['min'] > 0 else 0
                    st.metric("Velocidad Media", f"{avg_speed:.0f} km/h")
                
                with col_details[3]:
                    st.metric("Índice de Seguridad", f"{get_route_safety_score(selected):.0f}/100")
                
                # Información adicional basada en el horario
                if departure_time:
                    arrival_time = (datetime.combine(date.today(), departure_time) + 
                                  timedelta(minutes=int(selected['min']))).time()
                    st.info(f"🕐 **Salida:** {departure_time.strftime('%H:%M')} → **Llegada estimada:** {arrival_time.strftime('%H:%M')}")
    
    else:
        st.info("👆 Selecciona un origen y destino para calcular rutas seguras")
    
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