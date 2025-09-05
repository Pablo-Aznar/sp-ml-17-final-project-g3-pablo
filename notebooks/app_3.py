import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import numpy as np
import requests
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time
from scipy import spatial
import os
import hashlib

# Configuración básica
st.set_page_config(
    page_title="Barcelona Accident Predictor",
    page_icon="🚗",
    layout="wide"
)

# Inicializar estado de sesión para rutas
if 'route_data' not in st.session_state:
    st.session_state.route_data = {
        'ruta_directa': None,
        'riesgo_directo': None,
        'dist_directa': None,
        'dur_directa': None,
        'ruta_segura': None,
        'riesgo_seguro': None,
        'dist_segura': None,
        'dur_segura': None,
        'origen_coords': None,
        'destino_coords': None
    }

# Funciones
@st.cache_data
def load_predictions():
    try:
        csv_path = "predicciones_test.csv"
        if not os.path.exists(csv_path):
            st.error(f"Archivo no encontrado: {csv_path}. Asegúrate de que esté en el directorio de la app.")
            return pd.DataFrame()
        df = pd.read_csv(csv_path)
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

@st.cache_resource
def get_geolocator():
    return Nominatim(user_agent=f"barcelona_accident_predictor_{int(time.time())}", timeout=10)

@st.cache_data
def geocode_address(address, _geolocator):
    try:
        location = _geolocator.geocode(f"{address}, Barcelona, Spain")
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        st.warning(f"Error en geocodificación: {str(e)}")
        return None, None

def get_route_osrm(start_lat, start_lon, end_lat, end_lon, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            url = f"http://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
            response = requests.get(url, timeout=20)
            data = response.json()
            if data['code'] == 'Ok':
                route = data['routes'][0]
                coordinates = route['geometry']['coordinates']
                route_coords = [[coord[1], coord[0]] for coord in coordinates]
                distance = route['distance'] / 1000
                duration = route['duration'] / 60
                return route_coords, distance, duration
            else:
                st.warning(f"OSRM error: {data['code']}")
        except Exception as e:
            st.warning(f"Intento {attempt+1} fallido: {str(e)}")
        attempt += 1
        time.sleep(3)
    st.error("No se pudo obtener la ruta después de varios intentos.")
    return None, None, None

def calculate_route_risk(route_coords, df_risk, buffer_km=0.5):
    if not route_coords or df_risk.empty:
        return 0
    
    df_risk_sample = df_risk.sample(min(1000, len(df_risk))) if len(df_risk) > 1000 else df_risk
    risk_points = df_risk_sample[['lat', 'lon']].values
    risk_probas = df_risk_sample['y_proba'].values
    risk_tree = spatial.KDTree(risk_points)
    
    total_risk = 0
    risk_points_count = 0
    
    for route_point in route_coords[::10]:
        buffer_deg = buffer_km / 111
        indices = risk_tree.query_ball_point(route_point, r=buffer_deg)
        
        for idx in indices:
            actual_dist = geodesic(route_point, risk_points[idx]).kilometers
            if actual_dist <= buffer_km:
                weight = 1 - (actual_dist / buffer_km)
                total_risk += risk_probas[idx] * weight
                risk_points_count += 1
    
    return total_risk / max(risk_points_count, 1) if risk_points_count > 0 else 0

def find_safer_route(start_coords, end_coords, df_risk, num_waypoints=2, buffer_km=0.5):
    lat_min = min(start_coords[0], end_coords[0]) - 0.02
    lat_max = max(start_coords[0], end_coords[0]) + 0.02
    lon_min = min(start_coords[1], end_coords[1]) - 0.02
    lon_max = max(start_coords[1], end_coords[1]) + 0.02
    
    best_route = None
    best_risk = float('inf')
    best_distance = None
    best_duration = None
    
    progress_bar = st.progress(0)
    for i in range(num_waypoints):
        progress_bar.progress((i + 1) / num_waypoints)
        way_lat = np.random.uniform(lat_min, lat_max)
        way_lon = np.random.uniform(lon_min, lon_max)
        
        route1, dist1, dur1 = get_route_osrm(start_coords[0], start_coords[1], way_lat, way_lon)
        route2, dist2, dur2 = get_route_osrm(way_lat, way_lon, end_coords[0], end_coords[1])
        
        if route1 and route2:
            combined_route = route1 + route2[1:]
            combined_dist = (dist1 or 0) + (dist2 or 0)
            combined_dur = (dur1 or 0) + (dur2 or 0)
            
            route_risk = calculate_route_risk(combined_route, df_risk, buffer_km)
            distance_penalty = max(0, combined_dist - 20) * 0.01
            total_score = route_risk + distance_penalty
            
            if total_score < best_risk:
                best_risk = total_score
                best_route = combined_route
                best_distance = combined_dist
                best_duration = combined_dur
    
    progress_bar.empty()
    return best_route, best_distance, best_duration, best_risk

# Cargar datos
df_preds = load_predictions()

if df_preds.empty:
    st.error("❌ No se pudieron cargar los datos. Verifica que 'predicciones_test.csv' esté en el directorio.")
    st.stop()

# HEADER
st.title("🚗 Predictor de Accidentes Barcelona")
st.markdown("**Visualización interactiva del riesgo de accidentes viales y rutas seguras**")

# Barra lateral con controles
with st.sidebar:
    st.header("🎛️ Controles - Predicciones de Riesgo")
    fecha_min = df_preds["Fecha"].dt.date.min()
    fecha_max = df_preds["Fecha"].dt.date.max()
    fecha = st.date_input(
        "📅 Fecha:",
        value=fecha_min,
        min_value=fecha_min,
        max_value=fecha_max,
        key="fecha_pred"
    )
    hora = st.slider("🕐 Hora:", 0, 23, 12, key="hora_pred")
    st.write(f"**Hora seleccionada:** {hora:02d}:00")
    
    risk_filter = st.selectbox(
        "Nivel de riesgo:",
        ["Todos", "Solo alto riesgo (>70%)", "Solo medio-alto (>40%)", "Solo bajo (<40%)"],
        key="risk_filter"
    )
    map_style = st.radio(
        "Estilo de mapa:",
        ["Mapa de calor", "Puntos", "Ambos"],
        key="map_style"
    )
    
    st.markdown("---")
    st.header("🎛️ Controles - Rutas Seguras")
    fecha_ruta = st.date_input(
        "📅 Fecha para ruta:",
        value=fecha_min,
        min_value=fecha_min,
        max_value=fecha_max,
        key="fecha_ruta"
    )
    hora_ruta = st.slider("🕐 Hora para ruta:", 0, 23, 12, key="hora_ruta")
    st.write(f"**Hora para análisis:** {hora_ruta:02d}:00")
    
    input_method = st.radio(
        "Método de selección:",
        ["📝 Direcciones", "📍 Coordenadas"],
        key="input_method"
    )
    if input_method == "📝 Direcciones":
        origen_addr = st.text_input("🏠 Dirección de origen:", placeholder="Ej: Plaza Cataluña, Barcelona", key="origen_addr")
        destino_addr = st.text_input("🎯 Dirección de destino:", placeholder="Ej: Sagrada Familia, Barcelona", key="destino_addr")
    else:
        col1, col2 = st.columns(2)
        with col1:
            origen_lat = st.number_input("Origen Lat:", value=41.3851, format="%.6f", key="origen_lat")
            origen_lon = st.number_input("Origen Lon:", value=2.1734, format="%.6f", key="origen_lon")
        with col2:
            destino_lat = st.number_input("Destino Lat:", value=41.4036, format="%.6f", key="destino_lat")
            destino_lon = st.number_input("Destino Lon:", value=2.1744, format="%.6f", key="destino_lon")
    
    buffer_riesgo = st.slider("Buffer de riesgo (km):", 0.1, 2.0, 0.5, 0.1, key="buffer_riesgo")
    mostrar_alternativas = st.checkbox("Buscar rutas alternativas", value=True, key="mostrar_alt")
    num_alternativas = st.slider("Número de alternativas:", 1, 5, 2, key="num_alt") if mostrar_alternativas else 0
    calcular_ruta = st.button("🚀 Calcular Ruta Segura", type="primary", key="calcular_ruta")

# Filtrar datos para predicciones de riesgo
df_dia = df_preds[df_preds["Fecha"].dt.date == fecha]
df_filtrado = df_dia[df_dia["hora"] == hora].copy()

if risk_filter == "Solo alto riesgo (>70%)":
    df_filtrado = df_filtrado[df_filtrado['y_proba'] >= 0.7]
elif risk_filter == "Solo medio-alto (>40%)":
    df_filtrado = df_filtrado[df_filtrado['y_proba'] >= 0.4]
elif risk_filter == "Solo bajo (<40%)":
    df_filtrado = df_filtrado[df_filtrado['y_proba'] < 0.4]

# Filtrar datos para rutas
df_ruta_riesgo = df_preds[(df_preds["Fecha"].dt.date == fecha_ruta) & (df_preds["hora"] == hora_ruta)].copy()
df_ruta_riesgo = df_ruta_riesgo[df_ruta_riesgo['y_proba'] >= 0.4]

# Layout principal
col_mapa, col_info = st.columns([2, 1])

# Mapa de Riesgo
with col_mapa:
    st.subheader("🗺️ Mapa de Riesgo de Accidentes")
    map_risk_key = hashlib.md5(f"mapa_riesgo_{fecha}_{hora}_{map_style}_{risk_filter}".encode()).hexdigest()
    risk_map_placeholder = st.empty()
    
    m_risk = folium.Map(location=[41.3851, 2.1734], zoom_start=12, tiles='OpenStreetMap')
    
    if not df_filtrado.empty:
        if map_style in ["Mapa de calor", "Ambos"]:
            heat_data = df_filtrado[["lat", "lon", "y_proba"]].values.tolist()
            plugins.HeatMap(
                heat_data,
                radius=20,
                blur=15,
                min_opacity=0.3,
                gradient={0.0: '#28a745', 0.4: '#ffc107', 0.7: '#fd7e14', 1.0: '#dc3545'}
            ).add_to(m_risk)
        if map_style in ["Puntos", "Ambos"]:
            sample_points = df_filtrado.sample(min(100, len(df_filtrado)))
            for _, row in sample_points.iterrows():
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
                ).add_to(m_risk)
    
    with risk_map_placeholder:
        st_folium(m_risk, width=700, height=500, key=map_risk_key)

# Información de riesgo
with col_info:
    st.subheader("📊 Información de Riesgo")
    if not df_filtrado.empty:
        total_points = len(df_filtrado)
        avg_risk = df_filtrado['y_proba'].mean()
        high_risk_points = len(df_filtrado[df_filtrado['y_proba'] >= 0.7])
        max_risk = df_filtrado['y_proba'].max()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📍 Puntos", f"{total_points:,}")
            st.metric("🚨 Alto Riesgo", f"{high_risk_points:,}")
        with col2:
            st.metric("📊 Riesgo Promedio", f"{avg_risk:.1%}")
            st.metric("⚠️ Riesgo Máximo", f"{max_risk:.1%}")
        
        st.info(f"""
        **Análisis para {fecha} - {hora:02d}:00**
        📍 **Total de puntos:** {total_points:,}
        📊 **Riesgo promedio:** {avg_risk:.1%}
        🚨 **Puntos alto riesgo:** {high_risk_points:,}
        ⚠️ **Riesgo máximo:** {max_risk:.1%}
        """)
        
        st.subheader("🎯 Top 5 Zonas Peligrosas")
        top_risk = df_filtrado.nlargest(5, 'y_proba')
        for i, (_, row) in enumerate(top_risk.iterrows(), 1):
            color, level = get_risk_color(row['y_proba'])
            st.write(f"**{i}.** Riesgo: **{row['y_proba']:.1%}** {level}")
            st.write(f"   📍 {row['lat']:.4f}, {row['lon']:.4f}")
        
        st.subheader("📈 Distribución de Riesgo")
        fig = px.histogram(df_filtrado, x='y_proba', nbins=15, title="Distribución del Riesgo")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ No hay datos para {fecha} a las {hora:02d}:00")

# Sección de Rutas Seguras (debajo del mapa de riesgo)
st.markdown("---")
st.header("🛣️ Calculadora de Rutas Seguras")
col_mapa_ruta, col_info_ruta = st.columns([2, 1])

# Cálculo de rutas
if calcular_ruta:
    geolocator = get_geolocator()
    origen_coords = None
    destino_coords = None
    
    if input_method == "📝 Direcciones":
        if origen_addr and destino_addr:
            with st.spinner("🔍 Geocodificando direcciones..."):
                origen_lat, origen_lon = geocode_address(origen_addr, geolocator)
                destino_lat, destino_lon = geocode_address(destino_addr, geolocator)
                if origen_lat is not None and destino_lat is not None:
                    origen_coords = (origen_lat, origen_lon)
                    destino_coords = (destino_lat, destino_lon)
                else:
                    st.error("❌ No se pudieron geocodificar las direcciones. Usa nombres completos (ej: 'Plaza Cataluña, Barcelona').")
        else:
            st.warning("⚠️ Introduce origen y destino.")
    else:
        origen_coords = (origen_lat, origen_lon)
        destino_coords = (destino_lat, destino_lon)
    
    if origen_coords and destino_coords:
        with st.spinner("🚗 Calculando ruta directa..."):
            ruta_directa, dist_directa, dur_directa = get_route_osrm(
                origen_coords[0], origen_coords[1],
                destino_coords[0], destino_coords[1]
            )
            if ruta_directa:
                riesgo_directo = calculate_route_risk(ruta_directa, df_ruta_riesgo, buffer_riesgo)
                st.session_state.route_data.update({
                    'ruta_directa': ruta_directa,
                    'riesgo_directo': riesgo_directo,
                    'dist_directa': dist_directa,
                    'dur_directa': dur_directa,
                    'origen_coords': origen_coords,
                    'destino_coords': destino_coords
                })
        
        if mostrar_alternativas and num_alternativas > 0:
            with st.spinner("🔍 Buscando rutas alternativas..."):
                ruta_segura, dist_segura, dur_segura, riesgo_seguro = find_safer_route(
                    origen_coords, destino_coords, df_ruta_riesgo, num_waypoints=num_alternativas, buffer_km=buffer_riesgo
                )
                if ruta_segura:
                    st.session_state.route_data.update({
                        'ruta_segura': ruta_segura,
                        'riesgo_seguro': riesgo_seguro,
                        'dist_segura': dist_segura,
                        'dur_segura': dur_segura
                    })

# Mapa de Rutas (separado, con placeholder)
with col_mapa_ruta:
    route_map_placeholder = st.empty()
    map_route_key = hashlib.md5(f"mapa_rutas_{fecha_ruta}_{hora_ruta}_{st.session_state.route_data.get('origen_coords')}_{st.session_state.route_data.get('destino_coords')}_{calcular_ruta}".encode()).hexdigest()
    
    if st.session_state.route_data['origen_coords'] and st.session_state.route_data['destino_coords']:
        center_lat = (st.session_state.route_data['origen_coords'][0] + st.session_state.route_data['destino_coords'][0]) / 2
        center_lon = (st.session_state.route_data['origen_coords'][1] + st.session_state.route_data['destino_coords'][1]) / 2
        m_rutas = folium.Map(location=[center_lat, center_lon], zoom_start=13)
        
        # Añadir puntos de riesgo como contexto
        if not df_ruta_riesgo.empty:
            sample_points = df_ruta_riesgo.sample(min(100, len(df_ruta_riesgo)))
            for _, row in sample_points.iterrows():
                color, _ = get_risk_color(row['y_proba'])
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=4,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.4,
                    popup=f"Riesgo: {row['y_proba']:.1%}"
                ).add_to(m_rutas)
        
        # Añadir rutas
        if st.session_state.route_data['ruta_directa']:
            folium.PolyLine(
                locations=st.session_state.route_data['ruta_directa'],
                weight=5,
                color='red',
                opacity=0.8,
                popup=f"Ruta Directa<br>Riesgo: {st.session_state.route_data['riesgo_directo']:.3f}<br>Distancia: {st.session_state.route_data['dist_directa']:.1f} km<br>Tiempo: {st.session_state.route_data['dur_directa']:.0f} min"
            ).add_to(m_rutas)
        
        if st.session_state.route_data['ruta_segura']:
            folium.PolyLine(
                locations=st.session_state.route_data['ruta_segura'],
                weight=5,
                color='green',
                opacity=0.8,
                popup=f"Ruta Segura<br>Riesgo: {st.session_state.route_data['riesgo_seguro']:.3f}<br>Distancia: {st.session_state.route_data['dist_segura']:.1f} km<br>Tiempo: {st.session_state.route_data['dur_segura']:.0f} min"
            ).add_to(m_rutas)
        
        if st.session_state.route_data['origen_coords']:
            folium.Marker(
                location=st.session_state.route_data['origen_coords'],
                popup="🏠 Origen",
                icon=folium.Icon(color='blue', icon='home')
            ).add_to(m_rutas)
        
        if st.session_state.route_data['destino_coords']:
            folium.Marker(
                location=st.session_state.route_data['destino_coords'],
                popup="🎯 Destino",
                icon=folium.Icon(color='red', icon='flag')
            ).add_to(m_rutas)
        
        leyenda_html = '''
        <div style="position: fixed; 
                   bottom: 50px; left: 50px; width: 200px; height: auto; 
                   background-color: white; border:2px solid grey; z-index:9999; 
                   font-size:14px; padding: 10px">
        <p><b>Leyenda:</b></p>
        <p><span style="color:red;">━━━</span> Ruta Directa</p>
        <p><span style="color:green;">━━━</span> Ruta Segura</p>
        <p><span style="color:#dc3545;">●</span> Alto Riesgo</p>
        </div>
        '''
        m_rutas.get_root().html.add_child(folium.Element(leyenda_html))
        
        with route_map_placeholder:
            st_folium(m_rutas, width=700, height=500, key=map_route_key)
    else:
        with route_map_placeholder:
            st.info("👈 Configura y calcula la ruta segura en la barra lateral.")

# Información de rutas
with col_info_ruta:
    st.subheader("📊 Comparación de Rutas")
    if st.session_state.route_data['ruta_directa']:
        st.markdown("### 🔴 Ruta Directa")
        st.info(f"""
        📍 **Distancia:** {st.session_state.route_data['dist_directa']:.1f} km
        ⏱️ **Tiempo:** {st.session_state.route_data['dur_directa']:.0f} min
        ⚠️ **Riesgo:** {st.session_state.route_data['riesgo_directo']:.3f}
        """)
        
        if st.session_state.route_data['ruta_segura']:
            st.markdown("### 🟢 Ruta Segura")
            diff_dist = st.session_state.route_data['dist_segura'] - st.session_state.route_data['dist_directa']
            diff_dur = st.session_state.route_data['dur_segura'] - st.session_state.route_data['dur_directa']
            diff_risk = st.session_state.route_data['riesgo_seguro'] - st.session_state.route_data['riesgo_directo']
            st.success(f"""
            📍 **Distancia:** {st.session_state.route_data['dist_segura']:.1f} km
            ⏱️ **Tiempo:** {st.session_state.route_data['dur_segura']:.0f} min
            ⚠️ **Riesgo:** {st.session_state.route_data['riesgo_seguro']:.3f}
            """)
            
            st.markdown("### 📈 Diferencias")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Distancia", f"{st.session_state.route_data['dist_segura']:.1f} km", f"{diff_dist:+.1f} km")
            with col2:
                st.metric("Tiempo", f"{st.session_state.route_data['dur_segura']:.0f} min", f"{diff_dur:+.0f} min")
            with col3:
                st.metric("Riesgo", f"{st.session_state.route_data['riesgo_seguro']:.3f}", f"{diff_risk:+.3f}")
            
            if diff_risk < -0.01:
                mejora_pct = abs(diff_risk / st.session_state.route_data['riesgo_directo']) * 100 if st.session_state.route_data['riesgo_directo'] > 0 else 0
                st.success(f"🎯 **Mejora en seguridad:** {mejora_pct:.1f}%")
                if diff_dist < 5 and diff_dur < 15:
                    st.success("✅ **Se recomienda la ruta segura**")
                else:
                    st.warning("⚖️ **Evalúa el balance tiempo/seguridad**")
            else:
                st.info("ℹ️ **La ruta directa es aceptable**")
        
        st.markdown("---")
        st.markdown("### ℹ️ Información de Rutas")
        st.markdown(f"""
        **Parámetros:**
        - 📅 Fecha: {fecha_ruta}
        - 🕐 Hora: {hora_ruta:02d}:00
        - 📏 Buffer: {buffer_riesgo} km
        - 🎯 Puntos analizados: {len(df_ruta_riesgo):,}
        """)
    else:
        st.info("👈 Configura y calcula la ruta para ver la comparación.")
        st.markdown("### 🚀 ¿Cómo funciona?")
        st.markdown("""
        1. **Configura** origen y destino en la barra lateral
        2. **Ajusta** fecha, hora y buffer de riesgo
        3. **Calcula** la ruta segura
        **Mejoras:**
        - Mapa de rutas persistente con `st.session_state`
        - Placeholder dedicado para evitar desaparición
        - Cálculos optimizados
        """)
        st.markdown("### 🎯 Ventajas")
        st.markdown("""
        - Mapas separados para riesgo y rutas
        - Sin titileo ni desaparición
        - Ideal para despliegue
        """)

# Análisis Temporal
if not df_dia.empty:
    st.markdown("---")
    st.subheader("📈 Análisis Temporal")
    col1, col2 = st.columns(2)
    
    with col1:
        hourly_risk = df_dia.groupby('hora')['y_proba'].mean().reset_index()
        fig = px.line(hourly_risk, x='hora', y='y_proba', 
                      title=f"Riesgo Promedio por Hora - {fecha}")
        fig.add_vline(x=hora, line_dash="dot", annotation_text=f"Actual: {hora:02d}:00")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        high_risk_hourly = df_dia[df_dia['y_proba'] >= 0.7].groupby('hora').size().reset_index(name='count')
        fig = px.bar(high_risk_hourly, x='hora', y='count', 
                     title=f"Puntos Alto Riesgo por Hora - {fecha}")
        st.plotly_chart(fig, use_container_width=True)

# FOOTER
st.markdown("---")
st.markdown("🚗 **Barcelona Accident Risk Predictor** - Versión con mapas separados y persistencia")
st.markdown("Para despliegue, verifica la conectividad a OSRM/Nominatim. Usa APIs pagadas para producción si es necesario.")