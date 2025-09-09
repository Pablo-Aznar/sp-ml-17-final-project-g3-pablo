# 🚗 Sistema de Predicción de Accidentes de Tráfico - Barcelona

## 📱 Aplicación Web Desplegada
**🌐 URL:** [https://accident-predictor-app-pablo-aznar.streamlit.app/](https://accident-predictor-app-pablo-aznar.streamlit.app/)


## 📊 Descripción General


## ✨ Características Principales

### 🔮 Predicción en Tiempo Real
- **Análisis por zonas**: Predicción de riesgo para 101 clusters geográficos de Barcelona
- **Factores múltiples**: Considera hora, día de la semana, festividades, temperatura, precipitación y viento
- **Actualización dinámica**: Predicciones que se actualizan según la fecha y hora seleccionada
- **Zonificación inteligente**: Mapeo visual de zonas de alto, medio y bajo riesgo

### 🗺️ Planificador de Rutas Seguras
- **Geometría real**: Rutas que siguen exactamente el trazado de las carreteras
- **Múltiples algoritmos**: Ruta más segura, más rápida y equilibrada
- **Condiciones de tráfico**: Integración de factores de tráfico por horario
- **Tipos de vehículo**: Optimización específica para coches, motos, bicicletas y camiones

### 📈 Análisis Histórico Completo
- **8 años de datos**: Análisis de 67,424 accidentes (2017-2024)
- **Patrones temporales**: Distribución por hora, día, mes y año
- **Mapa de calor**: Visualización hora vs día de la semana
- **Tendencias**: Análisis de evolución temporal y estacionalidad

### 🎯 Características Técnicas Avanzadas
- **OSMnx Integration**: Red vial detallada de Barcelona sin simplificación
- **Shapely Geometry**: Interpolación de coordenadas para curvas suaves
- **Real-time Weather**: Datos meteorológicos de Open-Meteo API
- **Geocoding**: Búsqueda inteligente de direcciones específicas de Barcelona

## 🛠️ Tecnologías Utilizadas

### Frontend y Visualización
- **Streamlit**: Framework de aplicación web
- **Folium**: Mapas interactivos con Leaflet
- **Plotly**: Gráficos interactivos y dashboards
- **CSS3**: Diseño profesional y responsivo

### Machine Learning y Datos
- **Scikit-learn**: Modelos lightgbm de predicción y preprocesamiento
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Computación numérica optimizada
- **Joblib**: Serialización de modelos entrenados

### Procesamiento Geoespacial
- **OSMnx**: Descarga y análisis de redes de OpenStreetMap
- **NetworkX**: Algoritmos de grafos para routing
- **Shapely**: Operaciones geométricas avanzadas
- **Geopy**: Geocodificación y búsqueda de direcciones

### APIs y Servicios Externos
- **Open-Meteo API**: Datos meteorológicos en tiempo real
- **OpenStreetMap**: Datos de red vial y geografía
- **Nominatim**: Servicio de geocodificación






## 🔧 Procesos Principales

### 1. Sistema de Predicción
```python
# Modelo de Machine Learning con múltiples factores
- Features meteorológicas: temperatura, precipitación, viento
- Features temporales: hora, día semana, festivos
- Features cíclicas: transformaciones sin/cos para temporalidad
- Features de cluster: histórico de accidentes por zona
- Interacciones: hora punta + lluvia, fin de semana + noche
```

### 2. Procesamiento de Red Vial
```python
# Optimizaciones implementadas:
- OSMnx sin simplificación (simplify=False)
- Filtrado específico para vehículos motorizados
- Extracción de geometría real de cada segmento
- Interpolación de puntos en curvas largas
- Respeto de direcciones de circulación
```

### 3. Algoritmo de Routing
```python
# Cálculo de rutas alternativas:
- Más Segura: prioriza evitar zonas de alto riesgo
- Más Rápida: optimiza tiempo considerando tráfico
- Equilibrada: balance entre tiempo y seguridad
```

### 4. Integración de Tráfico
```python
# Factores de tráfico por horario:
- Madrugada (0-5h): Factor 0.5-0.6
- Hora punta mañana (7-9h): Factor 1.3-1.5  
- Mediodía (12-15h): Factor 1.0-1.3
- Hora punta tarde (17-19h): Factor 1.3-1.5
- Noche (21-23h): Factor 0.6-0.9
```



### Interfaz Profesional
- **Paleta de colores**: Azul corporativo con acentos verdes/naranjas
- **Componentes**: Cards con sombras sutiles y efectos hover
- **Responsive**: Adaptable a diferentes tamaños de pantalla
- **Accesibilidad**: Contraste optimizado y navegación clara

### Experiencia de Usuario
- **Flujo intuitivo**: Selección de parámetros → Predicción → Routing
- **Feedback visual**: Indicadores de carga y confirmaciones
- **Múltiples métodos**: Lugares populares, búsqueda, clic en mapa
- **Persistencia**: Rutas favoritas guardadas localmente

## 🏗️ Optimizaciones Implementadas

### Rendimiento
- **Caching inteligente**: `@st.cache_data` para operaciones costosas
- **Grafo persistente**: Descarga única de red vial con almacenamiento local
- **Predicción vectorizada**: Cálculo en lote para múltiples clusters
- **Geometría optimizada**: Balance entre precisión y velocidad

### Precisión de Rutas
- **Sin simplificación**: Mantiene todos los nodos intermedios de OSM
- **Geometría real**: Extracción de coordenadas reales de carreteras
- **Interpolación**: Puntos adicionales en curvas para visualización suave
- **Filtrado de carreteras**: Solo vías transitables por vehículos

### Factores Urbanos Realistas
- **Velocidades ajustadas**: Reducción por condiciones urbanas de Barcelona
- **Demoras contextuales**: +120% tiempo base por semáforos y cruces
- **Tipos de vía**: Diferenciación entre autopistas, avenidas y calles
- **Tráfico horario**: Multiplicadores específicos por franja temporal

## 📱 Funcionalidades de la Web App

### Panel de Control
- **Selector de fecha/hora**: Predicción hasta 30 días
- **Zona horaria local**: Automática para Barcelona (CET/CEST)
- **Meteorología en tiempo real**: Integración con Open-Meteo(API)
- **Estadísticas históricas**: Métricas clave y tendencias

### Mapas Interactivos
- **Zonas de riesgo**: Círculos coloreados por probabilidad
- **Rutas detalladas**: Trazado que sigue carreteras reales
- **Marcadores informativos**: Origen, destino y puntos de interés
- **Gradiente de riesgo**: Colores de ruta según peligrosidad

### Planificador de Rutas
- **Múltiples métodos**: Lugares populares, búsqueda, selección manual
- **Configuración avanzada**: Tipo de vehículo y condiciones de tráfico
- **Comparación de alternativas**: 3 rutas con métricas detalladas


### Roadmap Futuro
- [ ] Integración con APIs de tráfico en tiempo real (TomTom/Google)
- [ ] Expansión a otras ciudades españolas
- [ ] Modelo de deep learning con datos temporales
- [ ] App móvil nativa con notificaciones
- [ ] Integración con sistemas de navegación


- **Aplicación Web**: [https://accident-predictor-app-pablo-aznar.streamlit.app/](https://accident-predictor-app-pablo-aznar.streamlit.app/)


