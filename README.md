# 🚗 Sistema de Predicción de Accidentes de Tráfico - Barcelona

## 📱 Aplicación Web Desplegada
**🌐 URL:** [https://accident-predictor-app-pablo-aznar.streamlit.app/](https://accident-predictor-app-pablo-aznar.streamlit.app/)

## 📊 Descripción General

Sistema inteligente de predicción de accidentes de tráfico para la ciudad de Barcelona que utiliza modelos de Machine Learning para analizar el riesgo de accidentes en tiempo real. La aplicación combina datos históricos de accidentes (2017-2024), condiciones meteorológicas en tiempo real, y análisis de tráfico para proporcionar predicciones precisas y recomendaciones de rutas seguras.

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

## 📁 Estructura del Proyecto

```
├── app.py                          # Aplicación principal Streamlit
├── models/
│   └── barcelona_accident_model_enhanced.joblib    # Modelo ML entrenado
├── data/
│   └── graph/
│       ├── barcelona_drive_detailed.graphml        # Red vial detallada
│       └── barcelona_geometry.json                 # Geometría de carreteras
├── webapp/
│   └── app_final.py
├── requirements.txt                # Dependencias Python
├── favorite_routes.json           # Rutas favoritas (generado automáticamente)
└── README.md                      # Documentación
```

## 🚀 Instalación y Uso Local

### Prerequisitos
- Python 3.8+
- pip (gestor de paquetes)

### Instalación
```bash
# Clonar el repositorio
git clone <repository-url>
cd accident-predictor-barcelona

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
streamlit run app.py
```

### Dependencias Principales
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
folium>=0.14.0
streamlit-folium>=0.15.0
plotly>=5.17.0
requests>=2.31.0
osmnx>=1.6.0
networkx>=3.1
scipy>=1.11.0
geopy>=2.4.0
pytz
shapely>=2.0.0
lightgbm>=4.0.0
```

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

## 📊 Datasets y Modelo

### Datos de Entrenamiento
- **Fuente**: Accidentes de tráfico Barcelona (Open Data BCN)
- **Período**: 2017-2024
- **Volumen**: 67,424 registros de accidentes
- **Clustering**: 101 zonas geográficas optimizadas

### Características del Modelo
- **Algoritmo**: lightgbm
- **Métricas**: Precisión, Recall, F1-Score optimizados
- **Validación**: Cross-validation temporal
- **Threshold**: Optimizado para maximizar Recall e intentar minimizar falsos negativos

### Variables Predictoras
1. **Temporales**: hora, día_semana, mes, año, festivo
2. **Meteorológicas**: temperatura, precipitación, velocidad_viento
3. **Geográficas**: cluster_id, densidad_histórica
4. **Contextuales**: fin_semana, hora_punta, condiciones_especiales

## 🎨 Diseño y UX

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
- **Meteorología en tiempo real**: Integración con Open-Meteo
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
- **Rutas favoritas**: Guardar y cargar rutas frecuentes

## 🔒 Consideraciones de Seguridad

### Datos y Privacidad
- **Sin almacenamiento personal**: No se guardan datos del usuario
- **Rutas locales**: Favoritas almacenadas solo en el navegador
- **APIs públicas**: Solo uso de servicios de datos abiertos
- **Geocodificación anónima**: Sin tracking de ubicaciones

### Limitaciones y Disclaimers
- **Uso informativo**: Las predicciones son orientativas, no definitivas
- **Responsabilidad del conductor**: El usuario debe respetar señalización
- **Datos en tiempo real**: Sujetos a disponibilidad de APIs externas
- **Cobertura geográfica**: Específico para Barcelona ciudad

## 👥 Contribución y Desarrollo

### Estructura de Contribución
1. **Fork** del repositorio
2. **Branch** para nueva característica
3. **Desarrollo** con tests locales
4. **Pull Request** con descripción detallada

### Roadmap Futuro
- [ ] Integración con APIs de tráfico en tiempo real (TomTom/Google)
- [ ] Expansión a otras ciudades españolas
- [ ] Modelo de deep learning con datos temporales
- [ ] App móvil nativa con notificaciones
- [ ] Integración con sistemas de navegación

## 📞 Contacto y Soporte

- **Aplicación Web**: [https://accident-predictor-app-pablo-aznar.streamlit.app/](https://accident-predictor-app-pablo-aznar.streamlit.app/)
- **Autor**: Pablo Aznar
- **Tecnología**: Streamlit Cloud Deployment

---

### 📄 Licencia
Este proyecto está desarrollado para fines educativos y de investigación. Los datos utilizados provienen de fuentes públicas (Open Data BCN, OpenStreetMap, Open-Meteo).

### 🙏 Agradecimientos
- **Ajuntament de Barcelona**: Por los datos abiertos de accidentes
- **OpenStreetMap**: Por la cartografía colaborativa
- **Open-Meteo**: Por los datos meteorológicos gratuitos
- **Streamlit**: Por la plataforma de deployment

---

*Aplicación desarrollada con el objetivo de mejorar la seguridad vial en Barcelona mediante tecnología predictiva avanzada.*