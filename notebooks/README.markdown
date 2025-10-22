# Análisis Geoespacial de Accidentes en Barcelona (2017-2024)

Este proyecto realiza un análisis geoespacial de los accidentes registrados en Barcelona entre 2017 y 2024, con el objetivo de preparar un dataset para un modelo predictivo que identifique localizaciones de accidentes. El proceso incluye la clusterización de accidentes, la generación de no-accidentes en las horas con accidentes, y el rellenado de horas faltantes con no-accidentes, todo ello utilizando datos geoespaciales y técnicas de clustering.

## Descripción del Dataset
El dataset original (`df_acc`) contiene 67,424 registros de accidentes en Barcelona desde 2017 hasta 2024, con las siguientes columnas principales:
- `Numero_expedient`: Identificador único del accidente.
- `time_n`: Fecha y hora del accidente (resolución horaria).
- `Longitud` y `Latitud`: Coordenadas geográficas del accidente.
- `Descripcio_causa_vianant`, `Descripcio_causa_conductor`, `Descripcio_causa_mediata`, `Descripcio_tipus_accident`: Detalles sobre las causas y tipo de accidente.
- `location_id`: Identificador de ubicación (opcional, puede ser un alias de `cluster_id`).

El objetivo es generar un dataset completo que cubra todas las horas del período (70,080 horas) y todos los clusters, incluyendo tanto accidentes como no-accidentes, para entrenar un modelo predictivo.

## Requisitos
- **Librerías Python**:
  - `pandas`, `numpy`: Manipulación de datos.
  - `geopandas`: Manejo de datos geoespaciales.
  - `osmnx`: Descarga de datos de OpenStreetMap (red viaria y límites de Barcelona).
  - `shapely`: Operaciones geométricas.
  - `folium`: Visualización de mapas interactivos.
  - `scikit-learn`: Clustering con KMeans.
  - `scipy.spatial`: Generación de diagramas de Voronoi.

- **Entorno**: Codespaces con Python 3.12 y las librerías instaladas.
- **Datos**: Archivo `df_merged.csv` con los accidentes, ubicado en `../data/processed/`.

## Proceso

### 1. Preparación del GeoDataFrame de Accidentes
- **Limpieza**: Se eliminan registros con valores nulos en `Latitud` o `Longitud`.
- **Conversión a GeoDataFrame**: Se crea `gdf_acc` usando `geopandas`, con geometrías de puntos generadas a partir de `Longitud` y `Latitud` en el sistema de coordenadas EPSG:4326 (WGS84).
- **Filtrado geográfico**: Se descargan los límites de Barcelona desde OpenStreetMap (`osmnx`) y se filtran los accidentes dentro de estos límites, resultando en 67,209 accidentes válidos (de un total de 67,424).

### 2. Clusterización con KMeans
- **Clustering**: Se aplica el algoritmo KMeans con 100 clusters (`N_CLUSTERS=100`) a las coordenadas de los accidentes (`gdf_acc_in`) para agruparlos en regiones espaciales.
- **Centros de clusters**: Se generan los centros de los clusters (`gdf_centers`) como puntos en EPSG:4326, verificando que todos estén dentro de los límites de Barcelona (100/100 válidos).
- **Polígonos de Voronoi**: Se crean polígonos de Voronoi a partir de los centros de los clusters, recortados al límite de Barcelona, resultando en 100 polígonos válidos (`gdf_clusters`).
- **Asignación de `cluster_id`**: Cada accidente en `gdf_acc_in` se asigna a un `cluster_id` según el cluster más cercano, y esta columna se propaga a `gdf_acc`.

### 3. Generación de No-Accidentes en Horas con Accidentes
- **Red viaria**: Se descarga la red viaria de Barcelona (`drive`) usando `osmnx` y se proyecta a EPSG:25831 (UTM, metros) para cálculos espaciales.
- **Candidatos**: Se generan puntos candidatos cada 40 metros a lo largo de los tramos de carretera dentro de los polígonos de Voronoi, resultando en 123,884 candidatos (`gdf_candidates`).
- **No-accidentes**:
  - Para cada combinación de hora (`time_n`) y cluster con al menos un accidente, se generan hasta 2 no-accidentes (`N_NEG=2`).
  - Los no-accidentes se seleccionan de los candidatos, asegurando que estén a más de 15 metros (`BUF_M=15`) de cualquier accidente real en la misma hora y cluster.
  - Se crea `gdf_neg` con columnas `geometry`, `time_n`, `accident` (0), `cluster_id`, `tramo`, y las columnas adicionales (`Numero_expedient`, etc.) como `np.nan`.
- **Unión**: Se combinan accidentes (`gdf_acc`) y no-accidentes (`gdf_neg`) en `dataset`, con 37,179 horas cubiertas.

### 4. Rellenado de Horas Faltantes
- **Horas totales**: Se define el rango completo de horas desde 01/01/2017 00:00 hasta 31/12/2024 23:00 (70,080 horas).
- **Horas faltantes**: Se identifican 32,901 horas sin accidentes ni no-accidentes en `dataset`.
- **No-accidentes adicionales**:
  - Para cada hora faltante y cada cluster (100 clusters), se selecciona un candidato aleatorio de `gdf_candidates`.
  - Se genera un no-accidente con `accident=0`, `geometry`, `tramo`, y las columnas adicionales como `np.nan`.
- **Dataset final**: Se concatena `dataset` con los nuevos no-accidentes (`df_missing`), resultando en `dataset_final` con ~3.36M filas (37,179 + 32,901 × 100).

### 5. Visualización
- **Mapa Folium**:
  - Se crea un mapa interactivo centrado en Barcelona (lat: 41.3879, lon: 2.16992).
  - Se muestran los polígonos de Voronoi (azul claro, opacidad 0.2).
  - Se visualiza una muestra de hasta 30,000 puntos de `dataset_final`: accidentes en rojo, no-accidentes en azul.
  - Cada punto incluye un tooltip con `Numero_expedient` (o "No-accident"), `time_n`, y `cluster_id`.
  - El mapa se guarda como `full_accidents_non_accidents_clusters_map.html`.

### 6. Salida
- **Dataset final**: Guardado como `full_accidents_non_accidents.geojson`, con columnas:
  - `time_n`: Fecha y hora (datetime).
  - `accident`: 1 para accidentes, 0 para no-accidentes.
  - `geometry`: Punto geográfico (EPSG:4326).
  - `cluster_id`: Identificador del cluster (0-99).
  - `tramo`: ID del tramo de carretera (OSM `osmid`).
  - `Numero_expedient`, `Descripcio_causa_vianant`, `Descripcio_causa_conductor`, `Descripcio_causa_mediata`, `Descripcio_tipus_accident`, `location_id`: Datos del accidente o `np.nan` para no-accidentes.
- **Mapa**: Guardado como archivo HTML para inspección visual.

## Instrucciones de Uso
1. **Configuración**:
   - Instala las librerías: `pip install pandas numpy geopandas osmnx folium scikit-learn scipy`.
   - Asegúrate de que `df_merged.csv` esté en `../data/processed/`.
   - Resuelve el `DtypeWarning` especificando tipos de datos al cargar el CSV:
     ```python
     dtypes = {
         'Numero_expedient': str,
         'Codi_districte': str,
         'Codi_barri': str,
         'Codi_carrer': str,
         'Num_postal': str,
         'Descripcio_causa_vianant': str,
         'Descripcio_causa_conductor': str,
         'Descripcio_causa_mediata': str,
         'Descripcio_tipus_accident': str,
         'location_id': str
     }
     df = pd.read_csv('../data/processed/df_merged.csv', dtype=dtypes, low_memory=False)
     ```

2. **Ejecución**:
   - Corre los scripts en el orden: clustering, generación de no-accidentes en horas con accidentes, y rellenado de horas faltantes.
   - Si encuentras un error `OSError: Address already in use`:
     ```bash
     lsof -i :8888
     kill -9 <pid>
     ```
     Reinicia el kernel de Jupyter en Codespaces.

3. **Salidas**:
   - Revisa `full_accidents_non_accidents.geojson` para el dataset completo.
   - Abre `full_accidents_non_accidents_clusters_map.html` en un navegador para visualizar los resultados.

## Notas
- **Rendimiento**: La generación de ~3.29M no-accidentes (32,901 horas × 100 clusters) puede ser intensiva. Si es lento, considera reducir el número de clusters o usar Dask para paralelización.
- **Validación**: Asegúrate de que `location_id` exista en el dataset original o créalo (e.g., `df_acc['location_id'] = df_acc['cluster_id']`).
- **Próximos pasos**: Usa `dataset_final` para entrenar un modelo predictivo, incorporando features como `time_n`, `cluster_id`, y las columnas descriptivas.