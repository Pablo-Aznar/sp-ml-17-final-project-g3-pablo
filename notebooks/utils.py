import pandas as pd
import glob
import os

import pandas as pd
import glob
import os

# --- Función para leer y unificar un dataset multi-año ---
def cargar_y_unificar(ruta_carpeta, cols_redundantes, col_a_traer):
    """
    ruta_carpeta: carpeta donde están los CSV
    variantes_clave: lista de nombres posibles de la columna 'numero_expediente'
    col_a_traer: lista con los posibles nombres de la columna que queremos conservar
    """
    archivos = glob.glob(os.path.join(ruta_carpeta, "*.csv"))
    dfs = []

    for archivo in archivos:
        df = pd.read_csv(archivo)

        # --- Renombrar columna clave ---
        for col in cols_redundantes:
            if col in df.columns:
                df.rename(columns={col: "Numero_expedient"}, inplace=True)
                break

        # --- Renombrar columna de interés ---
        for col in col_a_traer:
            if col in df.columns:
                df.rename(columns={col: col_a_traer[0]}, inplace=True)  # usar el primero como nombre final
                break

        # --- Normalizar la clave ---
        df["Numero_expedient"] = df["Numero_expedient"].astype(str).str.strip()

        # --- Conservar solo clave y columna deseada ---
        dfs.append(df[["Numero_expedient", col_a_traer[0]]])

    # Unir todos los años en un solo DataFrame
    df_total = pd.concat(dfs, ignore_index=True)

    # --- Agrupar por Numero_expedient ---
    def combinar_valores(series):
        valores = [v for v in series if pd.notna(v) and str(v).strip() != ""]
        if len(valores) == 0:
            return pd.NA  # todos eran NaN o vacíos
        else:
            return ", ".join(sorted(set(map(str, valores))))  # combinar solo no nulos

    df_total = df_total.groupby("Numero_expedient", as_index=False).agg({
        col_a_traer[0]: combinar_valores
    })

    return df_total


# --- Función para limpiar la clave ---
def limpiar_clave(df, col="Numero_expedient"):
    """
    Normaliza la columna clave: la convierte a str, quita espacios, saltos de línea y caracteres invisibles.
    """
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.replace(r"[\u200B-\u200D\uFEFF]", "", regex=True)  # caracteres invisibles
        .str.replace("\n", "", regex=False)
        .str.replace("\r", "", regex=False)
    )
    return df