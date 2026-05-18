"""
    tarea1_template.py

    Plantilla base para la tarea de analisis de archivos CSV de gait.

    La idea de esta plantilla es que el estudiante complete las funciones
    necesarias para:
    1. cargar una "base de datos" simple con los nombres de los ficheros,
    2. leer los metadatos de cada CSV,
    3. leer las senales de interes de cada CSV,
    4. obtener la frecuencia de muestreo desde los metadatos,
    5. graficar Angle_X y Linear_Acceleration_Z usando un eje temporal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Estas son las columnas de senales que nos interesa conservar del CSV.
COLUMNAS_INTERES = [
    "Angle_X",
    "Linear_Acceleration_Z",
    "Segmentation_output",
    "Sync",
]


# ---------------------------------------------------------------------------
@dataclass
class RegistroCSV:
    # Estructura principal para representar un archivo CSV del dataset.
    # - nombre_fichero: nombre del CSV.
    # - metadatos: tabla con columnas "campo" y "valor".
    # - datos: tabla con las senales seleccionadas.

    nombre_fichero: str = ""
    metadatos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["campo", "valor"])
    )
    datos: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=COLUMNAS_INTERES)
    )

    # @property para usar un metodo como atributo
    @property
    def total_metadatos(self) -> int:
        # Cantidad de filas cargadas en la tabla de metadatos.
        return len(self.metadatos)
# ---------------------------------------------------------------------------
def listar_archivos_csv(ruta_carpeta: str) -> list[str]:
    # Recorre una carpeta y devuelve solo los nombres de archivos .csv.
    # La salida queda ordenada alfabeticamente para mantener un orden estable.
    archivos_csv = []

    for nombre in os.listdir(ruta_carpeta):
        nombre_en_minusculas = nombre.lower()

        if nombre_en_minusculas.endswith(".csv"):
            archivos_csv.append(nombre)

    archivos_csv.sort()
    return archivos_csv


# ---------------------------------------------------------------------------
def cargar_csv(path: str) -> list[str]:
    # Valida la carpeta de entrada y devuelve la lista de archivos CSV.
    path_data_base = os.path.abspath(os.path.expanduser(path))

    if not os.path.exists(path_data_base):
        raise FileNotFoundError(f"No existe la ruta: {path_data_base}")

    archivos_csv = listar_archivos_csv(path_data_base)

    if not archivos_csv:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en: {path_data_base}"
        )

    return archivos_csv


# ---------------------------------------------------------------------------
def imprimir_resumen(ruta_csv: str, num_ficheros: int) -> None:
    # Imprime un resumen basico de la carpeta cargada.
    print(f"Carpeta de ficheros: {ruta_csv}")
    print(f"# de ficheros .csv: {num_ficheros}")


# ---------------------------------------------------------------------------
def cargar_metadatos(ruta_carpeta: str, data_base: list[RegistroCSV]) -> None:
    # Carga el bloque inicial de metadatos en cada registro.
    # El CSV se lee linea por linea hasta encontrar la primera linea vacia.
    # Cada linea de metadato se separa en:
    # - campo
    # - valor
    for registro in data_base:
        archivo_csv = os.path.join(ruta_carpeta, registro.nombre_fichero)
        filas_metadatos = []

        with open(archivo_csv, "r", encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()

                if linea == "":
                    break

                partes = linea.split(",", 1)
                if len(partes) == 2:
                    campo = partes[0].strip()
                    valor = partes[1].strip()
                    filas_metadatos.append({"campo": campo, "valor": valor})

        registro.metadatos = pd.DataFrame(filas_metadatos, columns=["campo", "valor"])


# ---------------------------------------------------------------------------
def cargar_senales(ruta_carpeta: str, data_base: list[RegistroCSV]) -> None:
    # Carga solo las señales de interés del bloque numérico de cada CSV.
    for registro in data_base:
        archivo_csv = os.path.join(ruta_carpeta, registro.nombre_fichero)

        lineas_antes_bloque = 0
        with open(archivo_csv, "r", encoding="utf-8") as f:
            for linea in f:
                lineas_antes_bloque += 1
                if linea.strip() == "":
                    break

        df = pd.read_csv(archivo_csv, skiprows=lineas_antes_bloque)

        df.columns = [col.strip() for col in df.columns]

        columnas_disponibles = [col for col in COLUMNAS_INTERES if col in df.columns]
        df = df[columnas_disponibles].copy()

        for col in columnas_disponibles:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        registro.datos = df

# ---------------------------------------------------------------------------
def sombrear_intervalos_sync(ax, tiempo, sync) -> None:
    sync = np.asarray(sync)
    tiempo = np.asarray(tiempo)

    en_intervalo = False
    inicio = None

    for i in range(len(sync)):
        valor_actual = sync[i]

        if not en_intervalo and valor_actual == 1:
            inicio = tiempo[i]
            en_intervalo = True

        elif en_intervalo and valor_actual != 1:
            fin = tiempo[i]
            ax.axvspan(inicio, fin, color="lightgray", alpha=0.4)
            en_intervalo = False

    if en_intervalo:
        ax.axvspan(inicio, tiempo[-1], color="lightgray", alpha=0.4)


# ---------------------------------------------------------------------------
def graficar_registro(
    nombre_fichero: str,
    frecuencia_muestreo: float,
    angle_x,
    acc_z,
    sync,
) -> None:
    angle_x = pd.to_numeric(angle_x, errors="coerce").to_numpy()
    acc_z = pd.to_numeric(acc_z, errors="coerce").to_numpy()
    sync = pd.to_numeric(sync, errors="coerce").fillna(0).to_numpy()
    if frecuencia_muestreo is None or frecuencia_muestreo <= 0:
        raise ValueError("La frecuencia de muestreo no es válida.")
    n_muestras = len(angle_x)
    tiempo = np.arange(n_muestras) / frecuencia_muestreo

    figura, ejes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    sombrear_intervalos_sync(ejes[0], tiempo, sync)
    sombrear_intervalos_sync(ejes[1], tiempo, sync)

    ejes[0].plot(tiempo, angle_x, color="tab:blue")
    ejes[0].set_title("Angle_X")
    ejes[0].set_ylabel("Ángulo [deg]")
    ejes[0].grid(True, alpha=0.3)

    ejes[1].plot(tiempo, acc_z, color="tab:orange")
    ejes[1].set_title("Linear_Acceleration_Z")
    ejes[1].set_ylabel("Aceleración [m/s²]")
    ejes[1].set_xlabel("Tiempo [s]")
    ejes[1].grid(True, alpha=0.3)

    figura.suptitle(nombre_fichero)
    figura.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


# ---------------------------------------------------------------------------
def obtener_frecuencia_muestreo(registro: RegistroCSV) -> float | None:
    coincidencias = registro.metadatos[
        registro.metadatos["campo"].str.strip() == "Sampling Frequency"
    ]
    if coincidencias.empty:
        return None
    valor = coincidencias.iloc[0]["valor"]
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
def main() -> None:
    # Flujo principal del programa.
    data_base: list[RegistroCSV] = []

    # Calculamos la carpeta del dataset usando la ubicacion del script.
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(
        os.path.join(directorio_script, "..", "..", "data", "raw", "gait")
    )

    # Paso 1: listamos archivos CSV.
    ficheros = cargar_csv(db_path)

    # Paso 2: creamos un registro por cada archivo encontrado.
    for fichero in ficheros:
        data_base.append(RegistroCSV(nombre_fichero=fichero))

    # Paso 3: completamos metadatos y senales en cada registro.
    cargar_metadatos(db_path, data_base)
    cargar_senales(db_path, data_base)

    imprimir_resumen(db_path, len(ficheros))

    # Mostramos un resumen corto de cada archivo cargado.
    for registro in data_base[:5]:
        print(registro.nombre_fichero)
        print(registro.metadatos.head())
        print(registro.datos.head())

    # Selección interactiva del registro a graficar (reemplaza índice fijo).
    while True:
        try:
            indice = int(input(f"Seleccione un archivo para graficar (0 - {len(data_base)-1}): "))
            if 0 <= indice < len(data_base):
                break
            else:
                print("Ingrese un número válido.")
        except ValueError:
            print("Debe ingresar un entero.")

    registro = data_base[indice]
    frecuencia_muestreo = obtener_frecuencia_muestreo(registro)
    graficar_registro(
        registro.nombre_fichero,
        frecuencia_muestreo,
        registro.datos["Angle_X"],
        registro.datos["Linear_Acceleration_Z"],
        registro.datos["Sync"],
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
