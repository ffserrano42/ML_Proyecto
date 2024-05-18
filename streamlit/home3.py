import streamlit as st
import pandas as pd
from joblib import load


def predecir(data, modelo):
    """
    Función para realizar predicciones utilizando un modelo

    Args:
        data (pandas.DataFrame): El DataFrame que contiene los datos para la predicción
        modelo: El modelo de aprendizaje automático cargado

    Returns:
        list: Una lista de valores predichos
    """
    # Realiza predicciones utilizando el modelo
    predicciones = modelo.predict(data)
    return predicciones.tolist()  # Convierte las predicciones en una lista


def main():
    """
    Función principal para construir la aplicación Streamlit
    """
    st.markdown(
        """
        <style>
        .block-container {
            width: 100% !important;
            max-width: 100% !important;
            padding: 1em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Título y descripción para la aplicación
    st.title("ÁREAS URBANAS BASADA EN RIESGO DE SEGURIDAD Y DEMANDA DE VIVIENDA")
    st.write("Ingrese los datos para realizar la predicción:")

    try:
        modelo = cargar_modelo()
        st.write("¡Modelo cargado correctamente!")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return  # Salir si falla la carga del modelo

    # Datos para la grilla de localidades
    localidades = [
        "ANTONIO NARIÑO",
        "BARRIOS UNIDOS",
        "BOSA",
        "CANDELARIA",
        "CHAPINERO",
        "CIUDAD BOLIVAR",
        "ENGATIVA",
        "FONTIBON",
        "KENNEDY",
        "LOS MARTIRES",
        "PUENTE ARANDA",
        "RAFAEL URIBE URIBE",
        "SAN CRISTOBAL",
        "SANTA FE",
        "SUBA",
        "TEUSAQUILLO",
        "TUNJUELITO",
        "USAQUÉN",
        "USME",
    ]
    columnas = [
        "LOCALIDAD",
        "AnimalesYMedioAmbiente",
        "DanosYPeligrosEnPropiedadesEInfraestructuras",
        "EmergenciasMedicasYDeSalud",
        "EmergenciasPorSucesosNaturales",
        "IncendiosYExplosiones",
        "NoClasificado",
        "OtrosIncidentes",
        "PersonasEnSituacionDeRiesgo",
        "RescatesYSalvamento",
        "SeguridadYOrdenPublico",
    ]
    labels = {
        "LOCALIDAD": "Localidad",
        "AnimalesYMedioAmbiente": "Animales y Medio Ambiente",
        "DanosYPeligrosEnPropiedadesEInfraestructuras": "Daños y Peligros en Propiedades e Infraestructuras",
        "EmergenciasMedicasYDeSalud": "Emergencias Médicas y de Salud",
        "EmergenciasPorSucesosNaturales": "Emergencias por Sucesos Naturales",
        "IncendiosYExplosiones": "Incendios y Explosiones",
        "NoClasificado": "No Clasificado",
        "OtrosIncidentes": "Otros Incidentes",
        "PersonasEnSituacionDeRiesgo": "Personas en Situación de Riesgo",
        "RescatesYSalvamento": "Rescates y Salvamento",
        "SeguridadYOrdenPublico": "Seguridad y Orden Público",
    }

    # Inicializa con ceros para los campos de entrada
    valores_iniciales = {col: 0 for col in columnas[1:]}

    cols = st.columns(len(columnas))
    for i, col in enumerate(columnas):
        cols[i].write(labels[col])

    datos_localidades = []

    for localidad in localidades:
        cols = st.columns(len(columnas))
        entradas_usuario = [localidad]  # Primera columna no editable
        cols[0].write(localidad)
        for i, col in enumerate(columnas[1:], start=1):
            valor = cols[i].number_input(
                "-",
                min_value=0,
                value=valores_iniciales[col],
                key=f"{localidad}_{col}",
                label_visibility="hidden",
            )
            entradas_usuario.append(valor)
        datos_localidades.append(entradas_usuario)

    # Verifica si el usuario envió el formulario
    if st.button("Predecir"):
        # Crear un DataFrame con los datos de las localidades
        df_prueba = pd.DataFrame(data=datos_localidades, columns=columnas)
        # Realiza predicciones utilizando la función 'predecir'
        predicciones = predecir(df_prueba, modelo)

        # Mostrar la tabla final con las predicciones
        st.subheader("Predicciones para todas las localidades:")
        st.table(pd.DataFrame({"Localidad": localidades, "Predicción": predicciones}))


def cargar_modelo():
    # Reemplaza 'ruta/a/modelo.pkl' con la ruta real a tu modelo de scikit-learn almacenado
    modelo = load("./model/Proyecto2_vfs_regresionNOVIS_v2.pkl")
    return modelo


if __name__ == "__main__":
    main()
