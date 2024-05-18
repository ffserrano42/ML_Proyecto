import streamlit as st
import pandas as pd
from joblib import load


def predecir(data, modelos):
    """
    Función para realizar predicciones utilizando varios modelos

    Args:
        data (pandas.DataFrame): El DataFrame que contiene los datos para la predicción
        modelos (tuple): Una tupla de modelos de aprendizaje automático cargados

    Returns:
        list: Una lista de valores predichos por cada modelo
    """
    predicciones = []
    for modelo in modelos:
        pred = modelo.predict(data)
        predicciones.append(pred[0])
    return predicciones

def cargar_modelos():
    modelo1 = load('Proyecto2_vfs_regresionNOVIS_v2.pkl')
    modelo2 = load('Proyecto2_vfs_regresion_VIP.pkl')
    modelo3 = load('Proyecto2_vfs_regresion_VIS.pkl')
    return modelo1, modelo2, modelo3

def main():
    """
    Función principal para construir la aplicación Streamlit
    """
    # Título y descripción para la aplicación
    st.title("ÁREAS URBANAS BASADA EN RIESGO DE SEGURIDAD Y DEMANDA DE VIVIENDA")
    st.write("Ingrese los datos para realizar la predicción:")

    try:
        modelos = cargar_modelos()
        st.write("¡Modelos cargados correctamente!")
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return

    with st.form(key="formulario_entrada_usuario"):
        # Campos de entrada para los datos del usuario
        st.write("Ingrese los datos para la predicción:")
        columnas = [
            "LOCALIDAD", "AnimalesYMedioAmbiente",
            "DanosYPeligrosEnPropiedadesEInfraestructuras", "EmergenciasMedicasYDeSalud",
            "EmergenciasPorSucesosNaturales", "IncendiosYExplosiones", "NoClasificado",
            "OtrosIncidentes", "PersonasEnSituacionDeRiesgo", "RescatesYSalvamento",
            "SeguridadYOrdenPublico"
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
            "SeguridadYOrdenPublico": "Seguridad y Orden Público"
        }
        valores_iniciales = [0] * len(columnas)  # Inicializa con ceros para los campos de entrada
        entradas_usuario = []
        for i, col in enumerate(columnas):
            if col == "LOCALIDAD":
                entrada_usuario = st.selectbox(f"Seleccione la {labels[col]}:", [
                    "ANTONIO NARIÑO", "BARRIOS UNIDOS", "BOSA", "CANDELARIA", "CHAPINERO",
                    "CIUDAD BOLIVAR", "ENGATIVA", "FONTIBON", "KENNEDY", "LOS MARTIRES",
                    "PUENTE ARANDA", "RAFAEL URIBE URIBE", "SAN CRISTOBAL", "SANTA FE",
                    "SUBA", "TEUSAQUILLO", "TUNJUELITO", "USAQUÉN", "USME"
                ])
            else:
                entrada_usuario = st.number_input(f"Número de Incidentes de {labels[col]}:", value=valores_iniciales[i])
            entradas_usuario.append(entrada_usuario)

        # Verifica si el usuario envió el formulario
        boton_prediccion = st.form_submit_button(label="Predecir")

    if boton_prediccion:
        # Crear un DataFrame con los datos del usuario
        data_dict = {col: [val] for col, val in zip(columnas, entradas_usuario)}
        df_prueba = pd.DataFrame(data_dict)

        # Realiza predicciones utilizando la función 'predecir'
        predicciones = predecir(df_prueba, modelos)
        predicciones = [round(num) for num in predicciones]

        # Muestra el DataFrame de entrada del usuario como una tabla
        st.subheader("Entrada del Usuario:")
        st.table(df_prueba)  # Muestra el DataFrame como una tabla
        st.subheader("Resultados:")
        # Muestra las predicciones
        st.table(pd.DataFrame({"Proyecto de vivienda": ["Nuevas Viviendas de Interés Social (NOVIS)","Vivienda de Interés Prioritario (VIP)","Vivienda de Interés Social (VIS)"], "Predicción": predicciones}))

if __name__ == "__main__":
    main()
