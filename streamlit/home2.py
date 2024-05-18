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
    # Título y descripción para la aplicación
    st.title("ÁREAS URBANAS BASADA EN RIESGO DE SEGURIDAD Y DEMANDA DE VIVIENDA")
    st.write("Ingrese los datos para realizar la predicción:")

    try:
        modelo = cargar_modelo()
        st.write("¡Modelo cargado correctamente!")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return  # Salir si falla la carga del modelo

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
        predicciones = predecir(df_prueba, modelo)
        prediccion_real = round(predicciones[0])

        # Muestra el DataFrame de entrada del usuario como una tabla
        st.subheader("Entrada del Usuario:")
        st.table(df_prueba)  # Muestra el DataFrame como una tabla
        # Muestra las predicciones
        st.write(f"Valor Predicho: {prediccion_real}")

def cargar_modelo():
    # Reemplaza 'ruta/a/modelo.pkl' con la ruta real a tu modelo de scikit-learn almacenado
    modelo = load('./model/Proyecto2_vfs_regresionNOVIS_v2.pkl')
    return modelo

if __name__ == "__main__":
    main()
