import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(
    page_title="ÁREAS URBANAS BASADA EN RIESGO DE SEGURIDAD Y DEMANDA DE VIVIENDA", page_icon=":house:", layout="wide"
)

def cargar_modelos():
    modelo1 = load("Proyecto2_vfs_regresionNOVIS_v2.pkl")
    modelo2 = load("Proyecto2_vfs_regresion_VIP.pkl")
    modelo3 = load("Proyecto2_vfs_regresion_VIS.pkl")
    return modelo1, modelo2, modelo3


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


def mostrar_pesos_modelos(modelos, columnas, labels, labels_modelos):
    """
    Función para mostrar los pesos de las variables de los modelos

    Args:
        modelos (tuple): Una tupla de modelos de aprendizaje automático cargados
        columnas (list): Lista de nombres de columnas (variables)
        labels (dict): Un diccionario que mapea nombres de columnas a etiquetas descriptivas

    Returns:
        None
    """
    st.subheader("Importancia de las Variables")

    data = {}
    for i, modelo in enumerate(modelos):
        nombre_modelo = labels_modelos[i]
        importancias = None
        try:
            if hasattr(modelo, "coef_"):
                importancias = modelo.coef_
            elif hasattr(modelo, "feature_importances_"):
                importancias = modelo.feature_importances_
            elif hasattr(modelo, "named_steps") and "regression" in modelo.named_steps:
                regression_model = modelo.named_steps["regression"]
                if hasattr(regression_model, "coef_"):
                    importancias = regression_model.coef_
            else:
                st.write(f"No se pueden obtener las importancias para {nombre_modelo}.")
                continue

            importancias_dict = dict(
                zip([labels[col] for col in columnas[1:]], importancias)
            )
            data[nombre_modelo] = importancias_dict

        except Exception as e:
            st.write(
                f"No se pudieron obtener las importancias para {nombre_modelo}: {e}"
            )

    df = pd.DataFrame(data)
    st.write(df)


def main():
    """
    Función principal para construir la aplicación Streamlit
    """
    # Título y descripción para la aplicación
    st.title("ÁREAS URBANAS BASADA EN RIESGO DE SEGURIDAD Y DEMANDA DE VIVIENDA")

    with st.sidebar:
        st.write("Ingrese los datos para realizar la predicción:")

        try:
            modelos = cargar_modelos()
            st.write("¡Modelos cargados correctamente!")
        except Exception as e:
            st.error(f"Error al cargar los modelos: {e}")
            return

        with st.form(key="formulario_entrada_usuario"):
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
            valores_iniciales = [0] * len(columnas)

            entradas_usuario = []
            for i, col in enumerate(columnas):
                if col == "LOCALIDAD":
                    entrada_usuario = st.selectbox(
                        f"Seleccione la {labels[col]}:",
                        [
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
                        ],
                    )
                else:
                    entrada_usuario = st.number_input(
                        f"{labels[col]}:",
                        value=valores_iniciales[i],
                    )
                entradas_usuario.append(entrada_usuario)

            boton_prediccion = st.form_submit_button(label="Predecir")

    labels_modelos = [
        "Nuevas Viviendas de Interés Social (NOVIS)",
        "Vivienda de Interés Prioritario (VIP)",
        "Vivienda de Interés Social (VIS)",
    ]
    pesos, entrada, resultados = \
        st.tabs(["Importancia de las Variables", "Parámetros de Entrada", "Resultados"])
    
    with pesos:
        mostrar_pesos_modelos(modelos, columnas, labels, labels_modelos)
    if boton_prediccion:
        data_dict = {col: [val] for col, val in zip(columnas, entradas_usuario)}
        df_prueba = pd.DataFrame(data_dict)

        predicciones = predecir(df_prueba, modelos)
        predicciones = [round(num) for num in predicciones]
        with entrada:
            st.subheader("Entrada del Usuario")
            st.table(df_prueba)
        
        with resultados:
            st.subheader("Resultados")
            df_resultados = pd.DataFrame(
                {
                    "Proyecto de vivienda": labels_modelos,
                    "Predicción": predicciones,
                }
            )
            st.table(df_resultados)
            st.subheader("Histograma de Predicciones")
            st.bar_chart(df_resultados.set_index("Proyecto de vivienda"))

if __name__ == "__main__":
    main()
