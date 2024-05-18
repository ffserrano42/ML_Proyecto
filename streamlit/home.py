import streamlit as st
import pandas as pd
from joblib import load

# Import your model loading function (replace with your actual logic)
#from model_loader import load_model  # Assuming you have a separate module

def predict(data, model):
  """
  Function to make predictions using your model

  Args:
      data (pandas.DataFrame): The DataFrame containing data for prediction
      model: The loaded machine learning model

  Returns:
      list: A list of predicted values
  """
  # Preprocess the data if needed (e.g., feature selection, scaling)
  # ... your preprocessing steps ...

  # Make predictions using the model
  predictions = model.predict(data)
  return predictions.tolist()  # Convert predictions to a list

def main():
  """
  Main function to build the Streamlit app
  """
  # Title and description for the app
  st.title("Prediction App")
  st.write("Enter data for prediction:")  
  try:
    model = load_model()
    st.write("Model loaded successfully!")
  except Exception as e:
    st.error(f"Error loading model: {e}")
    return  # Exit if model loading fails
  with st.form(key="user_input_form"):
    
    submit_button = st.form_submit_button(label="Predict")

  # Check if user submitted the form
  if submit_button:
    # Convert user input table to a DataFrame
    column_names=["LOCALIDAD","AnimalesYMedioAmbiente","DanosYPeligrosEnPropiedadesEInfraestructuras","EmergenciasMedicasYDeSalud","EmergenciasPorSucesosNaturales","IncendiosYExplosiones","NoClasificado","OtrosIncidentes","PersonasEnSituacionDeRiesgo","RescatesYSalvamento","SeguridadYOrdenPublico"]
    # estos son los originales obtenidos de la base de datos la primera fila
    data_value=["BARRIOS UNIDOS",15,7,11,0,9,25,3,8,1,13]
    #estos son valores inventados para probar el modelo
    #data_value=["SUBA",0,5,1,153,0,8,13,1.333333333,28,10,1,5,0,13.33333333,10.66666667,4.666666667,11,22,2,5,3.666666667,0,2,5.666666667,16,0,0,12,4.5,0,5.333333333,0,4,11.33333333,0,11,13.66666667,52.33333333,9.333333333,17.33333333,1,1.666666667,4.333333333,6.5,3,3,2,10.66666667,0,0,33,2.333333333,54,9,1.5,0,3,4.5,28.66666667,0,1,5.666666667,49.66666667,3,2,1.333333333,1,3,4,1,38,1.333333333,1,160,119,0,1,0,3.333333333,4.333333333,2.666666667,19.66666667,13.33333333,29.33333333,1.5,79,3]
    df_test = pd.DataFrame(data=[data_value], columns=column_names)
    for col in df_test.columns:
        if pd.api.types.is_numeric_dtype(df_test[col]):  # Check if numeric
            df_test[col] = df_test[col].round().astype('int64')    
    # Make predictions using the 'predict' function
    predictions = predict(df_test,model)
    prediction_real= round(predictions[0])
    # Display the user input DataFrame as a table
    st.subheader("User Input:")
    st.table(df_test)  # Display the DataFrame as a table
    # Display the predictions
    st.write(f"Predictions:{prediction_real}")

def load_model():
  # Replace 'path/to/model.pkl' with the actual path to your pickled scikit-learn model
  model = load('./model/Proyecto2_vfs_regresionNOVIS_v2.pkl')
  return model

if __name__ == "__main__":
  main()
