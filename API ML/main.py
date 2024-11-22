#Api de inferencia - Obligatorio Machine Learning para Sistemas Inteligentes 2024 - Universidad ORT
#Autores: Joshua Leonel y Ariel Sadi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Inicializamos la API
app = FastAPI(
    title="API de Predicción de Ratings - Leonel, Sadi",
    description="""
    Esta API utiliza el mejor modelo desarrollado por nuestro equipo para poder predecir el Rating de videojuegos 
    basado en sus características como ventas, plataforma, género y más. El modelo utilizado es el Boosting

    ## Características de cada Endpoint
    - /predict: Predice el rating del videojuego.
    - /: Comprueba el estado de la API.
    """,
)

# Cargamos nuestro mejor modelo
model_path = "modelo/Boosting_entrenado.joblib"
try:
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {str(e)}")


# Definimos como necesitamos el input para hacer una prediccion con los atributos originales
class PredictionInput(BaseModel):
    game_title: str = Field(..., description="Título del juego.")
    year: float = Field(..., description="Año de lanzamiento del juego.")
    publisher: str = Field(..., description="Nombre del editor.")
    north_america: float = Field(..., description="Ventas en Norteamérica (en millones).")
    europe: float = Field(..., description="Ventas en Europa (en millones).")
    japan: float = Field(..., description="Ventas en Japón (en millones).")
    rest_of_world: float = Field(..., description="Ventas en el resto del mundo (en millones).")
    global_: float = Field(..., description="Ventas globales (en millones).")
    number_of_reviews: str = Field(..., description="Número de reseñas (puede incluir 'K').")
    summary: str = Field(None, description="Resumen o comentarios del juego. Si está vacío, se generará automáticamente.")
    wishlist: str = Field(..., description="Cantidad de deseos en la lista (puede incluir 'K').")
    platform: str = Field(..., description="Plataforma del juego.")
    genre: str = Field(..., description="Género del juego.")

#Dejamos ya cargado una entrada valida para ejemplo y que se pueda probar. Esta es la primera entrada del DF de test
    class Config:
        schema_extra = {
            "example": {
                "game_title": "NHL 2000",
                "year": 1998,
                "publisher": "Electronic Arts",
                "north_america": 0.48,
                "europe": 0.33,
                "japan": 0.00,
                "rest_of_world": 0.06,
                "global_": 0.87,
                "number_of_reviews": 100,
                "summary": "An exciting adventure game.",
                "wishlist": 100,
                "platform": "PS",
                "genre": "Sports"
            }
        }


# Endpoint de salud para verificar si la API está funcionando
@app.get("/", summary="Endpoint de salud", description="Verifica si la API está funcionando correctamente.")
def read_root():
    return {"message": "API de inferencia está funcionando"}


# Preprocesamiento de los datos tal como definimos en el ejercicio 1
def preprocess_input(data: PredictionInput):
    # Convertir a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Cambiar nombres de columnas para que coincidan con los usados en el modelo
    input_data.rename(columns={
        'year': 'Year',
        'north_america': 'North America',
        'europe': 'Europe',
        'japan': 'Japan',
        'rest_of_world': 'Rest of World',
        'global_': 'Global',
        'number_of_reviews': 'Number of Reviews',
        'wishlist': 'Wishlist'
    }, inplace=True)


    input_data['Summary'] = input_data.apply(
        lambda row: f"{row['game_title']} no comments" if pd.isna(row['summary']) else row['summary'], axis=1
    )
    input_data = input_data[(input_data['publisher'].notnull()) & (input_data['Year'].notnull()) & (input_data['Year'] != 0)]


    def convert_k_values(value):
        if pd.isna(value):
            return value
        value_str = str(value)
        if 'K' in value_str.upper():
            number = float(value_str.upper().replace('K', ''))
            return number * 1000
        return float(value_str)

    input_data['Number of Reviews'] = input_data['Number of Reviews'].apply(convert_k_values)
    input_data['Wishlist'] = input_data['Wishlist'].apply(convert_k_values)


    input_data['Rating'] = 0


    columns_to_scale = ['Year', 'North America', 'Europe', 'Japan', 'Rest of World', 'Global', 'Number of Reviews', 'Wishlist', 'Rating']

    # Escalar los datos usando el scaler preentrenado
    scaler = joblib.load('scaler/scaler.pkl')
    input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])


    df_numerico = input_data.select_dtypes(include=['float64', 'int64'])


    if 'id' in df_numerico.columns:
        df_numerico.drop('id', axis=1, inplace=True)


    df_numerico.drop('Rating', axis=1, inplace=True)

    # Convertir nuevamente a array para el modelo
    return df_numerico.values



# Endpoint de inferencia
@app.post("/predict/", summary="Realiza una predicción", description=" Este endpoint toma las características de un videojuego como entrada y devuelve el rating estimado en una escala de 0 a 10. A continuación damos un ejemplo de request (1er entrada del DF de test).")
def predict(input_data: PredictionInput):
    try:
        # Preprocesar los datos
        preprocessed_data = preprocess_input(input_data)

        # Realizar la predicción
        normalized_prediction = model.predict(preprocessed_data)

        # Desnormalizar la predicción: sabiendo que el rango es [0, 10]
        original_prediction = [pred * 10 for pred in normalized_prediction]


        # Retornar el resultado
        return {
            "Rating": original_prediction
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")
