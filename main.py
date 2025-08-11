from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import uvicorn

# Cargar el modelo y el scaler desde los archivos .pkl
with open('RandomForestReg_GS.pkl', 'rb') as archivo_modelo:
    modelo = joblib.load(archivo_modelo)

# Lista de características en el orden esperado por el modelo
columnas = ['Alim_CuT','Alim_CuS','Alim CuI','Ag','Pb',
            'Fe','P80_Alim_Ro300','pH_Ro300','Tratamiento_Turno',
            'Sol_Cit','Aire_Celdas','Nivel_Celdas']

# Crear la aplicación FastAPI
app = FastAPI(title="Prediccion Recuperación Cobre en Proceso Rougher")

# Definir el modelo de datos de entrada utilizando Pydantic
class Transaccion(BaseModel):
    Alim_CuT: float
    Alim_CuS: float
    Alim_CuI: float
    Ag: float
    Pb: float
    Fe: float
    P80_Alim_Ro300: float
    pH_Ro300: float
    Tratamiento_Turno: float
    Sol_Cit: float
    Aire_Celdas: float
    Nivel_Celdas: float

# Definir el endpoint para predicción
@app.post("/prediccion/")
async def predecir_recuperacion(transaccion: Transaccion):
    try:
        # Convertir la entrada en un DataFrame
        datos_entrada = pd.DataFrame([transaccion.dict()], columns=columnas)
        
        # Realizar la predicción con el modelo cargado
        prediccion = modelo.predict(datos_entrada)
        
        # Construir la respuesta
        resultado = {
            "Recuperación Estimada (%)": float(prediccion[0])
        }
        
        return resultado
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if __name__ == "__main__":
        uvicorn.run(app, port=8080,host="0.0.0.0")
