from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

# Cargar el modelo
with open('RandomForestReg_GS.pkl', 'rb') as archivo_modelo:
    modelo = joblib.load(archivo_modelo)

# Usa los MISMOS nombres que espera el modelo (evita espacios si puedes)
# O bien renombra en tiempo de ejecución (abajo te muestro cómo)
columnas_modelo = [
    'Alim_CuT','Alim_CuS','Alim_CuI','Ag','Pb',
    'Fe','P80_Alim_Ro300','pH_Ro300','Tratamiento_Turno',
    'Sol_Cit','Aire_Celdas','Nivel_Celdas'
]

app = FastAPI(title="Prediccion Recuperación Cobre en Proceso Rougher")

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

@app.get("/")
def root():
    return {"status": "ok", "msg": "API viva. Usa POST /prediccion/"}

@app.post("/prediccion/")
async def predecir_recuperacion(transaccion: Transaccion):
    try:
        # DataFrame desde el input
        df_in = pd.DataFrame([transaccion.dict()])

        # Asegurar el orden/columnas que espera el modelo
        # (si tu modelo fue entrenado con 'Alim CuI' con espacio, renómbralo aquí)
        # ejemplo de mapeo por si tu modelo quedó con espacios:
        # mapeo = {"Alim_CuI": "Alim CuI"}
        # df_in = df_in.rename(columns=mapeo)
        # columnas_entrada = ['Alim_CuT','Alim_CuS','Alim CuI', ...]  # si así fue entrenado

        df_in = df_in[columnas_modelo]

        # Predicción
        pred = modelo.predict(df_in)

        return {"Recuperación Estimada (%)": float(pred[0])}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
