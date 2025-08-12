from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging, traceback, os, sys

# ----------------------------
# Config / Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "RandomForestReg_GS.pkl"

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="API Predicción Recuperación", version="1.0.0")

logging.warning("Python version: %s", sys.version)
logging.warning("PORT: %s", os.getenv("PORT"))
logging.warning("Cargando modelo desde: %s", MODEL_PATH)

# ----------------------------
# Carga del modelo y columnas
# ----------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        modelo = joblib.load(f)
    logging.warning("Modelo cargado OK.")
    try:
        COLUMNAS = list(modelo.feature_names_in_)
        logging.warning("COLUMNAS detectadas desde el modelo: %s", COLUMNAS)
    except AttributeError:
        COLUMNAS = []
        logging.error("El modelo no tiene atributo feature_names_in_.")
except Exception as e:
    logging.exception("Error cargando modelo: %s", e)
    modelo = None
    COLUMNAS = []

# ----------------------------
# Esquema de entrada flexible
# ----------------------------
class Payload(BaseModel):
    __root__: Dict[str, Any] = Field(default_factory=dict)

# ----------------------------
# Helpers
# ----------------------------
def normaliza_claves(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliza claves del JSON: trim, reemplaza _ por espacio."""
    out = {}
    for k, v in d.items():
        if not isinstance(k, str):
            k = str(k)
        kk = k.strip().replace("_", " ")
        out[kk] = v
    return out

def validar_y_ordenar(entrada: Dict[str, Any]) -> pd.DataFrame:
    """Valida que estén todas las columnas requeridas y las ordena."""
    if not COLUMNAS:
        raise HTTPException(status_code=500, detail="No se detectaron columnas en el modelo.")

    entrada_norm = normaliza_claves(entrada)
    df = pd.DataFrame([entrada_norm])

    faltantes = [c for c in COLUMNAS if c not in df.columns]
    if faltantes:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Faltan características requeridas",
                "faltantes": faltantes,
                "recibidas": sorted(df.columns.tolist()),
                "esperadas": COLUMNAS,
            },
        )

    df = df[COLUMNAS].copy()

    try:
        df = df.apply(pd.to_numeric)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error convirtiendo tipos numéricos: {e}",
        )

    return df

# ----------------------------
# Rutas
# ----------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/healthz")
def healthz():
    ok = modelo is not None
    return {"ok": ok, "features": COLUMNAS}

@app.post("/prediccion")
def prediccion(payload: Payload):
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible en el servidor.")
    try:
        data = payload.__root__ or {}
        df_entrada = validar_y_ordenar(data)
        y_pred = modelo.predict(df_entrada.values.astype(np.float32))
        return {"Recuperacion_Estimada_pct": float(y_pred[0])}
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Unhandled prediccion: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))

@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logging.error("Unhandled global: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": "internal_error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        workers=1,
    )
