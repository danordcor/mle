import fastapi
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator
import pandas as pd
from typing import List
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

from challenge.model import DelayModel

app = fastapi.FastAPI()
cache_backend = InMemoryBackend()
FastAPICache.init(cache_backend)
model = DelayModel()


class FlightDetail(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("OPERA")
    def validate_OPERA(cls, v):
        OPERATORS = [
            "Grupo LATAM",
            "Sky Airline",
            "Aerolineas Argentinas",
            "Copa Air",
            "Latin American Wings",
            "Avianca",
            "JetSmart SPA",
            "Gol Trans",
            "American Airlines",
            "Air Canada",
            "Iberia",
            "Delta Air",
            "Air France",
            "Aeromexico",
            "United Airlines",
            "Oceanair Linhas Aereas",
            "Alitalia",
            "K.L.M.",
            "British Airways",
            "Qantas Airways",
            "Lacsa",
            "Austral",
            "Plus Ultra Lineas Aereas",
        ]
        if v not in OPERATORS:
            raise ValueError("Invalid Operator")
        return v

    @validator("TIPOVUELO")
    def validate_TIPOVUELO(cls, v):
        if v not in ["N", "I"]:
            raise ValueError("Invalid flight type")
        return v

    @validator("MES")
    def validate_MES(cls, v):
        if v not in range(1, 13):
            raise ValueError("Invalid month")
        return v


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return fastapi.responses.JSONResponse(
        status_code=400,
        content={"detail": str(exc.errors())}
    )


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


class FlightList(BaseModel):
    flights: List[FlightDetail]



def predict_cached(df):
    features = model.preprocess(df)
    predictions = model.predict(features)
    return predictions


@app.post("/predict", status_code=200)
#@cache(expire=60)  # Caché ttl 1 minute
async def post_predict(data: FlightList) -> dict:
    try:
        df = pd.DataFrame([flight.dict() for flight in data.flights])
        predictions = predict_cached(df)
        return {"predict": predictions}
    except Exception as e:
        return {"error": "An error occurred during prediction.", "detail": str(e)}
