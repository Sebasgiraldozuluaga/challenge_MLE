"""
Flight delay prediction API using FastAPI.

This module implements a REST API for predicting flight delays
using the trained XGBoost model.
"""
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from challenge.model import DelayModel

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    """
    Custom exception handler for validation errors.

    Converts HTTP 422 (Unprocessable Entity) to HTTP 400 (Bad Request)
    to match test expectations.
    """
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": exc.errors()}
    )

# Initialize the model
_delay_model = DelayModel()
_model_trained = False

# Valid values for input validation
VALID_AIRLINES = [
    'Aerolineas Argentinas', 'Aeromexico', 'Air Canada', 'Air France',
    'Alitalia', 'American Airlines', 'Austral', 'Avianca',
    'British Airways', 'Copa Air', 'Delta Air', 'Gol Trans',
    'Grupo LATAM', 'Iberia', 'JetSmart SPA', 'K.L.M.', 'Lacsa',
    'Latin American Wings', 'Oceanair Linhas Aereas',
    'Plus Ultra Lineas Aereas', 'Qantas Airways', 'Sky Airline',
    'United Airlines'
]

VALID_FLIGHT_TYPES = ['N', 'I']


class FlightData(BaseModel):
    """
    Flight information for prediction.

    Attributes:
        OPERA: Airline operator name
        TIPOVUELO: Flight type (N=Nacional, I=Internacional)
        MES: Month of the flight (1-12)
    """
    OPERA: str = Field(..., description="Airline operator name")
    TIPOVUELO: str = Field(..., description="Flight type (N or I)")
    MES: int = Field(..., ge=1, le=12, description="Month (1-12)")

    @validator('OPERA')
    def validate_airline(cls, v):
        """Validate airline name."""
        if v not in VALID_AIRLINES:
            raise ValueError(
                f"Invalid airline '{v}'. Must be one of: {VALID_AIRLINES}"
            )
        return v

    @validator('TIPOVUELO')
    def validate_flight_type(cls, v):
        """Validate flight type."""
        if v not in VALID_FLIGHT_TYPES:
            raise ValueError(
                f"Invalid flight type '{v}'. Must be one of: {VALID_FLIGHT_TYPES}"
            )
        return v


class PredictionRequest(BaseModel):
    """
    Request body for prediction endpoint.

    Attributes:
        flights: List of flight data for prediction
    """
    flights: List[FlightData] = Field(
        ...,
        description="List of flights to predict delays for",
        min_items=1
    )


class PredictionResponse(BaseModel):
    """
    Response body for prediction endpoint.

    Attributes:
        predict: List of predictions (0=no delay, 1=delay)
    """
    predict: List[int] = Field(
        ...,
        description="List of predictions (0=no delay, 1=delay)"
    )


def _train_model():
    """
    Train the model on startup with the full dataset.

    This function is called once when the API starts to ensure
    the model is ready for predictions.
    """
    global _model_trained

    if not _model_trained:
        # Load training data
        data = pd.read_csv('data/data.csv', low_memory=False)

        # Preprocess and train
        features, target = _delay_model.preprocess(
            data=data,
            target_column='delay'
        )

        _delay_model.fit(features=features, target=target)
        _model_trained = True


@app.on_event("startup")
async def startup_event():
    """Train model on API startup."""
    _train_model()


@app.get("/health", status_code=status.HTTP_200_OK)
async def get_health() -> dict:
    """
    Health check endpoint.

    Returns:
        dict: Status of the API
    """
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
async def post_predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict flight delays.

    Args:
        request: Prediction request containing flight data

    Returns:
        PredictionResponse: Predictions for each flight

    Raises:
        HTTPException: 400 if validation fails
    """
    try:
        # Convert flights to DataFrame
        flights_data = [flight.dict() for flight in request.flights]
        flights_df = pd.DataFrame(flights_data)

        # Preprocess features
        features = _delay_model.preprocess(data=flights_df)

        # Make predictions
        predictions = _delay_model.predict(features=features)

        return PredictionResponse(predict=predictions)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
