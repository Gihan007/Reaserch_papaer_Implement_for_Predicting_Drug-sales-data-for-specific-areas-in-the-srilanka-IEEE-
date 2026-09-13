from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from libs.common.metrics import install_metrics
from services.forecast_service.app.forecasting import forecast_sales, stacking_response_details


app = FastAPI(title="Forecast Service")
install_metrics(app, "forecast-service")


class ForecastRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    category: str
    date: str
    model_type: str = "ensemble"


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "forecast-service"}


@app.post("/forecast")
async def forecast(payload: ForecastRequest):
    try:
        forecast_value, closest_prediction_date, plot_file, model_used = forecast_sales(
            payload.category,
            payload.date,
            payload.model_type,
        )
        return {
            "success": True,
            "forecast_value": float(forecast_value) if forecast_value is not None else 0.0,
            "closest_prediction_date": (
                closest_prediction_date.strftime("%Y-%m-%d")
                if hasattr(closest_prediction_date, "strftime")
                else str(closest_prediction_date)
            ),
            "plot_url": plot_file,
            "model_used": str(model_used),
            "category": payload.category,
            "input_date": payload.date,
            **(stacking_response_details(payload.category, forecast_value, closest_prediction_date)
               if payload.model_type == "stacking" else {}),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
