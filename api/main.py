"""FastAPI service that hosts the CGPA prediction model.

Run locally:
    python -m api.main
or:
    uvicorn api.main:app

Endpoints:
    GET  /health   -> model loaded + ready
    POST /predict  -> { ...11 features... } -> tier/CGPA/at-risk/explanation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.predict import get_model, predict
from api.schema import PredictRequest, PredictResponse

app = FastAPI(
    title="Student CGPA Tier Prediction API",
    description="Predicts a student's next-semester CGPA tier (0-4) and at-risk "
    "flag from prior-semester record, using the RBF-SVM model.",
    version="1.0.0",
)

# The Streamlit UI is served cross-origin in a deployed setup, so allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_model() -> None:
    try:
        get_model()
    except FileNotFoundError as e:
        # Let /health report it; do not crash the whole app.
        pass


@app.get("/health")
def health() -> dict:
    try:
        get_model()
        return {"status": "ok", "model": "loaded"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> dict:
    try:
        return predict(req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:  # defensive: Surface clear message, not a stack trace.
        raise HTTPException(status_code=500, detail=f"prediction failed: {e}")


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)