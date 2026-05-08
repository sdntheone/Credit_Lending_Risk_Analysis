from fastapi import FastAPI

from app.routes import router


app = FastAPI(
    title="Credit Risk Prediction API",
    description="MLOps project using FastAPI",
    version="1.0"
)

app.include_router(router)