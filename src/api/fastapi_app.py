from fastapi import FastAPI
from src.api.routers import fraud as fraud_router

app = FastAPI(title="Aegis API")

# include routers
app.include_router(fraud_router.router, prefix="/fraud", tags=["fraud"])

@app.get("/health")
def health():
    return {"status": "ok"}
