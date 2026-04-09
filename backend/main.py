from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Medical Disease Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    return {"status": "healthy", "service": "Medical Disease Detection API"}

# Placeholder for router inclusion
from api.pneumonia_routes import router as pneumonia_router
from api.tuberculosis_routes import router as tuberculosis_router
app.include_router(pneumonia_router, prefix="/api/v1")
app.include_router(tuberculosis_router, prefix="/api/v1")
