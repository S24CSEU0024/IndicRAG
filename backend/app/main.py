"""
IndicRAG-QA: Evidence-Grounded Cross-Lingual Question Answering
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.evaluate import router as evaluate_router

app = FastAPI(
    title="IndicRAG-QA",
    description="Evidence-grounded cross-lingual question answering system for Indic and code-mixed languages",
    version="1.0.0"
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(health_router)
app.include_router(evaluate_router)


@app.get("/")
def home():
    return {
        "title": "IndicRAG-QA",
        "description": "Evidence-grounded cross-lingual QA for Indic and Code-Mixed Languages",
        "status": "online",
        "endpoints": {
            "chat": "/api/chat",
            "documents": "/api/documents",
            "health": "/health",
            "evaluate": "/api/evaluate",
            "difficult_cases": "/api/evaluate/difficult-cases"
        }
    }