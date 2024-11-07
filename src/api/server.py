import uvicorn
from fastapi import FastAPI
from src.core.config import Settings
from src.core.model_manager import ModelManager
from src.api import app

# Initialize settings and create singleton ModelManager
settings = Settings()
model_manager = ModelManager(max_active_models=settings.max_active_models)


async def get_model_manager():
    return model_manager


# Create FastAPI app
app = app.create_app(model_manager)


def run_server():
    """Run the API server using configuration from settings."""
    uvicorn.run(
        "src.api.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers,
    )


if __name__ == "__main__":
    run_server()
