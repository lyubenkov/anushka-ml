from fastapi import FastAPI, Depends
from src.core.config import Settings
from src.core.model_manager import ModelManager


def create_app(model_manager: ModelManager) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = Settings()

    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        debug=settings.debug,
    )

    async def get_model_manager():
        return model_manager

    from src.api.routes import router

    api_prefix = f"{settings.prefix}/{settings.api_version}"
    app.include_router(
        router, prefix=api_prefix, dependencies=[Depends(get_model_manager)]
    )

    app.dependency_overrides[ModelManager] = get_model_manager

    return app
