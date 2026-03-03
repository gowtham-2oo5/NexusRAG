from fastapi import FastAPI

from nexus_app.core.logging_config import configure_logging
from nexus_app.core.middleware import create_request_logger_middleware
from nexus_app.routes.api import create_router
from nexus_app.services.processing import init_executor
from nexus_app.utils.constants import MAX_WORKERS
import google.generativeai as genai
from nexus_app.core.settings import get_settings


def create_app() -> FastAPI:
    app = FastAPI(title="PlaylearnAI - RAG demo", version="3.0")

    logger = configure_logging()
    settings = get_settings()
    logger.info("✅ OpenAI API key loaded")
    logger.info("✅ Gemini API key loaded")

    genai.configure(api_key=settings.gemini_api_key)

    executor = init_executor(MAX_WORKERS)

    # Middleware
    app.middleware("http")(create_request_logger_middleware(logger))

    # Routes
    app.include_router(create_router(logger, executor))

    # Lifecycle events
    @app.on_event("startup")
    async def on_startup():
        logger.info(
            "🚀 API starting up with comprehensive file support: PPT, Images, Excel, ZIP, BIN with 1GB size limit"
        )

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("🔥 Shutting down thread pool executor")
        executor.shutdown(wait=True)

    return app


