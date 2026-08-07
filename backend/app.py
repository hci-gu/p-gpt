import logging

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from logging_config import configure_persistent_logging
from routers.evaluations import router as evaluations_router
from routers.models import router as models_router
from routers.personas import router as personas_router
from routers.pseudo_stream import router as pseudo_stream_router
from routers.requests import router as requests_router
from routers.speaker import router as speaker_router
from services.lifecycle import lifespan


logger = logging.getLogger("uvicorn.error.p_gpt")
persistent_log_path = configure_persistent_logging(
    logger,
    backup_count=settings.log_backup_count,
    level_name=settings.log_level,
    max_bytes=settings.log_max_bytes,
    path=settings.log_path,
)
logger.info(
    "P-GPT logging configured: level=%s persistent_log=%s",
    settings.log_level,
    persistent_log_path,
)
logger.info("Running mlflow on tracking URI: %s", mlflow.get_tracking_uri())

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(models_router)
app.include_router(evaluations_router)
app.include_router(personas_router)
app.include_router(requests_router)
app.include_router(pseudo_stream_router)
app.include_router(speaker_router)
