from fastapi import FastAPI

from global_model import GlobalMLPModel
from settings import Settings

from routers import client_list, global_weights


settings = Settings()

# Initialize global model
global_model = GlobalMLPModel.get_instance()
global_model.build_model(settings.MODEL_INPUT_SHAPE)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
)

app.include_router(client_list.router)
app.include_router(global_weights.router)

@app.get("/")
async def root():
    """Api version"""
    return {"version": settings.APP_VERSION}
