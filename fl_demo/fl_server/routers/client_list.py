import redis
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services import ClientListService
from settings import Settings


settings = Settings()
router = APIRouter(
    prefix="",
    tags=["ClientList"],
)


@router.get("/client_list")
def get_client_list():
    client_list, client_done = ClientListService().get_client_list()

    return JSONResponse({
        "client_list": client_list,
        "client_done": client_done,
    })
