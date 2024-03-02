import redis
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from schemas import ModelWeightsRequest
from client_manager import ClientManager
from services import GlobalWeightsService
from settings import Settings


settings = Settings()
router = APIRouter(
    prefix="",
    tags=["GlobalModelUpdates"],
)


@router.get("/global_weights")
def get_global_weigths():
    weights_service = GlobalWeightsService()
    weights = weights_service.get_global_model_weights()

    config = {
        "ap_high_max": 250,
        "ap_high_min": 60,
        "ap_low_max": 200,
        "ap_low_min": 60,
    }

    return JSONResponse({
        "global_weights": weights,
        "config": config,
        "E": settings.E
    })

@router.post("/global_weights/{client_id}")
def update_global_weights(weights_request: ModelWeightsRequest, client_id: int):
    client_manager = ClientManager.get_instance()

    weights = weights_request.weights
    training_test_size = weights_request.training_test_size

    # Add recived updates to updates list
    weights_service = GlobalWeightsService()
    weights_service.append_to_updated_weights_list({
        "client_id": client_id,
        "weights": weights,
        "training_test_size": training_test_size,
    })
    # Mark client as done
    client_manager.set_client_done(client_id)
    # If all clients done, update globl model
    if client_manager.check_all_done():
        weights_service.update_global_model_weights()
        client_manager.set_training_done()
        client_manager.reset_clients_lists()
        # Save global model
        weights_service.save_model()

    return JSONResponse({"message": "done"})
