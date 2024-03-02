import json
import base64
import pickle

import redis

from client_manager import ClientManager
from global_model import GlobalMLPModel
from settings import Settings


settings = Settings()


class GlobalWeightsService:
    def get_global_model_weights(self) -> str:
        global_model = GlobalMLPModel.get_instance()
        weights = global_model.get_weights()

        weights = pickle.dumps(weights)
        weights = base64.b64encode(weights).decode()

        return weights

    def append_to_updated_weights_list(self, weights_info: dict) -> None:
        global_model = GlobalMLPModel.get_instance()

        weights = base64.b64decode(weights_info["weights"])
        weights_info["weights"] = pickle.loads(weights)

        global_model.append_to_updated_weights_list(weights_info)

    def update_global_model_weights(self):
        global_model = GlobalMLPModel.get_instance()
        global_model.average_weights()

    def save_model(self):
        global_model = GlobalMLPModel.get_instance()
        global_model.save()
