from __future__ import annotations

import math
import random

from settings import Settings


settings = Settings()


class ClientManager:

    _instance = None

    def __init__(self) -> None:
        self.clients_list = []
        self.clients_done = []
        self.training_in_progress = False

    @classmethod
    def get_instance(cls) -> ClientManager:
        if cls._instance:
            return cls._instance
        cls._instance = cls()
        return cls._instance

    def set_training_in_progress(self) -> None:
        self.training_in_progress = True

    def set_training_done(self) -> None:
        self.training_in_progress = False

    def generate_clients_list(self) -> list:
        # Do not generate a new list if a training session
        # is in progress alreaady
        if self.training_in_progress:
            return

        self.clients_list = random.sample(
            range(1, settings.K + 1),
            math.floor(settings.C * settings.K),
        )

    def get_clients_list(self) -> list:
        return self.clients_list

    def get_clients_done_list(self) -> list:
        return self.clients_done

    def set_client_done(self, client_id: int) -> None:
        if client_id not in self.clients_done:
            self.clients_done.append(client_id)

    def check_all_done(self) -> bool:
        if sorted(self.clients_list) == sorted(self.clients_done):
            return True
        return False

    def reset_clients_lists(self) -> None:
        self.clients_list = []
        self.clients_done = []
