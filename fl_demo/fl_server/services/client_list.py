import redis

from client_manager import ClientManager
from settings import Settings


settings = Settings()


class ClientListService:
    def get_client_list(self) -> list:
        # Get global clients state instance
        client_manager = ClientManager.get_instance()
        # Check if flag is set
        redis_conn = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
        )

        training_start = redis_conn.get("training_start")
        if not training_start:  # This means that it is the first time
                                # the service is run. Flag does not exist
            # 0 means False in this context
            redis_conn.set("training_start", "0")
        else:
            training_start = bool(int(training_start))

        if training_start:
            client_manager.generate_clients_list()
            # Lock manager
            client_manager.set_training_in_progress()
            # Set flag back to False (0)
            redis_conn.set("training_start", "0")

        return (
            client_manager.get_clients_list(),
            client_manager.get_clients_done_list()
        )
