from typing import List, Tuple, Optional

import os
import sys
import time
import pickle
import base64
import logging

import pandas as pd
import requests
from requests.exceptions import RequestException

from sklearn.model_selection import train_test_split

from model import MLPModel
from database import SQLite
from schemas import Configuration
from transformations import FeatureTransformer
from settings import Settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
settings = Settings()


def pooling_wait() -> None:
    """Wait time before pooling"""
    logging.info("Waiting for next pooling cycle...")
    time.sleep(settings.POOLING_TIME)

def get_remote_client_list() -> Tuple[List[str], List[str]]:
    """
    Get the list of clients that will participate
    on the training round
    """
    try:
        response = requests.get(
            f"{settings.SERVER_URL}/client_list",
            timeout=settings.SERVER_TIMEOUT,
        )
        parsed_response = response.json()
        client_list = parsed_response["client_list"]
        client_done = parsed_response["client_done"]
    except (RequestException, KeyError) as err:
        logging.error(err)
        return [], []

    logging.info(f"Received client list from server: {client_list}")

    return client_list, client_done

def get_training_params() -> Tuple[Optional[list], Optional[Configuration]]:
    """Get training params from server"""
    try:
        response = requests.get(
            f"{settings.SERVER_URL}/global_weights",
            timeout=settings.SERVER_TIMEOUT,
        )
        parsed_response = response.json()
        global_weights = parsed_response["global_weights"]
        configuration = Configuration(**parsed_response["config"])
        E = parsed_response["E"]
    except (RequestException, KeyError) as err:
        logging.error(err)
        return None, None, None

    # Convert weights from base64
    global_weights = base64.b64decode(global_weights)
    global_weights = pickle.loads(global_weights)

    logging.info(f"Got training parameters from server:")
    logging.info(f"Configuration: {str(configuration)}")
    logging.info(f"E: {str(E)}")

    return global_weights, configuration, E

def send_updated_weights_to_server(
        weights: list,
        training_test_size: int,
    ) -> None:
    """Send updated weights to server"""
    # Convert weights to base64
    weights = pickle.dumps(weights)
    weights = base64.b64encode(weights).decode()

    try:
        logging.info("Sending updated weights to server")

        response = requests.post(
            f"{settings.SERVER_URL}/global_weights/{settings.CLIENT_ID}",
            timeout=settings.SERVER_TIMEOUT,
            json={
                "weights": weights,
                "training_test_size": training_test_size,
            },
        )
        if response.status_code != 200:
            raise RequestException
    except RequestException as err:
        logging.error(err)

def get_local_training_data() -> pd.DataFrame:
    """Get local training data"""
    database = SQLite(database=settings.LOCAL_DATABASE)
    data = database.get_local_training_set()
    return data

def get_target_data(data: pd.DataFrame) -> pd.Series:
    """Get target data from training data"""
    target = data["cardio"]
    data.drop("cardio", axis=1, inplace=True)
    return target

def save_history(history: dict) -> None:
    """Save the train history data"""
    if os.path.exists(f"{settings.TRAIN_HISTORY_DIR}/history.pickle"):
        with open(f"{settings.TRAIN_HISTORY_DIR}/history.pickle", "rb") as fp:
            old_history = pickle.load(fp)

            for key in history.keys():
                history[key] = [*old_history[key], *history[key]]

    with open(f"{settings.TRAIN_HISTORY_DIR}/history.pickle", "wb") as fp:
        pickle.dump(history, fp)


def main_loop():
    """Main client loop"""
    while True:
        # Pooling wait time
        pooling_wait()
        # Get clent list from server
        client_list, client_done = get_remote_client_list()
        # Prevent client from start another communication round
        # before all clients are done
        if settings.CLIENT_ID in client_done:
            continue

        if settings.CLIENT_ID in client_list:
            logging.info(
                "Found own ID on client list. Training is about to begin"
            )
            global_weights, configuration, E = get_training_params()
            # Get local data
            data = get_local_training_data()
            # Get target for training
            target = get_target_data(data)
            # Transform local data
            features, droped_indexes = FeatureTransformer(configuration).apply(data=data)
            # Adjust target data
            target = target.drop(index=droped_indexes)
            # Create local model
            model = MLPModel(features.shape[1])
            model.set_weights(weights=global_weights)
            model.compile()
            # Start training
            history = model.fit(
                x_train=features,
                y_train=target,
                epochs=E,
                batch_size=32,
                validation_split=0.2,
            )
            # Send new model weights to server
            send_updated_weights_to_server(
                weights=model.get_weights(),
                training_test_size=features.shape[0],
            )
            # Save training history
            save_history(history.history)

            logging.info("Communication round done")

if __name__ == "__main__":
    if not settings.CLIENT_ID:
        logging.error("Could not determine client id. Exiting...")
        sys.exit(1)

    main_loop()
