from __future__ import annotations

import pickle
import logging
from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from settings import Settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
settings = Settings()


class GlobalMLPModel:

    _instance = None

    def __init__(self) -> None:
        self.model = None
        self.updated_weights_list = []

    @classmethod
    def get_instance(cls) -> GlobalMLPModel:
        if cls._instance:
            return cls._instance
        cls._instance = cls()
        return cls._instance

    def build_model(self, num_features: int) -> None:
        """Scaffold the model"""
        self.model = Sequential(
            [
                Dense(128, activation="relu", input_shape=(num_features,)),
                Dropout(0.2),
                Dense(128, activation="relu"),
                Dropout(0.2),
                Dense(64, activation="relu", input_shape=(num_features,)),
                Dropout(0.2),
                Dense(1, activation="sigmoid"),
            ]
        )

    def get_weights(self) -> list:
        return self.model.get_weights()

    def append_to_updated_weights_list(self, updated_weights: dict) -> None:
        self.updated_weights_list.append(updated_weights)

    def average_weights(self) -> None:
        scaled_weights = []
        # Calculate the total number of data points from all clients
        clients_local_data_points = [
            weights_info["training_test_size"]
            for weights_info in self.updated_weights_list
        ]
        total_data_points = sum(clients_local_data_points)
        # Apply scaling factor to all layers of each model
        for weights_info in self.updated_weights_list:
            scaling_factor = weights_info["training_test_size"] / total_data_points

            scaled_local_model_weights = []
            for i in range(len(weights_info["weights"])):
                scaled_local_model_weights.append(
                    scaling_factor * weights_info["weights"][i]
                )

            scaled_weights.append(scaled_local_model_weights)
        # Sum all local models weights
        updated_weights = []
        for i in range(len(scaled_weights[0])): # Length of the weights vector
            averaged_weights = []
            for weights in scaled_weights:
                averaged_weights.append(weights[i])

            updated_weights.append(sum(averaged_weights))
        # Set updated weights to global model
        self.model.set_weights(updated_weights)

        # Reset all structures
        self.updated_weights_list = []

        logging.info("Averaging process done")
        logging.info(f"Clients local data points: {clients_local_data_points}")
        logging.info(f"Total data points: {total_data_points}")

        logging.info(f"Length of scaled weights: {len(scaled_weights)}")

        logging.info("Weights for local model 1:")
        logging.info(scaled_weights[0][0])

        logging.info("Weights for local model 2:")
        logging.info(scaled_weights[1][0])

        logging.info("Averaged weights:")
        logging.info(updated_weights[0])
        

    def save(self) -> None:
        with open(
            f'{settings.SAVE_MODEL_PATH}/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-global_model.pickle',
            "wb"
        ) as fp:
            pickle.dump(self.model, fp)
