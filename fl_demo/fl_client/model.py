import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard


class MLPModel:
    def __init__(self, num_features: int) -> None:
        self.model = self._build_model(num_features)

    def _build_model(self, num_features: int) -> Sequential:
        """Scaffold the model"""
        return Sequential(
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

    def set_weights(self, weights) -> None:
        """Set model weights"""
        self.model.set_weights(weights)

    def get_weights(self):
        """Get model weights"""
        return self.model.get_weights()

    def compile(
        self,
        optimizer: str = "adam",
        loss: str = "binary_crossentropy",
        metrics: str = "accuracy",
    ) -> None:
        """Compile model using hyperparams"""
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    def fit(
        self,
        x_train,
        y_train,
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.2,
    ) -> object:
        """Train model"""
        history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )

        return history
