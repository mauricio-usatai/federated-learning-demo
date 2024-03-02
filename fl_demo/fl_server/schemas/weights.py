from pydantic import BaseModel


class ModelWeightsRequest(BaseModel):
    weights: str
    training_test_size: int
