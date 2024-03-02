from pydantic import BaseModel


class Configuration(BaseModel):
    ap_high_max: int
    ap_high_min: int
    ap_low_max: int
    ap_low_min: int
