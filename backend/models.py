
from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict


class TrainApiData(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())
    model_name: str
    hyperparams: dict[str, Any]
    epochs: int
    model_type: str


class PredictApiData(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())
    input_image: Any
    model_name: str

class EvaluateApiData(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())
    model_name: str
    # model_version: Optional[Union[list[int], int]]  # list | int in python 10

class DeleteApiData(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())
    model_name: str
    # model_version: Optional[Union[list[int], int]]  # list | int in python 10
