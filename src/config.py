import yaml
from typing import List
from pydantic import BaseSettings, BaseModel

class Settings(BaseSettings):
    IP: str
    SOCKETPORT: int
    SIZE: int
    ROUNDS: int
    FIT_CLIENTS: int
    EVAL_CLIENTS: int
    NUM_CLIENTS: int
    SAVE: bool

    class Config:
        env_file: str = "../.env"

class nnConfig(BaseModel):
    name: str
    learning_rates: List[float]
    batch_size: List[int]
    epochs: List[int]
    layers: List[List[int]]
    dpg: bool

class autoConfig(BaseModel):
    name: str
    learning_rates: List[float]
    batch_size: List[int]
    epochs: List[int]
    layers: List[List[int]]

class logisticConfig(BaseModel):
    name: str
    solver : List[str]
    penalty : List[str]
    C : List[float]
    max_iter : List[int]

class hyperConfig(BaseModel):
    nn_config: nnConfig
    auto_config: autoConfig
    lg_config: logisticConfig

def create_config():
    config_path = "../hyperparameters.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    config = hyperConfig(**config_dict)
    return config

config = create_config()  