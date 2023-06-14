from typing import List
from scripts.envs.params.env_params import EnvParams
import yaml
from scripts.spawnables.robot import Robot
from scripts.spawnables.obstacle import *
from scripts.rewards.reward import Reward
from scripts.rewards.distance import Distance


def parse_config(path: str) -> EnvParams:
    # load yaml config from path
    with open(path, 'r') as file:
        content = yaml.load(file, yaml.SafeLoader)

        # parse required parameters
        engine = content["engine"]
        robots = [_parse_robot(params) for params in _parse_params(content["robots"])]
        obstacles = [_parse_obstacle(params) for params in _parse_params(content["obstacles"])]
        rewards = [_parse_reward(params) for params in _parse_params(content["rewards"])]
        resets = []

        # parse optional parameters
        return EnvParams(engine, robots, obstacles, rewards, resets)

def _parse_params(config: dict) -> List[dict]:
    """
    Per default, the key of the given dictionary is the name of the object.
    Extracts the name and adds it as as key
    """
    
    # save name in parameters
    for name, params in config.items():
        params["name"] = name
    
    # return parameters
    return config.values()

def _parse_robot(params: dict) -> Robot:
    return Robot(**params)

def _parse_obstacle(params: dict) -> Obstacle:
    selector = {
        "Cube" : Cube,
        "Sphere" : Sphere,
        "Cylinder" : Cylinder
    }

    # extract required type
    type = params["type"]

    # make sure parsing of obstacle type is implemented
    if type not in selector:
        raise Exception(f"Obstacle parsing of {type} is not implemented")
    
    # remove type parameter from dict to allow passing params directly to constructor
    params.pop("type")

    # return instance of parsed obstacle
    return selector[type](**params)

def _parse_reward(params: dict) -> Reward:
    selector = {
        "Distance": Distance
    }

    # extract required type
    type = params["type"]

    # make sure parsing of obstacle type is implemented
    if type not in selector:
        raise Exception(f"Reward parsing of {type} is not implemented")
    
    # remove type parameter from dict to allow passing params directly to constructor
    params.pop("type")

    # return instance of parsed reward
    return selector[type](**params)
