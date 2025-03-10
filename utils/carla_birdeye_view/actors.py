import carla
from typing import NamedTuple, List
from utils.config import Config
from utils.logger import log_info

is_vehicle = lambda actor: "vehicle" in actor.type_id
is_pedestrian = lambda actor: "walker" in actor.type_id
is_traffic_light = lambda actor: "traffic_light" in actor.type_id


class SegregatedActors(NamedTuple):
    vehicles: List[carla.Actor]
    pedestrians: List[carla.Actor]
    traffic_lights: List[carla.Actor]


def segregate_by_type(actors: List[carla.Actor]) -> SegregatedActors:
    vehicles = []
    pedestrians = []
    traffic_lights = []
    for actor in actors:
        if is_vehicle(actor):
            vehicles.append(actor)
        elif is_pedestrian(actor):
            pedestrians.append(actor)
        elif is_traffic_light(actor):
            traffic_lights.append(actor)
    return SegregatedActors(vehicles, pedestrians, traffic_lights)


def query_all(world: carla.World) -> List[carla.Actor]:
    snapshot: carla.WorldSnapshot = world.get_snapshot()
    all_actors = []
    for actor_snapshot in snapshot:
        actor = world.get_actor(actor_snapshot.id)
        if actor is not None:
            all_actors.append(actor)
    return all_actors


def get_parked_cars(world: carla.World) -> List[carla.Actor]:
    snapshot: carla.WorldSnapshot = world.get_snapshot()
    parked_cars = []
    for actor_snapshot in snapshot:
        actor = world.get_actor(actor_snapshot.id)
        if actor is not None:
            if is_vehicle(actor):
                if actor.attributes["role_name"] is Config.Carla.PARKED_VEHICLE_ROLENAME:
                    parked_cars.append(actor)
    return parked_cars