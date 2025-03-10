import carla
import numpy as np
import os

from enum import IntEnum, auto, Enum
from pathlib import Path
from typing import List
from filelock import FileLock

import cv2.cv2 as cv2

from utils.config import Config
from utils.logger  import log_info, log_debug

from utils.carla_birdeye_view import actors, cache
from utils.carla_birdeye_view.colors import RGB
from utils.carla_birdeye_view.mask import (
    PixelDimensions,
    Coord,
    CroppingRect,
    MapMaskGenerator,
    Mask,
    COLOR_ON,
    RenderingWindow,
    Dimensions,
    MAP_BOUNDARY_MARGIN
)


__all__ = ["BirdViewProducer", "DEFAULT_HEIGHT", "DEFAULT_WIDTH"]

DEFAULT_HEIGHT = 336  # its 84m when density is 4px/m
DEFAULT_WIDTH = 150  # its 37.5m when density is 4px/m

BirdView = np.ndarray  # [np.uint8] with shape (level, y, x)
RgbCanvas = np.ndarray  # [np.uint8] with shape (y, x, 3)


class BirdViewCropType(Enum):
    FRONT_AND_REAR_AREA = auto()  # Freeway mode
    FRONT_AREA_ONLY = auto()  # Like in "Learning by Cheating"


class BirdViewMasks(IntEnum):
    TOP_LEFT = 10
    TOP_RIGHT = 9
    BOTTOM_RIGHT = 8
    BOTTOM_LEFT = 7
    AGENT = 6
    PAST_AGENT_VEHICLE_TRAJECTORY = 5
    FUTURE_AGENT_VEHICLE_TRAJECTORY = 4
    VEHICLES = 3
    PEDESTRIANS = 2
    LANES = 1
    ROAD = 0

    @staticmethod
    def top_to_bottom() -> List[int]:
        return list(BirdViewMasks)

    @staticmethod
    def bottom_to_top() -> List[int]:
        return list(reversed(BirdViewMasks.top_to_bottom()))


RGB_BY_MASK = {
    BirdViewMasks.TOP_LEFT: RGB.BROWN,
    BirdViewMasks.TOP_RIGHT: RGB.GOLD,
    BirdViewMasks.BOTTOM_RIGHT: RGB.VIOLET,
    BirdViewMasks.BOTTOM_LEFT: RGB.CYAN,
    BirdViewMasks.ROAD: RGB.DIM_GRAY,
    BirdViewMasks.LANES: RGB.WHITE,
    BirdViewMasks.PAST_AGENT_VEHICLE_TRAJECTORY: RGB.GREEN,
    BirdViewMasks.FUTURE_AGENT_VEHICLE_TRAJECTORY: RGB.YELLOW,
    BirdViewMasks.PEDESTRIANS: RGB.RED,
    BirdViewMasks.VEHICLES: RGB.ORANGE,
    BirdViewMasks.AGENT: RGB.BLUE
}

BIRDVIEW_SHAPE_CHW = (len(RGB_BY_MASK), DEFAULT_HEIGHT, DEFAULT_WIDTH)
BIRDVIEW_SHAPE_HWC = (DEFAULT_HEIGHT, DEFAULT_WIDTH, len(RGB_BY_MASK))


def rotate(image, angle, center=None, scale=1.0):
    assert image.dtype == np.uint8

    """Copy paste of imutils method but with INTER_NEAREST and BORDER_CONSTANT flags"""
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # return the rotated image
    return rotated


def circle_circumscribed_around_rectangle(rect_size: Dimensions) -> float:
    """Returns radius of that circle."""
    a = rect_size.width / 2
    b = rect_size.height / 2
    return float(np.sqrt(np.power(a, 2) + np.power(b, 2)))


def square_fitting_rect_at_any_rotation(rect_size: Dimensions) -> float:
    radius = circle_circumscribed_around_rectangle(rect_size)
    side_length_of_square_circumscribed_around_circle = radius * 2
    return side_length_of_square_circumscribed_around_circle    


class BirdViewProducer:
    """Responsible for producing top-down view on the map, following agent's vehicle.

    About BirdView:
    - top-down view, fixed directly above the agent (including vehicle rotation), cropped to desired size
    - consists of stacked layers (masks), each filled with ones and zeros (depends on MaskMaskGenerator implementation).
        Example layers: road, vehicles, pedestrians. 0 indicates -> no presence in that pixel, 1 -> presence
    - convertible to RGB image
    - Rendering full road and lanes masks is computationally expensive, hence caching mechanism is used
    """

    def __init__(
        self,
        client: carla.Client,
        target_size: PixelDimensions,
        pixels_per_meter: int,
        crop_type: BirdViewCropType,
        all_parked_vehicle_transforms: dict
    ) -> None:
        self.client = client
        self.target_size = target_size
        self._pixels_per_meter = pixels_per_meter
        self._crop_type = crop_type

        if crop_type is BirdViewCropType.FRONT_AND_REAR_AREA:
            rendering_square_size = round(square_fitting_rect_at_any_rotation(self.target_size))
        elif crop_type is BirdViewCropType.FRONT_AREA_ONLY:
            # We must keep rendering size from FRONT_AND_REAR_AREA (in order to avoid rotation issues)
            enlarged_size = PixelDimensions(width=target_size.width, height=target_size.height*2)
            rendering_square_size = round(square_fitting_rect_at_any_rotation(enlarged_size))
        else:
            raise NotImplementedError
        
        self.rendering_area = PixelDimensions(
            width=rendering_square_size, height=rendering_square_size
        )
        self._world = client.get_world()
        self._map = self._world.get_map()
        self.masks_generator = MapMaskGenerator(
            client, pixels_per_meter=pixels_per_meter
        )

        cache_path = self.parametrized_cache_path()
        if Path(cache_path).is_file():
            log_debug(f"loading map cache for carla-birdeye view car intention generation from {cache_path}...")
            with FileLock(f"{cache_path}.lock"):
                static_cache = np.load(cache_path)
                self.full_road_cache = static_cache[0]
                self.full_lanes_cache = static_cache[1]
                self.full_centerlines_cache = static_cache[2]
        else:
            log_debug(f"saving map cache for carla-birdeye view car intention generation at {cache_path}...")
            self.full_road_cache = self.masks_generator.road_mask()
            self.full_lanes_cache = self.masks_generator.lanes_mask()
            self.full_centerlines_cache = self.masks_generator.centerlines_mask()
            static_cache = np.stack([self.full_road_cache, self.full_lanes_cache, self.full_centerlines_cache])
            with FileLock(f"{cache_path}.lock"):
                np.save(cache_path, static_cache, allow_pickle=False)

        # pre-compute and save the masks of all parked vehicles across as scnearios
        # so we can avoid redundant computational overhead when generating car intention images
        # also: this is required for parallelization through worker processes
        self.scenario_to_parked_vehicle_mask = {}
        for scenario, parked_vehicle_transforms in all_parked_vehicle_transforms.items():
            scenario_parked_vehicle_mask = self.masks_generator.vehicles_mask(parked_vehicle_transforms)
            self.scenario_to_parked_vehicle_mask.update({scenario:scenario_parked_vehicle_mask})

        # we must set all carla-related objects to None in order to be able to pickle the entire BridViewProducer class
        # for the sake of parallelizing car intention image generation
        # luckily all of the below are only needed initially to create the road, lanes and centerlines caches
        # and can be safely discarded afterwards
        self.client = None
        self._world = None
        self._map = None

        

    @staticmethod
    def as_rgb(birdview: BirdView) -> RgbCanvas:
        _, h, w = birdview.shape
        rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        for mask_type in BirdViewMasks.bottom_to_top():
            rgb_color = RGB_BY_MASK[mask_type]
            mask = birdview[mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            rgb_canvas[nonzero_indices(mask)] = rgb_color
        return rgb_canvas
    

    def parametrized_cache_path(self) -> str:
        cache_dir = Path(f"{os.getcwd()}/utils/carla_birdeye_view/birdview_v2_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        opendrive_content_hash = cache.generate_opendrive_content_hash(self._map)
        cache_filename = (
            f"{str(self._map.name).replace('/', '_')}__"
            f"px_per_meter={self._pixels_per_meter}__"
            f"opendrive_hash={opendrive_content_hash}__"
            f"margin={MAP_BOUNDARY_MARGIN}.npy"
        )
        return str(cache_dir / cache_filename)
    

    def remove_already_traversed_waypoints(self, agent_vehicle_loc: carla.Location, agent_vehicle_future_trajectory: list):
        # find indeces of future waypoints that have already been traversed during IS-DESPOT's search
        loc_indeces = []
        for idx, (future_x, future_y) in enumerate(agent_vehicle_future_trajectory):
            # if waypoint is within a small distance to agent vehicle's current position
            if np.linalg.norm([agent_vehicle_loc.x - future_x, agent_vehicle_loc.y - future_y]) < 1e-3:
                loc_indeces.append(idx)
        # remove all waypoints that come before and including this point
        return agent_vehicle_future_trajectory[loc_indeces[-1] + 1:]


    def sparsify_trajectory(self, agent_vehicle_trajectory: list):
        # nothing to sparsify
        if agent_vehicle_trajectory is None: return
        if len(agent_vehicle_trajectory) <= 2: return agent_vehicle_trajectory

        # x and y values of last waypoint that is being kept
        (previous_x, previous_y) = agent_vehicle_trajectory[0]
        # first waypoint is always kept
        waypoint_indeces = [0]
        # sparsify trajectories
        for idx, (current_x, current_y) in enumerate(agent_vehicle_trajectory):
            # if agent vehicle has travelled a sufficiently big distance
            if np.linalg.norm([previous_x - current_x, previous_y - current_y]) >= 1.5:
                waypoint_indeces.append(idx)
                previous_x = current_x
                previous_y = current_y

        # last waypoint is always kept as well
        waypoint_indeces.append(-1)
        sparse_agent_vehicle_trajectory = [agent_vehicle_trajectory[waypoint_index] for waypoint_index in waypoint_indeces]
        return sparse_agent_vehicle_trajectory

    # produce for simulated world states
    def produce(
        self,
        agent_vehicle_transform: carla.Transform,
        pedestrian_transform: carla.Transform,
        agent_vehicle_past_trajectory: list,
        agent_vehicle_future_trajectory: list,
        scenario_id: int
    ) -> BirdView:
        
        agent_vehicle_loc = agent_vehicle_transform.location

        agent_vehicle_future_trajectory = self.remove_already_traversed_waypoints(
            agent_vehicle_loc,
            agent_vehicle_future_trajectory
        )

        # calculate and insert origin of future trajectory as top center point of agent vehicle
        radians = agent_vehicle_transform.rotation.yaw * np.pi / 180
        center_top_x = agent_vehicle_loc.x + ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        center_top_y = agent_vehicle_loc.y + ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))
        agent_vehicle_future_trajectory.insert(0, (center_top_x, center_top_y))

        # infer center bottom point of ego vehicle (point of connection for past trajectory line)
        center_bottom_x = agent_vehicle_loc.x - ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        center_bottom_y = agent_vehicle_loc.y - ((Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))
        agent_vehicle_past_trajectory.append((center_bottom_x, center_bottom_y))

        agent_vehicle_past_trajectory = self.sparsify_trajectory(agent_vehicle_past_trajectory)
        agent_vehicle_future_trajectory = self.sparsify_trajectory(agent_vehicle_future_trajectory)

        # Reusing already generated static masks for whole map
        self.masks_generator.disable_local_rendering_mode()
        agent_global_px_pos = self.masks_generator.location_to_pixel(agent_vehicle_loc)

        # specified PixelDimension centered around the agent vehicle
        cropping_rect = CroppingRect(
            x=int(agent_global_px_pos.x - self.rendering_area.width / 2),
            y=int(agent_global_px_pos.y - self.rendering_area.height / 2),
            width=self.rendering_area.width,
            height=self.rendering_area.height,
        )

        masks = np.zeros(
            shape=(
                len(BirdViewMasks),
                self.rendering_area.height,
                self.rendering_area.width,
            ),
            dtype=np.uint8,
        )

        masks[BirdViewMasks.ROAD.value] = self.full_road_cache[
            cropping_rect.vslice, cropping_rect.hslice
        ]
        masks[BirdViewMasks.LANES.value] = self.full_lanes_cache[
            cropping_rect.vslice, cropping_rect.hslice
        ]

        # create mask for past trajectory
        masks[BirdViewMasks.PAST_AGENT_VEHICLE_TRAJECTORY.value] = \
            self.masks_generator.agent_vehicle_trajectory_mask(agent_vehicle_past_trajectory)[
                cropping_rect.vslice, cropping_rect.hslice
        ]
        # create mask for future trajectory
        masks[BirdViewMasks.FUTURE_AGENT_VEHICLE_TRAJECTORY.value] = \
            self.masks_generator.agent_vehicle_trajectory_mask(agent_vehicle_future_trajectory)[
                cropping_rect.vslice, cropping_rect.hslice
        ]
    
        # parked cars (inferred during __init__)
        parked_vehicle_mask = self.scenario_to_parked_vehicle_mask.get(scenario_id)
        if parked_vehicle_mask is not None:
            masks[BirdViewMasks.VEHICLES.value] = parked_vehicle_mask[
            cropping_rect.vslice, cropping_rect.hslice
            ]

        # dynamic masks
        rendering_window = RenderingWindow(
            origin=agent_vehicle_loc, area=self.rendering_area
        )
        self.masks_generator.enable_local_rendering_mode(
            rendering_window
        )

        masks = self._render_actors_masks(
            agent_vehicle_transform, 
            pedestrian_transform, 
            masks
        )

        cropped_masks = self.apply_agent_following_transformation_to_masks(
            agent_vehicle_transform, masks, scenario_id
        )

        ordered_indices = [mask.value for mask in BirdViewMasks.bottom_to_top()]
        return cropped_masks[ordered_indices]
    

    def _render_actors_masks(
        self,
        agent_vehicle_transform,
        pedestrian_transform,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Fill masks with ones and zeros (more precisely called as "bitmask").
        Although numpy dtype is still the same, additional semantic meaning is being added.
        """

        agent_vehicle_mask, _ = self.masks_generator.agent_vehicle_mask(agent_vehicle_transform)

        # colorize corner of agent vehicle to infer correct borders (for debugging)
        '''
        masks[BirdViewMasks.BOTTOM_LEFT] = agent_vehicle_corners_masks[0]
        masks[BirdViewMasks.TOP_LEFT] = agent_vehicle_corners_masks[1]
        masks[BirdViewMasks.TOP_RIGHT] = agent_vehicle_corners_masks[2]
        masks[BirdViewMasks.BOTTOM_RIGHT] = agent_vehicle_corners_masks[3]
        '''
        masks[BirdViewMasks.AGENT.value] = agent_vehicle_mask

        masks[BirdViewMasks.PEDESTRIANS.value] = self.masks_generator.pedestrians_mask(
            pedestrian_transform
        )
        return masks
    

    def apply_agent_following_transformation_to_masks(
        self, agent_vehicle_transform, masks: np.ndarray, scenario_id: int
    ) -> np.ndarray:
        angle = (
            agent_vehicle_transform.rotation.yaw + 90 # vehicle's front will point to the top
        )  

        # Rotating around the center
        crop_with_car_in_the_center = masks
        masks_n, h, w = crop_with_car_in_the_center.shape
        rotation_center = Coord(x=w // 2, y=h // 2)

        # warpAffine from OpenCV requires the first two dimensions to be in order: height, width, channels
        crop_with_centered_car = np.transpose(
            crop_with_car_in_the_center, axes=(1, 2, 0)
        )
        
        # costly and not really necessary
        if scenario_id in [6, 9, 11]:
            rotated = rotate(crop_with_centered_car, angle, center=rotation_center)
            rotated = np.transpose(rotated, axes=(2, 0, 1))
        else:
            rotated = np.transpose(crop_with_centered_car, axes=(2, 0, 1))

        half_width = self.target_size.width // 2
        hslice = slice(rotation_center.x - half_width, rotation_center.x + half_width)

        if self._crop_type is BirdViewCropType.FRONT_AREA_ONLY:
            vslice = slice(rotation_center.y - self.target_size.height, rotation_center.y)
        elif self._crop_type is BirdViewCropType.FRONT_AND_REAR_AREA:
            half_height = self.target_size.height // 2
            vslice = slice(
                rotation_center.y - half_height, rotation_center.y + half_height
            )
        else:
            raise NotImplementedError
        assert (
            vslice.start > 0 and hslice.start > 0
        ), "Trying to access negative indexes is not allowed, check for calculation errors!"
        car_on_the_bottom = rotated[:, vslice, hslice]
        return car_on_the_bottom