import carla
import numpy as np
from cv2 import cv2 as cv
from typing import NamedTuple, List, Tuple, Optional

from utils.config import Config
from utils.logger import log_info, log_debug

from utils.carla_birdeye_view import lanes
from utils.carla_birdeye_view.lanes import LaneSide
from utils.carla_birdeye_view.colors import RGB

Mask = np.ndarray  # of shape (y, x), stores 0 and 1, dtype=np.int32
RoadSegmentWaypoints = List[carla.Waypoint]

COLOR_OFF = 0
COLOR_ON = 1


class Coord(NamedTuple):
    x: int
    y: int


class Dimensions(NamedTuple):
    width: int
    height: int


PixelDimensions = Dimensions
Pixels = int
Meters = float
Canvas2D = np.ndarray  # of shape (y, x)

MAP_BOUNDARY_MARGIN: Meters = 300


class MapBoundaries(NamedTuple):
    """Distances in carla.World coordinates"""

    min_x: Meters
    min_y: Meters
    max_x: Meters
    max_y: Meters


class CroppingRect(NamedTuple):
    x: int
    y: int
    width: int
    height: int

    @property
    def vslice(self) -> slice:
        return slice(self.y, self.y + self.height)

    @property
    def hslice(self) -> slice:
        return slice(self.x, self.x + self.width)


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


class RenderingWindow(NamedTuple):
    origin: carla.Location
    area: PixelDimensions


class MapMaskGenerator:
    """Generates 2D, top-down representations of a map.

    Each mask covers area specified by rendering window or whole map (when rendering window is disabled).
    Note that layer, mask, canvas are somewhat interchangeable terms for the same thing.

    Rendering is implemented using OpenCV, so it can be easily adjusted
    to become a regular RGB renderer (just change all `color` arguments to 3-element tuples)
    """

    def __init__(self, client, pixels_per_meter: int) -> None:
        self.client = client
        self.pixels_per_meter = pixels_per_meter
        self.rendering_window: Optional[RenderingWindow] = None

        self._world = client.get_world()
        self._map = self._world.get_map()
        self._topology = self._map.get_topology()
        self._waypoints = self._map.generate_waypoints(2)
        self._map_boundaries = self._find_map_boundaries()
        self._each_road_waypoints = self._generate_road_waypoints()
        self._mask_size: PixelDimensions = self.calculate_mask_size()

        # we must set all carla-related objects to None in order to be able to pickle the entire BridViewProducer class
        # for the sake of parallelizing car intention image generation
        # luckily all of the below are only needed initially to create the road, lanes and centerlines caches
        # and can be safely discarded afterwards
        # self.client = None
        # self._world = None
        # self._map = None
        # self._topology = None
        # self._waypoints = None
        # self._each_road_waypoints = None


    def _find_map_boundaries(self) -> MapBoundaries:
        """Find extreme locations on a map.

        It adds a decent margin because waypoints lie on the road, which means
        that anything that is slightly further than the boundary
        could cause out-of-range exceptions (e.g. pavements, walkers, etc.)
        """
        return MapBoundaries(
            min_x=min(
                self._waypoints, key=lambda x: x.transform.location.x
            ).transform.location.x
            - MAP_BOUNDARY_MARGIN,
            min_y=min(
                self._waypoints, key=lambda x: x.transform.location.y
            ).transform.location.y
            - MAP_BOUNDARY_MARGIN,
            max_x=max(
                self._waypoints, key=lambda x: x.transform.location.x
            ).transform.location.x
            + MAP_BOUNDARY_MARGIN,
            max_y=max(
                self._waypoints, key=lambda x: x.transform.location.y
            ).transform.location.y
            + MAP_BOUNDARY_MARGIN,
        )


    def calculate_mask_size(self) -> PixelDimensions:
        """Convert map boundaries to pixel resolution."""
        width_in_meters = self._map_boundaries.max_x - self._map_boundaries.min_x
        height_in_meters = self._map_boundaries.max_y - self._map_boundaries.min_y
        width_in_pixels = int(width_in_meters * self.pixels_per_meter)
        height_in_pixels = int(height_in_meters * self.pixels_per_meter)
        return PixelDimensions(width=width_in_pixels, height=height_in_pixels)


    def disable_local_rendering_mode(self):
        self.rendering_window = None


    def enable_local_rendering_mode(self, rendering_window: RenderingWindow):
        self.rendering_window = rendering_window


    def location_to_pixel(self, loc: carla.Location) -> Coord:
        """Convert world coordinates to pixel coordinates.

        For example: top leftmost location will be a pixel at (0, 0).
        """
        min_x = self._map_boundaries.min_x
        min_y = self._map_boundaries.min_y

        # Pixel coordinates on full map
        x = int(self.pixels_per_meter * (loc.x - min_x))
        y = int(self.pixels_per_meter * (loc.y - min_y))

        if self.rendering_window is not None:
            # global rendering area coordinates
            origin_x = self.pixels_per_meter * (self.rendering_window.origin.x - min_x)
            origin_y = self.pixels_per_meter * (self.rendering_window.origin.y - min_y)
            topleft_x = int(origin_x - self.rendering_window.area.width / 2)
            topleft_y = int(origin_y - self.rendering_window.area.height / 2)

            # x, y becomes local coordinates within rendering window
            x -= topleft_x
            y -= topleft_y

        return Coord(x=int(x), y=int(y))
    

    def agent_vehicle_trajectory_to_pixel(self, agent_vehicle_trajectory) -> Coord:
        agent_vehicle_trajectory_global_px_pos = []    
        for waypoint_index in range(len(agent_vehicle_trajectory)):
            waypoint_location = carla.Location(x=agent_vehicle_trajectory[waypoint_index][0],
                                               y=agent_vehicle_trajectory[waypoint_index][1],
                                               z=0.0)
            waypoint_location_global_px_pos = self.location_to_pixel(waypoint_location)
            agent_vehicle_trajectory_global_px_pos.append(waypoint_location_global_px_pos)   

        return agent_vehicle_trajectory_global_px_pos


    def make_empty_mask(self) -> Mask:
        if self.rendering_window is None:
            shape = (self._mask_size.height, self._mask_size.width)
        else:
            shape = (
                self.rendering_window.area.height,
                self.rendering_window.area.width,
            )
        return np.zeros(shape, np.uint8)


    def _generate_road_waypoints(self) -> List[RoadSegmentWaypoints]:
        """Return all, precisely located waypoints from the map.

        Topology contains simplified representation (a start and an end
        waypoint for each road segment). By expanding each until another
        road segment is found, we explore all possible waypoints on the map.

        Returns a list of waypoints for each road segment.
        """
        precision: Meters = 0.05
        road_segments_starts: carla.Waypoint = [
            road_start for road_start, road_end in self._topology
        ]

        each_road_waypoints = []
        for road_start_waypoint in road_segments_starts:
            road_waypoints = [road_start_waypoint]

            # Generate as long as it's the same road
            next_waypoints = road_start_waypoint.next(precision)

            if len(next_waypoints) > 0:
                # Always take first (may be at intersection)
                next_waypoint = next_waypoints[0]
                while next_waypoint.road_id == road_start_waypoint.road_id:
                    road_waypoints.append(next_waypoint)
                    next_waypoint = next_waypoint.next(precision)

                    if len(next_waypoint) > 0:
                        next_waypoint = next_waypoint[0]
                    else:
                        # Reached the end of road segment
                        break
            each_road_waypoints.append(road_waypoints)
        return each_road_waypoints


    def road_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        # FIXME Refactor that crap
        print(self._each_road_waypoints)
        for road_waypoints in self._each_road_waypoints:
            road_left_side = [
                lateral_shift(w.transform, -w.lane_width * 0.5) for w in road_waypoints
            ]
            road_right_side = [
                lateral_shift(w.transform, w.lane_width * 0.5) for w in road_waypoints
            ]

            polygon = road_left_side + [x for x in reversed(road_right_side)]
            polygon = [self.location_to_pixel(x) for x in polygon]
            if len(polygon) > 2:
                polygon = np.array([polygon], dtype=np.int32)
                # FIXME Hard to notice the difference without polylines
                cv.polylines(
                    img=canvas, pts=polygon, isClosed=True, color=COLOR_ON, thickness=5
                )
                cv.fillPoly(img=canvas, pts=polygon, color=COLOR_ON)
        return canvas


    def lanes_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        for road_waypoints in self._each_road_waypoints:
            # if not road_waypoints[0].is_junction:
            # NOTE This block was inside if statement - some junctions may not have proper lane markings drawn
            # Left Side
            lanes.draw_lane_marking_single_side(
                canvas,
                road_waypoints,
                side=LaneSide.LEFT,
                location_to_pixel_func=self.location_to_pixel,
                color=COLOR_ON,
            )

            # Right Side
            lanes.draw_lane_marking_single_side(
                canvas,
                road_waypoints,
                side=LaneSide.RIGHT,
                location_to_pixel_func=self.location_to_pixel,
                color=COLOR_ON,
            )
        return canvas


    def centerlines_mask(self) -> Mask:
        canvas = self.make_empty_mask()
        for road_waypoints in self._each_road_waypoints:
            polygon = [self.location_to_pixel(wp.transform.location) for wp in road_waypoints]
            if len(polygon) > 2:
                polygon = np.array([polygon], dtype=np.int32)
                cv.polylines(
                    img=canvas, pts=polygon, isClosed=False, color=COLOR_ON, thickness=1
                )
        return canvas
    

    def agent_vehicle_trajectory_mask(self, agent_vehicle_trajectory) -> Mask:
        canvas = self.make_empty_mask()
        polygon = self.agent_vehicle_trajectory_to_pixel(agent_vehicle_trajectory)
        if len(polygon) > 2:
            polygon = np.array([polygon], dtype=np.int32)
            cv.polylines(img=canvas, pts=polygon, isClosed=False, color=COLOR_ON, thickness=1)
        return canvas


    # explicitly passing transform allows to manually manipulate location and rotation
    def agent_vehicle_mask(self, agent_vehicle_transform) -> Mask:
        agent_vehicle_canvas = self.make_empty_mask()

        agent_vehicle_x_expansion = Config.Carla.EGO_VEHICLE_LENGTH / 2
        agent_vehicle_y_expansion = Config.Carla.EGO_VEHICLE_WIDTH / 2

        agent_vehicle_corners = [
            # bottom-left
            carla.Location(x=-agent_vehicle_x_expansion, y=-agent_vehicle_y_expansion),
            # top-left
            carla.Location(x=agent_vehicle_x_expansion, y=-agent_vehicle_y_expansion),
            # top-right
            carla.Location(x=agent_vehicle_x_expansion, y=agent_vehicle_y_expansion),
            # bottom-right
            carla.Location(x=-agent_vehicle_x_expansion, y=agent_vehicle_y_expansion),
        ]

        agent_vehicle_transform.transform(agent_vehicle_corners)  
        agent_vehicle_corners_pxl_loc = [self.location_to_pixel(loc) for loc in agent_vehicle_corners]
        cv.fillPoly(img=agent_vehicle_canvas, pts=np.int32([agent_vehicle_corners_pxl_loc]), color=COLOR_ON)

        # for debugging (draw corners of agent vehicle)
        ''' 
        log_debug(
                f"\nagent vehicle corner positions:\n"
                f"\t- top-left [brown] & top-right [gold]: ({agent_vehicle_corners[1].x:.2f}, {agent_vehicle_corners[1].y:.2f}) & "
                f"({agent_vehicle_corners[2].x:.2f}, {agent_vehicle_corners[2].y:.2f})\n"
                f"\t- bottom-left [cyan] & bottom-right [violett]: ({agent_vehicle_corners[0].x:.2f}, {agent_vehicle_corners[0].y:.2f}) & "
                f"({agent_vehicle_corners[3].x:.2f}, {agent_vehicle_corners[3].y:.2f})\n"
            )

        # colorize each corner of the agent vehicle differently in order to be able to distinguish them
        corner_x_expansion = Config.Carla.EXO_PEDESTRIAN_LENGTH / 2
        corner_y_expansion = Config.Carla.EXO_PEDESTRIAN_WIDTH / 2

        expanded_corner_canvases = []
        for corner in agent_vehicle_corners:
            expanded_corner_canvas = self.make_empty_mask()

            expanded_corner = [
                # bottom-left
                carla.Location(x=corner.x-corner_x_expansion, y=corner.y-corner_y_expansion),
                # top-left
                carla.Location(x=corner.x+corner_x_expansion, y=corner.y-corner_y_expansion),
                # top-right
                carla.Location(x=corner.x+corner_x_expansion, y=corner.y+corner_y_expansion),
                # bottom-right
                carla.Location(x=corner.x-corner_x_expansion, y=corner.y+corner_y_expansion),
            ]

            expanded_corner = [self.location_to_pixel(loc) for loc in expanded_corner]
            # turn this on fpr debugging
            cv.fillPoly(img=expanded_corner_canvas, pts=np.int32([expanded_corner]), color=COLOR_OFF)
            expanded_corner_canvases.append(expanded_corner_canvas)
        '''
        expanded_corner_canvases = None
        return agent_vehicle_canvas, expanded_corner_canvases
    

    def vehicles_mask(self, exo_agent_vehicle_transforms: List[carla.Transform]) -> Mask:
        canvas = self.make_empty_mask()
        for exo_agent_vehicle_transform in exo_agent_vehicle_transforms:

            '''
            log_info(f"exo_agent_vehicle_transform.location.x: {exo_agent_vehicle_transform.location.x:.2f}, "
                     f"exo_agent_vehicle_transform.location.y: {exo_agent_vehicle_transform.location.y:.2f}, "
                     f"exo_agent_vehicle_transform.location.z: {exo_agent_vehicle_transform.location.z:.2f}")
            '''
            exo_agent_vehicle_x_expansion = Config.Carla.EGO_VEHICLE_LENGTH / 2
            exo_agent_vehicle_y_expansion = Config.Carla.EGO_VEHICLE_WIDTH / 2

            exo_agent_vehicle_corners = [
                # bottom-left
                carla.Location(x=-exo_agent_vehicle_x_expansion, y=-exo_agent_vehicle_y_expansion),
                # top-left
                carla.Location(x=exo_agent_vehicle_x_expansion, y=-exo_agent_vehicle_y_expansion),
                # top-right
                carla.Location(x=exo_agent_vehicle_x_expansion, y=exo_agent_vehicle_y_expansion),
                # bottom-right
                carla.Location(x=-exo_agent_vehicle_x_expansion, y=exo_agent_vehicle_y_expansion),
            ]

            exo_agent_vehicle_transform.transform(exo_agent_vehicle_corners)
            exo_agent_vehicle_corners_pxl_loc = [self.location_to_pixel(loc) for loc in exo_agent_vehicle_corners]
            cv.fillPoly(img=canvas, pts=np.int32([exo_agent_vehicle_corners_pxl_loc]), color=COLOR_ON)
        return canvas
    
    
    def pedestrians_mask(self, pedestrian_transform) -> Mask:
        if pedestrian_transform is None: return
        canvas = self.make_empty_mask()

        bb_x = Config.Carla.EXO_PEDESTRIAN_LENGTH / 2
        bb_y = Config.Carla.EXO_PEDESTRIAN_WIDTH / 2

        corners = [
            carla.Location(x=-bb_x, y=-bb_y),
            carla.Location(x=bb_x, y=-bb_y),
            carla.Location(x=bb_x, y=bb_y),
            carla.Location(x=-bb_x, y=bb_y),
        ]

        pedestrian_transform.transform(corners)
        corners = [self.location_to_pixel(loc) for loc in corners]
        cv.fillPoly(img=canvas, pts=np.int32([corners]), color=COLOR_ON)
        return canvas


    def traffic_lights_masks(self, traffic_lights: List[carla.Actor]) -> Tuple[Mask]:
        red_light_canvas = self.make_empty_mask()
        yellow_light_canvas = self.make_empty_mask()
        green_light_canvas = self.make_empty_mask()
        tls = carla.TrafficLightState
        for tl in traffic_lights:
            world_pos = tl.get_location()
            pos = self.location_to_pixel(world_pos)
            radius = int(self.pixels_per_meter * 1.2)
            if tl.state == tls.Red:
                target_canvas = red_light_canvas
            elif tl.state == tls.Yellow:
                target_canvas = yellow_light_canvas
            elif tl.state == tls.Green:
                target_canvas = green_light_canvas
            else:
                # Unknown or off traffic light
                continue

            cv.circle(
                img=target_canvas,
                center=pos,
                radius=radius,
                color=COLOR_ON,
                thickness=cv.FILLED,
            )
        return red_light_canvas, yellow_light_canvas, green_light_canvas
