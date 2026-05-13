"""
JH Jan 2025: These internal types are used within the open_pacific_algo
for populating datastreams.

NOTE: Most types inherit from `TimeStampedData`, which includes a timestamp field.
Since we use or plan to use timestamp information during processing, tracking
timestamps for data in a time series is crucial, hence why most internal types
explicitly track their stamps.
"""

# Python libraries
from itertools import chain
from dataclasses import dataclass
from typing import Any, Tuple, Union
from enum import Enum
from copy import copy

# Third Party
import numpy as np
from pyquaternion import Quaternion
import gtsam

# In House
from open_pacific.utils.coordinate_utils import ecef_to_lla, lla_to_ecef, utm_to_lla, lla_to_utm
from open_pacific.math.geom import get_dual
from open_pacific.logger import log_info_message, log_error_message, log_warning_message
from open_pacific.utils.coordinate_utils import CoordinateFrame


@dataclass
class MessageData:
    topic: str
    msg_type: Any
    timestamp: int
    raw_data: Any
    msg: Any


@dataclass(slots=True)
class TimeStampedData:
    """ Time stamped data uses slots instead of dictionaries to store attributes. This reduces
        the memory burden each instance of a TimeStampedData child but requires some care
        when defining the __copy__() special method.  NOTE all children need also have slots=True
        in their @dataclass decorator
    """
    ts: float

    def __copy__(self):
        """ This copy class functionality works for classes that use slots instead of dictionaries
            for storing and accessing attributes.
        """
        # Get class type
        cls = self.__class__

        # Construct new instance of the class
        new_copy = cls.__new__(cls)

        # Get all __slots__ of the derived class
        slots = chain.from_iterable(getattr(s, "__slots__", []) for s in self.__class__.__mro__)

        # Loop through all slots to set attributes of new instance
        for var in slots:
            setattr(new_copy, var, copy(getattr(self, var)))

        return new_copy


@dataclass(slots=True)
class CovarianceData:
    cov: np.ndarray = None


class MeasurementTypes(Enum):
    """
    Hook types for measurements that are processed by state estimation.
    """
    eLOE      = "LOE"
    eGYE      = "GYE"
    eGPE      = "GPE"
    eGRAV     = "GRAV"
    ePIM      = "PIM"
    eAIRSPEED = "AIRSPEED"
    eBARO     = "BAROMETER"
    eATTITUDE = "ATTITUDE"


class GlobalPosition:
    """ Timestamped datatype that tracks global position.  Can be constructed
        from an ECEF (x,y,z) array, a (utm_x, utm_y) doublet and alt, or
        lat, lon, alt. If altitude is not provided, then it will be set to zero.
        ECEF takes priority over LLA.  LLA takes priority over UTM.

        NOTE 1: if instantiating using utm_x, utm_y, and lat then you also need to
            provide a zone.
        NOTE 2: UTM will be the least accurate due to projection errors.
        NOTE 3: If position is provided, it will be assumed to be an ECEF three vector.
    """
    __slots__ = [
        "ts",
        "ecef",
        "lat",
        "lon",
        "utm_x",
        "utm_y",
        "zone",
        "alt",
        "cov"
    ]
    def __init__(self, ts: float, cov: np.ndarray = None, **kwargs):
        """
        Args:
            ts (float): Timestamp.
            ecef (np.ndarray, optional): ECEF cartesian vector in meters.
            position (np.ndarray, optional): Three vector assumed to be an ECEF cartesian
                vector in meters.
            lat (float, optional): Latitude in degrees
            lon (float, optional): Longitude in degrees
            alt (float, optional): Altitude in meters.  If None, then an altitude
                of zero meters is assumed.
            utm_x (float, optional): Easting UTM distance in meters.
            utm_y (float, optional): Northing UTM distance in meters.
            zone (str, optional): Zone integer and zone letter e.g. "31K" or "2C".
            cov (np.ndarray, optional): 3x3 covariance matrix.  Defaults to None.
        """
        if len(kwargs) == 0:
            raise ValueError("Either ECEF, LLA, or UTM+altitude needed!")
        self.ts = ts
        self.ecef = kwargs.get("ecef")
        self.lat = kwargs.get("lat")
        self.lon = kwargs.get("lon")
        self.utm_x = kwargs.get("utm_x")
        self.utm_y = kwargs.get("utm_y")
        self.zone = kwargs.get("zone")
        self.alt = kwargs.get("alt")
        self.cov = cov

        if self.alt is None and self.ecef is None:
            log_info_message("Altitude not provided.  Setting to zero.")
            self.alt = 0.

        self._calculate_all_representations()

    def _calculate_all_representations(self):
        # checks to see what starting representations are available and
        # build other representations accordingly.  ECEF takes precedent if
        # duplicate inputs are given at construction.
        if self.ecef is not None:
            self._match_construction("ecef")

        elif self.lat is not None and self.lon is not None:
            self._match_construction("lla")

        elif self.utm_x is not None and self.utm_y is not None and self.zone is not None:
            self._match_construction("utm")

        else:
            log_error_message(
                "Invalid GlobalPosition contruction or required inputs missing."
            )

    def _match_construction(self, mode: str):
        """ Matches a construction mode to representation construction order.

        Args:
            mode (str): Constuction mode string.  Must be 'ecef', 'lla', or 'utm'.
        """
        match mode:
            case "ecef":
                self.lat, self.lon, self.alt = ecef_to_lla(self.ecef)
                self.utm_x, self.utm_y, _, self.zone = lla_to_utm(
                    self.lat, self.lon, self.alt
                )

            case "lla":
                self.ecef = lla_to_ecef(self.lat, self.lon, self.alt)
                self.utm_x, self.utm_y, _, self.zone = lla_to_utm(
                    self.lat, self.lon, self.alt
                )

            case "utm":
                self.lat, self.lon, _ = utm_to_lla(
                    self.utm_x, self.utm_y, self.alt, self.zone
                )
                self.ecef = lla_to_ecef(self.lat, self.lon, self.alt)

            case _:
                log_error_message(
                    "Invalid GlobalPosition contruction mode."
                )

    def get_ecef(self) -> np.ndarray:
        return self.ecef

    def get_lla(self) -> Tuple:
        return self.lat, self.lon, self.alt

    def get_utm(self) -> Tuple:
        return self.utm_x, self.utm_y, self.alt

    def get_ned(self):
        """ NOTE: by 'NED' we mean UTM-like coordinates that face North-East-Down
            instead of the usual East-North-Up.
        """
        return self.utm_y, self.utm_x, -self.alt

    def get_zone(self) -> str:
        return self.zone

    def __str__(self):
        return str(self.lat) + str(self.lon)

    def add_ecef_translation(self, delta_ecef: np.ndarray):
        """ Adds a transation to ECEF and recalculates all representations.

        Args:
            delta_ecef (np.ndarray): ECEF translation in meters
        """
        self.ecef += delta_ecef
        self._match_construction("ecef")

    def add_utm_translation(
        self,
        delta_utm_x: float,
        delta_utm_y: float,
        delta_alt: float = 0.
    ):
        """ Calculates a translation in UTM and recalculates all representations.
            NOTE: Currently doesn't work at UTM zone boundaries!

        Args:
            delta_utm_x (float): Easting change in meters
            delta_utm_y (float): Northing change in meters
            delta_alt (float, optional): Altitude change in meters. Defaults to 0.0.
        """
        self.utm_x += delta_utm_x
        self.utm_y += delta_utm_y
        self.alt += delta_alt
        self._match_construction("utm")

    def add_lla_translation(
        self,
        delta_lat: float,
        delta_lon: float,
        delta_alt: float = 0.
    ):
        """ Calculates a translation in lat, lon and recalculates all representations

        Args:
            delta_lat (float): Latitude change in degrees
            delta_lon (float): Longitude change in degrees
            delta_alt (float, optional): Altitude change in meters. Defaults to 0.0.
        """
        new_lat = self.lat + delta_lat
        new_lon = self.lon + delta_lon
        self.alt += delta_alt
        # Need to account for cyclic nature of latitude and longitude.
        if np.abs(new_lat) > 90.:
            self.lat = -np.sign(new_lat) * (180. - np.abs(new_lat))
        else:
            self.lat = new_lat

        if np.abs(new_lon) > 180.:
            self.lon = -np.sign(new_lon) * (360. - np.abs(new_lon))
        else:
            self.lon = new_lon

        self._match_construction("lla")

    def __copy__(self):
        """ This copy class functionality works for classes that use slots instead of dictionaries
            for storing and accessing attributes.
        """
        # Get class type
        cls = self.__class__

        # Construct new instance of the class
        new_copy = cls.__new__(cls)

        # Get all __slots__ of the derived class
        slots = chain.from_iterable(getattr(s, "__slots__", []) for s in self.__class__.__mro__)

        # Loop through all slots to set attributes of new instance
        for var in slots:
            setattr(new_copy, var, copy(getattr(self, var)))

        return new_copy


@dataclass(slots=True)
class Unit3():
    vector: np.ndarray
    cov: np.ndarray = None
    ts: float = None

    def __post_init__(self):
        # Normalize vector.
        self.vector = np.asarray(self.vector, dtype=float)
        self.vector = self.vector / np.linalg.norm(self.vector)

    def basis(self) -> np.ndarray:
        """
        Returns a 3x2 basis matrix for the tangent space at this point.
        """
        return get_dual(self.vector)


@dataclass(slots=True)
class Vec3(TimeStampedData):
    vector: np.ndarray
    cov: np.ndarray = None


@dataclass(slots=True)
class Rot3(TimeStampedData):
    orientation: Quaternion
    cov: np.ndarray = None


@dataclass(slots=True)
class Position(TimeStampedData):
    position: np.ndarray
    cov: np.ndarray = None


@dataclass(slots=True)
class Velocity(TimeStampedData):
    velocity: np.ndarray
    cov: np.ndarray = None


@dataclass(slots=True)
class Twist(TimeStampedData):
    velocity: Velocity
    ω: np.ndarray
    cov: np.ndarray = None


@dataclass(slots=True)
class Transform(TimeStampedData):
    """ Transform between poses """
    t: np.ndarray
    orientation: Quaternion
    cov: np.ndarray = None


@dataclass(slots=True)
class Pose(TimeStampedData):
    """
    General pose between any coordinate frames with no restrictions.
    """
    position: np.ndarray
    orientation: Quaternion
    cov: np.ndarray = None

    def transform_between(self, tgt_pose: Any):
        """
        Find the transformation between the current pose.

        Args:
            tgt_pose (Any): Pose to find the transformation to.
        """
        if not isinstance(tgt_pose, Pose):
            # NOTE: GlobalPose is a subclass of Pose, so it will pass this check.
            log_warning_message("Cannot only find the transform between two poses.")
            return None

        # Find the relative transform between two poses
        tgt_pos = tgt_pose.position
        if isinstance(tgt_pose.position, GlobalPosition):
            tgt_pos = tgt_pos.get_ecef()
        src_pos = self.position
        if isinstance(self.position, GlobalPosition):
            src_pos = src_pos.get_ecef()
        t_world  = tgt_pos - src_pos

        # (World=>FRD1).inv * (World=>FRD2) = (FRD1=>FRD2)
        dR       = self.orientation.inverse * tgt_pose.orientation
        t_local  = self.orientation.inverse.rotate(t_world)  # World => FRD

        ts = max(tgt_pose.ts, self.ts)
        return Transform(ts, t_local, dR)

    def add_local_translation(self, delta_x: np.ndarray):
        """ Add a translation delta_x to position.

        Args:
            delta_x (np.ndarray): Translation vector in meters
        """
        self.position += delta_x

    def get_isometry(self):
        """ Convert pose information to a 3D isometry object (4x4 array)"""
        T = np.eye(4)
        T[:3, :3] = self.orientation.rotation_matrix
        T[:3, 3] = self.position
        return T


@dataclass(slots=True)
class GlobalPose(Pose):
    """
    Global pose, 6DoF relating the transformation from FRD to World
    """
    # NOTE: This overwrites Pose.position, which in Pose is a numpy.ndarray.
    position: GlobalPosition
    cov: np.ndarray = None

    def __post_init__(self):
        """
        We assume that the position is a global position and we want flexibility to convert between
        global position types. We also want to ensure that we have a singular timestamp to resolve
        any ambiguity.
        """
        if not isinstance(self.position, GlobalPosition):
            log_error_message(
                f"Pose requires global position data type for clarity, but its type is {type(self.position)}."
            )
            pass

    def get_isometry(self):
        """ Convert pose information to a 3D isometry object (4x4 array). The translation component is in ECEF."""
        T = np.eye(4)
        T[:3, :3] = self.orientation.rotation_matrix
        T[:3, 3] = self.position.get_ecef()
        return T


@dataclass(slots=True)
class PIM(TimeStampedData):
    # NOTE: this is a special case that wraps a GTSAM internal type. Since we use GTSAM
    # to do preintegration AND back end optimization, it makes sense to explicitly
    # use this type here.
    pim: gtsam.PreintegratedCombinedMeasurements


def rotate_IMU_attributes(
    q: Quaternion,
    imu: Any,
    accel_attribute: str,
    gyro_attribute: str,
    mag_attribute: str
) -> Any:
    """ Helper function that rotates the acceleration and gyro vectors of
        an IMU-like data type.  Only IMU-type data with vector values for accelerometer
        and gyroscope data rotated.

    Args:
        q (Quaternion): Quaternion used to rotate the acceleration and gyro vectors
        imu (Any): IMU-like object.
        accel_attribute (str): Name of the accelerometer vector attribute for the IMU
        gyro_attribute (str): Name of the gyroscope vector attribute for the IMU
        mag_attribute (str): Name of the magnetometer vector attribute for the IMU

    Returns:
        Any: Rotated IMU instance
    """
    # Only try to update attributes if data are vectors
    if isinstance(getattr(imu, accel_attribute), np.ndarray) and isinstance(getattr(imu, gyro_attribute), np.ndarray):
        rotated_accel = q.rotate(getattr(imu, accel_attribute))
        rotated_gyro = q.rotate(getattr(imu, gyro_attribute))
        # TODO: implement magnotometer rotation.
        rotated_mag = np.zeros(3)

        # Manually update attributes
        setattr(imu, accel_attribute, rotated_accel)
        setattr(imu, gyro_attribute, rotated_gyro)
        setattr(imu, mag_attribute, rotated_mag)

    return imu


@dataclass(slots=True)
class IMU(TimeStampedData):
    """
    a: Accelerometer measurement
    ω: Gyroscrope measurement
    mag: Magnetometer measurement
    cov: Optional IMU covariance.  If provided, it should be a 9x9 covariance matrix.
    """
    a: np.ndarray
    ω: np.ndarray
    mag: np.ndarray = np.zeros(3)
    cov: np.ndarray = None

    def rotate(self, q: Quaternion):
        """ Rotate attributes by the provided rotation quaternion.  Values changed in
            place.

        Args:
            q (Quaternion): Rotation quaternion
        """
        return rotate_IMU_attributes(q, self, "a", "ω", "mag")


@dataclass(slots=True)
class IMUBias(TimeStampedData):
    """
    a_bias: Bias for the accelerometer.
    ω_bias: Bias for the gyroscope.
    mag_bias: Bias for the magnetometer.
    """
    a_bias: np.ndarray
    ω_bias: np.ndarray
    mag_bias: np.ndarray = np.zeros(3)
    cov: np.ndarray = None

    def rotate(self, q: Quaternion):
        """ Rotate attributes by the provided rotation quaternion.  Values changed in
            place.

        Args:
            q (Quaternion): Rotation quaternion
        """
        return rotate_IMU_attributes(q, self, "a_bias", "ω_bias", "mag_bias")


@dataclass(slots=True)
class IMUErrorScale(TimeStampedData):
    """
    a_bias: Error scale for the accelerometer.
    ω_bias: Error scale  for the gyroscope.
    mag_bias: Error scale  for the magnetometer.
    """
    a_err: Union[np.ndarray, float]
    ω_err: Union[np.ndarray, float]
    mag_err: np.ndarray = np.zeros(3)

    def rotate(self, q: Quaternion):
        """ Rotate attributes by the provided rotation quaternion.  Values changed in
            place.

        Args:
            q (Quaternion): Rotation quaternion
        """
        return rotate_IMU_attributes(q, self, "a_err", "ω_err", "mag_err")


@dataclass(slots=True)
class PublishRate(TimeStampedData):
    rate: float  # in Hertz


@dataclass(slots=True)
class Heading(TimeStampedData):
    """ Global heading (yaw) """
    yaw: float
    cov: float = None
    frame: CoordinateFrame = None


@dataclass(slots=True)
class Image(TimeStampedData):
    img: np.ndarray


@dataclass(slots=True)
class Barometer(TimeStampedData):
    pressure: float
    temperature: float
    altitude: float
    pressure_cov: float = None
    temperature_cov: float = None
    altitude_cov: float = None
    altitude_bias: float = None


@dataclass(slots=True)
class Airspeed(TimeStampedData):
    """
    airspeed: Airspeed in the body frame in m/s.
    perp_attitude_constraint_std: Used to set noise model for perpendicular attitude
        prior for the airpseed factor.  Has units of rad
    """
    airspeed: float
    cov: float = None
    perp_attitude_constraint_std: float = None


@dataclass(slots=True)
class RCOut(TimeStampedData):
    """Raw servo/motor PWM output channels (e.g. from mavros /rc_out).

    channels: array of PWM values in microseconds (typically 1000–2000 µs),
              one entry per channel.
    """
    channels: np.ndarray


@dataclass(slots=True)
class ActuatorControl(TimeStampedData):
    """Normalized actuator/mixer control outputs (e.g. from /physics/actuators_in).

    controls: float32 array of normalized values, typically 8 channels.
              For a quadcopter group_mix=0: controls[0:4] are per-motor
              throttle commands in [0, 1].
    group_mix: mixer group index (0 = main, 1 = gimbal, etc.)
    """
    controls: np.ndarray
    group_mix: int = 0


@dataclass(slots=True)
class KEF_GPS(TimeStampedData):
    """
    position: The global position
    eph: Horizontal position accuracy (m).
    env: Vertical position accuracy (m).
    velocity: Norm of the velocity vector.
    velocity_north: North component of the NED velocity.
    velocity_east: East component of the NED velocity.
    velocity_down: Down component of the NED velocity.
    cog: Course over ground, direction of movement (NOT HEADING) (-pi, pi).
    satellites_visible: Satellites visible for triangulation.
    """
    position: GlobalPosition
    eph: float
    epv: float
    velocity: float
    velocity_north: float
    velocity_east: float
    velocity_down: float
    cog: float
    satellites_visible: float


@dataclass
class InitState:
    """ Initial condition values.  Used for information transfer between
        bootstrapping/initialization objects.
    """
    grav_body: np.ndarray = np.zeros(3)
    accel_bias: np.ndarray = np.zeros(3)
    gyro_bias: np.ndarray = np.zeros(3)
    accel_error_scale: np.ndarray = np.zeros(3)
    gyro_error_scale: np.ndarray = np.zeros(3)
    accel_95th_percentile: np.ndarray = np.zeros(3)
    gyro_95th_percentile: np.ndarray = np.zeros(3)
    sun_bearing: np.ndarray = np.zeros(3)
    orientation: Quaternion = Quaternion()
    position: GlobalPosition = GlobalPosition(ts=0., ecef=np.zeros(3))
    imu_pub_rate: float = 0

def transform_position_to_target(gp: GlobalPosition, target_frame: CoordinateFrame):
    """
    Transform pose to target frame.

    Args:
        gp (GlobalPosition): Global position.
        target_frame (CoordinateFrame): Target frame.
    """
    if target_frame == CoordinateFrame.NED:
        pose_t = gp.get_ned()
    elif target_frame == CoordinateFrame.ENU:
        pose_t = gp.get_utm()
    elif target_frame == CoordinateFrame.ECEF:
        pose_t = gp.get_ecef()
    return pose_t