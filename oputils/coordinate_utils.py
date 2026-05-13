from string import ascii_uppercase
import math
from enum import Enum

# Third Party
import numpy as np
from pyquaternion import Quaternion
import pyproj
import utm

# In House
from open_pacific.logger import log_error_message, log_info_message


# CONSTANTS:
"""
NOTE: these might seem like they should be passive rotations since they are transforms
between coordinate frames.  Quaternion.rotate() is an active rotation, though, so
these quaternions all correspond to an active rotation from one frame to another.
For ENU->NED and the inverse, this is trivial since this transform is its own inverse.
"""
# Rotation quaternion from ENU to NED frames.
Q_ENU_TO_NED = Quaternion(w=0., x=np.sqrt(2.) / 2., y=np.sqrt(2.) / 2., z=0.)
# Rotation quaternion from NED to ENU frames.
Q_NED_TO_ENU = Quaternion(w=0., x=np.sqrt(2.) / 2., y=np.sqrt(2.) / 2., z=0.)

# NOTE: this dictionary maps zone letter to true or false, where true meaning the zone
# represents a location in the southern hemisphere.
CHAR_TO_HEMISPHERE = {char: ord(char) < ord("N") for char in ascii_uppercase}

# NOTE: these are used to construct pyproj transformation instances
ECEF = {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"}
LLA = {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"}

# NOTE: if we change to these standard coordinate systems, then we need to 1) swap lat and lon
# in ecef_to_lla() and lla_to_ecef() and 2) maybe make a similar change for the utm transforms.
# We will also need to update unit tests for these transform functions because lat and lon will
# be switched!
# LLA = pyproj.CRS("EPSG:9707")
# ECEF = pyproj.CRS("EPSG:4978")
LLA_TO_ECEF = pyproj.Transformer.from_crs(LLA, ECEF)

# NOTE: Singleton to track what UTM-to-LLA transforms have been instantiated.
# By caching them here we prevent expensive reinstantiation calls.
UTM_TO_LLA_DICT = {}

# NOTE: Constants of Earth's rotation
GM = 3.986004418e14   # gravitational constant × Earth mass (m³/s²)
Ω = 7.2921151467e-5  # Earth's angular velocity (rad/s)

# WGS84 Constants
WGS84_A = 6378137.0               # Semi-major axis (meters)
WGS84_B = 6356752.314245          # Semi-minor axis (meters)
WGS84_F = 1.0 / 298.257223563     # Flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F**2  # First eccentricity squared
WGS84_E_PRIME2 = WGS84_E2 / (1 - WGS84_E2)  # Second eccentricity squared


class CoordinateFrame(Enum):
    """
    Enum to describe the coordinate frames we can work with in the Open Pacific code base.
    I'd recommend adding to this list and take this as an argument when do any conversions too.
    """
    NED  = "ned"   # NED (North-East-Down). local tangent plane.
    ENU  = "enu"   # ENU (East-North-Up).
    ECEF = "ecef"  # Earth-centered earth fixed.
    BODY = "body"  # Body frame FRD (front-right-down).


def get_zone_string(lat: float, lon: float) -> str:
    # Helper function to return UTM zone string form lat, lon.
    if lat is not None and lon is not None:
        _, _, zone_int, zone_letter = utm.from_latlon(lat, lon)
        return f"{zone_int}{zone_letter}"
    else:
        log_info_message("Lat or Lon are None.  No zone string returned.")
        return None


def get_UTM_dict(zone: str) -> dict:
    """ Constructs UTM transformation dictionary for a provided zone string

    Args:
        zone (str): Zone string "<zone_int><zone_letter>"

    Returns:
        dict: UTM transform dictionary.
    """
    zone_int = int(zone[:-1])
    zone_letter = zone[-1]
    return {
        "proj": "utm",
        "zone": zone_int,
        "ellps": "WGS84",
        "south": CHAR_TO_HEMISPHERE[zone_letter]
    }


def get_utm_to_lla_transform(
    lat: float = None,
    lon: float = None,
    zone_str: str = None
):
    """
    Get UTM to LLA transform from lat lon.
    Either lat+lon or the zone_str is needed.

    Args:
        lat (float): Latitude (degrees).
        lon (float): Longitude (degrees).
        zone_str(str): The zone of interest.

    Returns:
        pyproj.Transform: Transform from UTM => LLA
    """
    if not zone_str:
        if lat is None or lon is None:
            log_error_message("Lat and Lon are None with no zone string!")
        zone_str = get_zone_string(lat, lon)
    utm_to_lla = UTM_TO_LLA_DICT.get(zone_str)
    if not utm_to_lla:
        utm_dict = get_UTM_dict(zone_str)
        utm_to_lla = pyproj.Transformer.from_crs(utm_dict, LLA)
        UTM_TO_LLA_DICT[zone_str] = utm_to_lla
    return utm_to_lla, zone_str


def normalize_yaw(yaw):
    """
    Normalizes yaw angle to stay within [-π, π] range.

    Args:
        yaw (float): Angle in radians

    Returns:
        float: Normalized angle in radians between -π and π
    """
    # First wrap to [-2π, 2π] using modulo
    normalized = yaw % (2 * math.pi)

    # If angle is greater than π, subtract 2π to get into [-π, π] range
    if normalized > math.pi:
        normalized -= 2 * math.pi
    # If angle is less than -π, add 2π to get into [-π, π] range
    elif normalized < -math.pi:
        normalized += 2 * math.pi

    return normalized


def ecef_to_lla(
    ecef: np.ndarray
) -> tuple:
    """ convenience wrapper for calculating lat, lon, alt from ecef (x,y,z)

    Args:
        ecef (np.array): (x,y,z) ECEF cartesian triplet.

    Returns:
        tuple: Latitude, longitude, altitude
    """
    x, y, z = ecef
    lon, lat, alt = LLA_TO_ECEF.transform(x, y, z, radians=False, direction="INVERSE")
    return lat, lon, alt


def lla_to_ecef(
    lat: float,
    lon: float,
    alt: float
) -> np.ndarray:
    """ Convenience function for calculating ecef (x,y,z) from lat, lon, alt

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.
        alt (float): Altitude in meters.

    Returns:
        np.ndarray: (x,y,z) ECEF vector array in meters
    """
    return np.asarray(
        LLA_TO_ECEF.transform(lon, lat, alt, radians=False)
    )


def lla_to_utm(
    lat: float,
    lon: float,
    alt: float,
) -> tuple:
    """
    Convert lat, lon, alt to UTM (x,y), alt.  Altitude is copied over.
    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees
        alt (float): Altitude in meters

    Returns:
        tuple: (x,y) UTM position and alt in meters, and zone string.
    """
    if lat is None or lon is None:
        # log_warning_message(
        #     "Latitude or longitude is None!"
        # )
        return None, None, alt, None
    elif lat > 84. or lat < -80.:
        # log_warning_message(
        #     f"Latitude {lat} must be between -80 and +84 for UTM conversion."
        # )
        return None, None, alt, None

    # Use zone integer and letter string to build UTM transformation.
    tr_to_utm, zone_str = get_utm_to_lla_transform(lat=lat, lon=lon)
    easting, northing = tr_to_utm.transform(lon, lat, radians=False, direction="INVERSE")
    return easting, northing, alt, zone_str


def utm_to_lla(
    easting: float,
    northing: float,
    alt: float,
    zone: str
) -> tuple[float]:
    """ Convert UTM x,y, alt to lat, lon, alt.  Altitude is copied over.

    Args:
        x (float): UTM x component in meters.
        y (float): UTM y component in meters.
        alt (float): Altitude in meters
        zone (string): UTM zone integer and letter

    Returns:
        tuple: Latitude, longitude, altitude
    """
    tr_to_lla, _ = get_utm_to_lla_transform(zone_str=zone)
    lon, lat = tr_to_lla.transform(easting, northing, radians=False)
    return lat, lon, alt


def ecef_to_ned_rotation(lat_rad: float, lon_rad: float) -> np.ndarray:
    """
    Calculate rotation matrix from ECEF to NED

    Args:
        lat_rad (float): Latitude in radians
        lon_rad (float): Longitude in radians

    Returns:
        numpy.ndarray: 3x3 rotation matrix from ECEF to NED
    """
    # Calculate trigonometric values
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Create rotation matrix
    rotation = np.zeros((3, 3))

    # First row: North direction in ECEF coordinates
    rotation[0, 0] = -sin_lat * cos_lon
    rotation[0, 1] = -sin_lat * sin_lon
    rotation[0, 2] = cos_lat

    # Second row: East direction in ECEF coordinates
    rotation[1, 0] = -sin_lon
    rotation[1, 1] = cos_lon
    rotation[1, 2] = 0.0

    # Third row: Down direction in ECEF coordinates
    rotation[2, 0] = -cos_lat * cos_lon
    rotation[2, 1] = -cos_lat * sin_lon
    rotation[2, 2] = -sin_lat

    return rotation


def calculate_ecef_gravity(ecef_position: np.ndarray) -> np.ndarray:
    """
    Calculate gravity vector in ECEF frame.

    Args:
        ecef_position (np.ndarray): [x, y, z] in meters

    Returns:
        (np.ndarray): representing gravity vector in ECEF frame (m/s²)
    """

    # Extract position components
    x, y, z = ecef_position

    # Compute position magnitude
    r = np.sqrt(x*x + y*y + z*z)

    # Compute pure gravitational acceleration (Newton's law)
    # Force points toward center of Earth
    gravity_direction = -ecef_position / r
    gravity_magnitude = GM / (r * r)
    gravitational = gravity_direction * gravity_magnitude

    # Compute centrifugal acceleration
    # This acts perpendicular to Earth's rotation axis
    p = np.sqrt(x*x + y*y)  # distance from rotation axis
    if p > 0:
        centrifugal_direction = np.array([x/p, y/p, 0.0])
        centrifugal_magnitude = Ω * Ω * p
        centrifugal = centrifugal_direction * centrifugal_magnitude
    else:
        centrifugal = np.zeros(3)

    # Total acceleration is gravitational + centrifugal
    gravity = gravitational + centrifugal

    return gravity