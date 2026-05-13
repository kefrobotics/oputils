# Third Party
import os
from pathlib import Path

import numpy as np
from ament_index_python.packages import get_package_share_directory
from rosbags.typesys import Stores, get_typestore, get_types_from_msg


def register_typestore(
    additional_types: list[str] = ["mavros_msgs", "kef_messages"]
):
    # Register additional message typestores.
    typestore = get_typestore(Stores.LATEST)
    types = register_types_from_list(additional_types)
    typestore.register(types)
    return typestore


def register_types_from_list(msgs: list[str]) -> dict:
    """ Registers message class of the given types.

    Args:
        msgs (list[str]): list of message classes I.e. Mavlink, kef_messages, etc.

    Returns:
        dict: Mapping between message name and type.
    """
    additional_types = {}
    for msg in msgs:
        # get path to messages *pkg*/msg
        msg_path = os.path.join(get_package_share_directory(msg), "msg")

        # Load all types from the .msg files
        for file_name in os.listdir(msg_path):
            if file_name.endswith(".msg"):
                file_path = os.path.join(msg_path, file_name)

                msg_text = Path(file_path).read_text()
                # basically do e.g. MavlinkBarometer.msg -> mavros_msgs/msg/MavlinkBarometer
                fname = file_name.split(".")[0]
                msg_long = f"{msg}/msg/{fname}"

                msg_types = get_types_from_msg(msg_text, msg_long)
                additional_types.update(msg_types)

    return additional_types


# Constants
TYPESTORE = register_typestore()


def get_type(type_str: str):
    t = TYPESTORE.types[type_str]
    return t, t.__msgtype__


def create_gps_kef_msg(sec: int,
                       nanosec: int,
                       fix_type: int,
                       lat_deg: float,
                       lon_deg: float,
                       alt_m: float,
                       eph: float,
                       epv: float,
                       vel: np.ndarray,
                       cog: float,
                       satellites_visible: float,
                       frame_id: str = "world"):
    """
    Create the GPS KEF message for use.

    Args:
        sec(int): Seconds for timestamp.
        nanosec(int): Nanoseconds for timestamp.
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitdue in degrees.
        alt_m (float): Altitude in meters.
        eph(float): GPS horizontal position accuracy (m).
        epv(float): GPS vertical position accuracy (m).
        vel (np.ndarray): Velocity in meters per second.
        cog (float): Course over ground, direction of movement (-pi, pi).
        satellites_visible (float): Satellite fixes.

    Returns:
        _type_: _description_
    """
    t, msgtype = get_type('kef_messages/msg/GPS')
    header, _ = get_type("std_msgs/msg/Header")
    time, _ = get_type("builtin_interfaces/msg/Time")

    # Create the message
    header = header(time(sec=sec, nanosec=nanosec), frame_id=frame_id)
    msg = t(header, fix_type, lat_deg, lon_deg, alt_m, eph, epv, np.linalg.norm(vel), vel[0], vel[1], vel[2], cog, satellites_visible)
    return msg, TYPESTORE.serialize_cdr(msg, msgtype)

def create_imu_bias_kef_msg(sec: int,
                            nanosec: int,
                            acc_bias: np.ndarray,
                            gyro_bias: np.ndarray,
                            mag_bias: np.ndarray,
                            frame_id: str = "body"):
    """
    Create an IMU bias message.

    Args:
        sec (int): Seconds.
        nanosec (int): Nanoseconds.
        acc_bias (np.ndarray): Accelerometer bias.
        gyro_bias (np.ndarray): Gyroscope bias.
        mag_bias (np.ndarray): Magnetometer bias.
        frame_id (str, optional): Frame string. Defaults to "body".

    Returns:
        message, serialized_message: Return both the raw message and the serialized message for writing
    """

    imu_bias, msgtype = get_type("kef_messages/msg/ImuBias")
    header, _  = get_type("std_msgs/msg/Header")
    time, _    = get_type("builtin_interfaces/msg/Time")
    vector3, _ = get_type("geometry_msgs/msg/Vector3")

    # Create the message
    header = header(time(sec=sec, nanosec=nanosec), frame_id=frame_id)
    acc_bias_ros  = vector3(acc_bias[0], acc_bias[1], acc_bias[2])
    gyro_bias_ros = vector3(gyro_bias[0], gyro_bias[1], gyro_bias[2])
    mag_bias_ros  = vector3(mag_bias[0], mag_bias[1], mag_bias[2])
    msg = imu_bias(header, acc_bias_ros, gyro_bias_ros, mag_bias_ros)
    return msg, TYPESTORE.serialize_cdr(msg, msgtype)

def create_airspeed_kef_msg(sec: int,
                            nanosec: int,
                            aspd: float,
                            frame_id: str="body"):
    """
    Create KEF airspeed message.

    Args:
        sec (int): Seconds
        nanosec (int): Nanoseconds
        aspd (float): Airspeed value.
        frame_id (str, optional): Frame string. Defaults to "body".

    Returns:
        message, serialized_message: Return both the raw message and the serialized message for writing
    """
    # Get the types
    airspeed, msgtype = get_type("kef_messages/msg/Airspeed")
    header, _ = get_type("std_msgs/msg/Header")
    time, _ = get_type("builtin_interfaces/msg/Time")

    # Create the message
    header = header(time(sec=sec, nanosec=nanosec), frame_id=frame_id)
    msg = airspeed(header, aspd)
    return msg, TYPESTORE.serialize_cdr(msg, msgtype)

def create_imu_msg(sec: int,
                   nanosec: int,
                   orientation: list,
                   ω: list,
                   a: list,
                   frame_id: str="body"):
    """
    Create an IMU message.

    Args:
        sec (int): Seconds.
        nanosec (int): Nanoseconds.
        orientation (list): Quaternion orientation, [x,y,z,w]
        a (list): Acceleration
        frame_id (str, optional): Frame string. Defaults to "body".

    Returns:
        message, serialized_message: Return both the raw message and the serialized message for writing
    """

    # Get the types
    imu, msgtype = get_type("sensor_msgs/msg/Imu")
    quaternion, _ = get_type("geometry_msgs/msg/Quaternion")
    vec3, _ = get_type("geometry_msgs/msg/Vector3")
    header, _ = get_type("std_msgs/msg/Header")
    time, _ = get_type("builtin_interfaces/msg/Time")

    # Create the message
    header = header(time(sec=sec, nanosec=nanosec), frame_id=frame_id)
    ω = vec3(ω[0], ω[1], ω[2])
    a = vec3(a[0], a[1], a[2])
    orientation = quaternion(orientation[0], orientation[1], orientation[2], orientation[3])
    orientation_cov = np.array([0.0]*9)
    a_cov = np.array([0.0]*9)
    ω_cov = np.array([0.0]*9)
    msg = imu(header, orientation, orientation_cov, ω, ω_cov, a, a_cov)
    return msg, TYPESTORE.serialize_cdr(msg, msgtype)
