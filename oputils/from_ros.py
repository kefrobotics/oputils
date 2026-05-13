from typing import Sequence, Union, list, Iterable
from abc import abstractmethod
import os
import yaml

# Third Party
import numpy as np
import cv2
from pyquaternion import Quaternion
import builtin_interfaces.msg
from sensor_msgs.msg import Image, NavSatFix, Imu
from mavros_msgs.msg import MavlinkBarometer
# from kef_messages.msg import GPS as msg_KEF_GPS
# from kef_messages.msg import ImuBias as msg_ImuBias
# from kef_messages.msg import Airspeed as msg_KEF_Airspeed
# from kef_messages.msg import State as msg_State
from geometry_msgs.msg import Pose, PoseWithCovariance
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import rosbags
import rosbags.typesys

# In House
from open_pacific.data.internal_dataclasses import (
    GlobalPosition as internal_GlobalPosition,
    IMU as internal_IMU,
    Transform as internal_Transform,
    Pose as internal_Pose,
    GlobalPose as internal_GlobalPose,
    Twist as internal_Twist,
    Image as internal_Image,
    Barometer as internal_Barometer,
    KEF_GPS as internal_KEF_GPS,
    IMUBias as internal_IMU_Bias,
    Airspeed as internal_Airspeed,
    RCOut as internal_RCOut,
    ActuatorControl as internal_ActuatorControl,
)
from open_pacific.utils.file_utils import create_dir


def timestampFromRos(rts: builtin_interfaces.msg.Time) -> float:
    """ Convert stamp.sec and stamp.nanosec to float

    Args:
        rts (builtin_interfaces.msg.Time): Time stamp message instance.

    Returns:
        float: Corresponding time in seconds.
    """
    ts = 0
    if hasattr(rts, "sec") and hasattr(rts, "nanosec"):
        ts = rts.sec + rts.nanosec / 1.e9
    else:
        ts = rts
    return ts


def quaternionFromRos(pose_with_cov: Union[PoseWithCovariance, Pose]) -> Quaternion:
    """ Converts PoseWithCovariance message to a quaternion

    Args:
        pose_with_cov (PoseWithCovariance or Pose): ROS2 pose message that
            includes orientation information.

    Returns:
        Quaternion: pyquaternion object of quaternion informaiton extracted
            from message.
    """
    return Quaternion(
        w=pose_with_cov.pose.orientation.w,
        x=pose_with_cov.pose.orientation.x,
        y=pose_with_cov.pose.orientation.y,
        z=pose_with_cov.pose.orientation.z
    )


def _sync(master_ts: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """ Find closest indices between master_ts and ts
    """
    # TODO verify this
    closest_indices = np.abs(master_ts[:, np.newaxis] - ts).argmin(axis=1)
    return closest_indices


class TimeSeries:
    def __init__(self, dir, **kwargs):
        self.dir = dir
        self.data = []
        self.ts = []
        self.N = 0
        self.name = kwargs["name"]
        self.dir = dir

        # self.history_length = kwargs.get("history_length, ")
        self.history_length = 1

    @abstractmethod
    def from_ros(self, msg):
        """
        An abstract method that should handle going from a ROS2 message/data packet to a numpy array.

        Args:
            msg: ROS2 message
        """
        pass

    def ingest(self, data, ts):
        self.ts.append(ts)
        self.data.append(data)
        self.N += 1

    def _save_ts(self):
        savedir = os.path.join(self.dir, self.name, "timestamps")
        ts = np.array(self.ts)
        np.save(savedir, ts)

    def _load_ts(self):
        # TODO compute hz
        # TODO figure out way to define time window + overlap etc
        savedir = os.path.join(self.dir, self.name, "timestamps.npy")
        self.ts = np.load(savedir)
        self.mapping = np.arange(self.ts.shape[0])

    def save(self):
        assert len(self.data) > 0, "Trying to overwrite (potentially) existing data with empty array. Something is probably wrong..."

        self._save_ts()

        savedir = os.path.join(self.dir, self.name, "data")
        data = np.array(self.data)
        np.save(savedir, data)

    def load(self):
        self._load_ts()

        savedir = os.path.join(self.dir, self.name, "data.npy")
        self.data = np.load(savedir)

    def sync(self, master_convert):
        # TODO warn if master_ts is less than this
        # still keep old data, that way can still grab intermediate data
        # e.g. for a given camera timestamp, we can still grab the past 20 IMU samples
        # TODO cache
        create_dir(os.path.join(self.dir, "sync_cache", master_convert.name), warning="")

        cache_path = os.path.join(self.dir, "sync_cache", master_convert.name, self.name + "_sync.npy")
        if os.path.exists(cache_path):
            print("WARNING: Using cached time sync for " + master_convert.name + "<-->" + self.name)
            self.mapping = np.load(cache_path)
        else:
            self.mapping = _sync(master_convert.ts, self.ts)
            np.save(cache_path, self.mapping)

    def get(self, idx):
        # TODO support multiple samples
        # this also involves shifting the data so that t1 starts at data[t1+history_length]
        # return self.data[self.mapping[idx]]
        synced_idx = self.mapping[idx]

        if synced_idx - self.history_length >= 0:
            return self.data[synced_idx - self.history_length + 1:synced_idx + 1]
        else:
            return self.data[0:synced_idx + 1]  # hack temp solution

    def plot(self, data, ax):
        ax.plot(data)

    def set_history_length(self, length):
        self.history_length = length

class NSF(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        out = np.array([msg.latitude, msg.longitude, msg.altitude])

        return out

    def plot(self, data, ax):
        ax.plot(data[:,0], data[:,1])

class barometer(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        out = np.array([msg.pressure, msg.temperature, msg.altitude])
        return out

    def plot(self, data, ax):
        ax.plot(data[:,2])

class odom(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        pdot = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        qdot = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        res = np.array(p + q + pdot + qdot)

        return res

    def plot(self, data, ax):
        ax.plot(data[:,0], data[:,1])

class imu(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        out = []
        out += [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        out += [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        out += [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]

        return np.array(out)

class Twist(TimeSeries):
    """
    Twist data class.
    """
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        """
        Convert ROS message to array.

        Args:
            msg (nav_msgs/Twist): ROS2 message
        """
        out = []
        out += [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        out += [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]

        return np.array(out)

class image(TimeSeries):
    def __init__(self, dir, encoding="bgr8", viz=False, ext = ".png", **kwargs):
        self.timeseries = False
        self.bridge = CvBridge()
        self.encoding = encoding
        self.viz = viz
        self.ext = ext
        self.N = 0
        self.name = kwargs["name"]
        self.dir = dir
        self.ts = []

    def from_ros(self, msg):
        im = self.bridge.imgmsg_to_cv2(msg)#, desired_encoding=self.encoding)

        if self.viz:
            cv2.imshow("fpv", im)
            cv2.waitKey(1)

        return im

    def ingest(self, data, ts):
        self.ts.append(ts)
        sname = os.path.join(self.dir, self.name, str(self.N) + self.ext)
        cv2.imwrite(sname, data)

        self.N += 1

    def save(self):
        self._save_ts()

    def load(self):
        self._load_ts()

    def get(self, idx):
        # Get data with respect to mapping (if synced with other sensor)
        img = cv2.imread(os.path.join(self.dir, self.name, str(idx) + self.ext))
        return img

    def plot(self, data, ax):
        ax.imshow(data[:,:,::-1])

class IMUBias(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        out = []
        out += [msg.acc.x, msg.acc.y, msg.acc.z]
        out += [msg.gyro.x, msg.gyro.y, msg.gyro.z]
        out += [msg.mag.x, msg.mag.y, msg.mag.z]
        return out

class KEF_GPS(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        out = []
        out += [msg.latitude_deg, msg.longitude_deg, msg.altitude]
        out += [msg.eph, msg.epv]
        out += [msg.velocity, msg.velocity_north, msg.velocity_east, msg.velocity_down]
        out += [msg.cog, msg.satellites_visible]
        return out

class Airspeed(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        out = []
        out += [msg.airspeed]
        return out

class State(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        out = []
        out += [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        out += [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
        out += [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        out += [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
        out += [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        return out


class RCOut(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        return list(msg.channels)  # uint16 PWM values per channel


class ActuatorControl(TimeSeries):
    def __init__(self, dir, **kwargs):
        super().__init__(dir, **kwargs)
        self.timeseries = True

    def from_ros(self, msg):
        return list(msg.controls)  # float32[8] normalized values


CONVERTER_DICT = {Image: image,
                  NavSatFix: NSF,
                  MavlinkBarometer: barometer,
                  Odometry: odom,
                  Imu: imu}

CONVERTER_DICT_STR = {"Image": image,
                      "NavSatFix": NSF,
                      "MavlinkBarometer": barometer,
                      "Odometry": odom,
                      "IMU": imu,
                      "IMUBias": IMUBias,
                      "KEF_GPS": KEF_GPS,
                      "Airspeed": Airspeed,
                      "Twist": Twist,
                      "RCOut": RCOut,
                      "ActuatorControl": ActuatorControl,
                      "State": State}


def generate_dtype(typename, savedir, **kwargs):
    dtype = CONVERTER_DICT_STR[typename]

    return dtype(savedir, **kwargs)


class DataManager:
    def __init__(self,
                 data_dir: str,
                 modalities: list,
                 master: str = None):
        """Modular offline iterating through dataset
           - Provides timesynced data through each sensor in an iterable fashion

        Args:
            data_dir (string): directory location of dataset
            modalities (list): desired modalities
            master (string, optional): All other modalities will be synced to this topic
        """
        self.data_dir = data_dir
        self.modalities = modalities

        if len(modalities) > 1:
            assert master is not None, "Please specify topic to sync to"

        with open(os.path.join(self.data_dir, "config.yaml"), "r") as file:
            config = yaml.safe_load(file)

        self.converters = {}
        ct = config["topics"]
        for topic in ct:
            name = ct[topic]["name"]

            if name in modalities:
                converter = generate_dtype(ct[topic]["type"], self.data_dir, **ct[topic])
                converter.load()

                self.converters[name] = converter

        if master is not None:
            master_convert = self.converters[master]

            for converter in self.converters.values():
                converter.sync(master_convert)
        else:
            master_convert = self.converters[master]

        self.N = master_convert.ts.shape[0]

    def __len__(self):
        return self.N

    def get(self, idx):
        output = {}
        for modality in self.modalities:
            output[modality] = self.converters[modality].get(idx)

        return output

def convert_tf_static_to_transform(data: Sequence[Odometry]):
    """
    Convert TF static transform to internal data type

    Args:
        data (Transform): Transform message.
    """
    transform_out = None
    for i in data[0].transforms:
        if i.child_frame_id == "odom_ned":
            r = i.transform.rotation
            t = i.transform.translation
            ts = timestampFromRos(i.header.stamp)
            q = Quaternion(x=r.x, y=r.y, z=r.z, w=r.w)
            transform_out = internal_Transform(
                ts=ts,
                t=np.array([t.x, t.y, t.z]),
                orientation=q
            )
    return transform_out

def convert_nav_odom_to_Pose(data: Sequence[Odometry]):
    """
    Convert nav odom message to a Pose data class.

    Args:
        data (Odometry): Navigation odometry message

    Returns:
        dataclasses.Pose: Pose data structure.
    """
    out = []
    converter = generate_dtype("Odometry", "", name="")
    for i in data:
        np_data = converter.from_ros(i)
        ts = timestampFromRos(i.header.stamp)
        orientation = Quaternion(x=np_data[3], y=np_data[4], z=np_data[5], w=np_data[6])
        pose = internal_Pose(ts=ts, position=np_data[:3], orientation=orientation)
        out.append(pose)
    return out

def convert_nsf_to_GlobalPosition(data: Sequence[NavSatFix]):
    """
    Convert GPS topic to Point3 data structure.

    Args:
        data  Sequence[NavSatFix]: NavSatFix messages.

    Returns:
        Position: Position data structure.
    """
    out = []
    converter = generate_dtype("NavSatFix", "", name="")
    for i in data:
        lat, lon, alt = converter.from_ros(i)
        ts = timestampFromRos(i.header.stamp)
        pos = internal_GlobalPosition(ts=ts, lat=lat, lon=lon, alt=alt)
        out.append(pos)
    return out

def convert_nav_odom_to_GlobalPose(data: Sequence[Odometry]):
    """
    Convert nav odom message to a GlobalPose data class.
    NOTE: this assumes NAV_ODOM position information is in ECEF units

    Args:
        data (Odometry): Navigation odometry message

    Returns:
        dataclasses.GlobalPose: Pose data structure.
    """
    out = []
    converter = generate_dtype("Odometry", "", name="")
    for i in data:
        np_data = converter.from_ros(i)
        ts = timestampFromRos(i.header.stamp)
        orientation = Quaternion(x=np_data[3], y=np_data[4], z=np_data[5], w=np_data[6])
        global_position = internal_GlobalPosition(ts=ts, ecef=np_data[:3])
        pose = internal_GlobalPose(ts=ts, position=global_position, orientation=orientation)
        out.append(pose)
    return out

def convert_nav_odom_to_velocity(data: Sequence[Odometry]):
    """
    Extract velocity from nav_msgs/Odometry twist.twist.linear.
    Used to pull velocity from /odom/px4 (or similar) as GPS_VEL.

    Args:
        data (Sequence[Odometry]): Navigation odometry messages.

    Returns:
        list[internal_Twist]: Velocity in whatever frame the topic publishes in.
    """
    out = []
    for i in data:
        ts = timestampFromRos(i.header.stamp)
        v = np.array([
            i.twist.twist.linear.x,
            i.twist.twist.linear.y,
            i.twist.twist.linear.z,
        ], dtype=np.float64)
        twist_i = internal_Twist(ts, v, np.zeros(3))
        out.append(twist_i)
    return out

def convert_imu(data: Sequence[Imu]):
    """
    Convert ROS IMU data to internal data class

    Args:
        data (sensor_msgs/msg/IMU): IMU data
    """
    out = []
    converter = generate_dtype("IMU", "", name="")
    for i in data:
        np_data = converter.from_ros(i)
        ts = timestampFromRos(i.header.stamp)
        imu_data = internal_IMU(ts=ts, ω=np_data[4:7], a=np_data[7:])
        out.append(imu_data)
    return out

def convert_barometer(data: list):
    """
    Convert barometer data to a list of floats

    Args:
        data (list[MavlinkBarometer]): Barometer messages
    """
    # return convert_generic(data, "MavlinkBarometer")
    out = []
    converter = generate_dtype("MavlinkBarometer", "", name="")
    for i in data:
        pressure, temp, alt = converter.from_ros(i)
        ts = timestampFromRos(i.header.stamp)
        baro_data = internal_Barometer(
            ts=ts, pressure=pressure, temperature=temp, altitude=alt
        )
        out.append(baro_data)
    return out

def convert_image(data: list):
    """
    Convert image messages to a list of np images.

    Args:
        data (list[Image]): Image messages
    """
    # return convert_generic(data, "Image")
    out = []
    converter = generate_dtype("Image", "", name="")
    for i in data:
        img = converter.from_ros(i)
        ts = timestampFromRos(i.header.stamp)
        img_data = internal_Image(ts=ts, img=img)
        out.append(img_data)
    return out

def convert_twist(data: list):
    """
    Convert twist ROS2 types to internal.

    Args:
        data (list): list of ROS2 Twist messages.

    Returns:
        list: list of internal data.
    """
    out = []
    # Convert twist to numpy
    converter = generate_dtype("Twist", "", name="")
    for i in data:
        np_data = converter.from_ros(i)
        ts = timestampFromRos(i.header.stamp)
        v = np_data[:3]
        ω = np_data[3:]
        twist_i = internal_Twist(ts, v, ω)
        out.append(twist_i)

    return out

def convert_kef_state_velocity(data: list):
    """
    Extract NED velocity from kef_messages/msg/State (e.g. /state/px4).
    The twist.linear field is in NED frame for PX4 state estimates.

    Args:
        data (list): list of kef_messages/msg/State messages.

    Returns:
        list[internal_Twist]: Velocity (linear only; angular set to zero).
    """
    out = []
    for i in data:
        ts = timestampFromRos(i.header.stamp)
        v = np.array([i.twist.linear.x, i.twist.linear.y, i.twist.linear.z], dtype=np.float64)
        twist_i = internal_Twist(ts, v, np.zeros(3))
        out.append(twist_i)
    return out

def convert_kef_gps(data: Iterable):
    """
    Convert KEF GPS

    Args:
        data (Iterable): The ROS2 KEF GPS messages

    Returns:
        list: list of internal data.
    """
    out = []
    converter = generate_dtype("KEF_GPS", "", name="")
    for i in data:
        ts = timestampFromRos(i.header.stamp)
        np_data = converter.from_ros(i)
        lat_deg, lon_deg, alt_m = np_data[:3]
        pos = internal_GlobalPosition(ts, lat=lat_deg, lon=lon_deg, alt=alt_m)
        eph, epv = np_data[3:5]
        velocity, vn, ve, vd = np_data[5:9]
        cog, sat_viz = np_data[9:]
        gps_i = internal_KEF_GPS(ts, pos, eph, epv, velocity, vn, ve, vd, cog, sat_viz)
        out.append(gps_i)
    return out

def convert_imu_bias(data):
    """
    Convert IMU bias

    Args:
        data (Iterable): The ROS2 IMU bias messages

    Returns:
        list: list of internal data.
    """
    out = []
    converter = generate_dtype("IMUBias", "", name="")
    for i in data:
        ts = timestampFromRos(i.header.stamp)
        np_data = converter.from_ros(i)
        bias_i = internal_IMU_Bias(ts, np_data[:3], np_data[3:6], np_data[6:])
        out.append(bias_i)
    return out


def convert_kef_gps(data: Iterable):
    """
    Convert KEF GPS

    Args:
        data (Iterable): The ROS2 KEF GPS messages

    Returns:
        list: list of internal data.
    """
    out = []
    converter = generate_dtype("KEF_GPS", "", name="")
    for i in data:
        ts = timestampFromRos(i.header.stamp)
        np_data = converter.from_ros(i)
        lat_deg, lon_deg, alt_m = np_data[:3]
        pos = internal_GlobalPosition(ts, lat=lat_deg, lon=lon_deg, alt=alt_m)
        eph, epv = np_data[3:5]
        velocity, vn, ve, vd = np_data[5:9]
        cog, sat_viz = np_data[9:]
        gps_i = internal_KEF_GPS(ts, pos, eph, epv, velocity, vn, ve, vd, cog, sat_viz)
        out.append(gps_i)
    return out

def convert_imu_bias(data):
    """
    Convert IMU bias

    Args:
        data (Iterable): The ROS2 IMU bias messages

    Returns:
        list: list of internal data.
    """
    out = []
    converter = generate_dtype("IMUBias", "", name="")
    for i in data:
        ts = timestampFromRos(i.header.stamp)
        np_data = converter.from_ros(i)
        bias_i = internal_IMU_Bias(ts, np_data[:3], np_data[3:6], np_data[6:])
        out.append(bias_i)
    return out

def convert_airspeed(data: Iterable):
    """
    Convert airspeed message.

    Args:
        data (Iterable): The ROS2 airspeed message.

    Returns:
        list: list of internal data.
    """
    out = []
    converter = generate_dtype("Airspeed", "", name="")
    for i in data:
        ts = timestampFromRos(i.header.stamp)
        np_data = converter.from_ros(i)
        airspeed_i = internal_Airspeed(ts, np_data[0])
        out.append(airspeed_i)
    return out

def convert_rc_out(data: Iterable):
    """Convert mavros RCOut messages to internal RCOut dataclasses.

    Args:
        data (Iterable): ROS2 mavros_msgs/RCOut messages.

    Returns:
        list[internal_RCOut]: list of internal RC output objects.
    """
    out = []
    converter = generate_dtype("RCOut", "", name="")
    for i in data:
        ts = timestampFromRos(i.header.stamp)
        channels = np.array(converter.from_ros(i), dtype=np.float32)
        out.append(internal_RCOut(ts=ts, channels=channels))
    return out


def convert_actuator_control(data: Iterable):
    """Convert mavros ActuatorControl messages to internal ActuatorControl dataclasses.

    Args:
        data (Iterable): ROS2 mavros_msgs/ActuatorControl messages from e.g. /physics/actuators_in.

    Returns:
        list[internal_ActuatorControl]: list of internal actuator control objects.
    """
    out = []
    converter = generate_dtype("ActuatorControl", "", name="")
    for i in data:
        ts = timestampFromRos(i.header.stamp)
        controls = np.array(converter.from_ros(i), dtype=np.float32)
        out.append(internal_ActuatorControl(ts=ts, controls=controls, group_mix=int(i.group_mix)))
    return out


def convert_generic(data: list, dtype_str: str):
    """
    Convert for a generic data type, for something to be returned as a default type, not a special data structure

    Args:
        data (list): The data of interest.
        dtype_st (str): String for the converter.
    """
    out = []
    converter = generate_dtype(dtype_str, "", name="")
    for i in data:
        c = converter.from_ros(i)
        out.append(c)
    return out

def build_pose_msg(stamp: str,
                   p: np.ndarray,
                   rot: Quaternion,
                   typestore: rosbags.typesys.Stores):
    """
    Build pose message from internal types

    Args:
        p (np.ndarray): position.
        rot (Quaternion): orientation.
        typestore (rosbags.typestore): ROS typestore.

    Returns:
        _type_: _description_
    """
    # Extract type information
    time         = typestore.types["builtin_interfaces/msg/Time"]
    header       = typestore.types["std_msgs/msg/Header"]
    pose_stamped = typestore.types["geometry_msgs/msg/PoseStamped"]
    pose         = typestore.types["geometry_msgs/msg/Pose"]
    position     = typestore.types["geometry_msgs/msg/Point"]
    orientation  = typestore.types["geometry_msgs/msg/Quaternion"]

    # Fill the ROS2 message
    parts = stamp.split(".")
    sec = int(parts[0])
    if len(parts) == 1:  # No decimal point
        nsec = 0
    else:
        fractional = parts[1]
        nsec = int(fractional.ljust(9, '0')[:9])
    h = header(frame_id="world", stamp=time(sec=sec, nanosec=nsec))
    p = position(x=p[0], y=p[1], z=p[2])
    o = orientation(x=rot.x, y=rot.y, z=rot.z, w=rot.w)
    p2 = pose(position=p, orientation=o)
    pose_stamp = pose_stamped(pose=p2, header=h)
    return pose_stamp
