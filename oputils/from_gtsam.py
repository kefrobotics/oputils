from gtsam import Pose3, Point3, Rot3
from gtsam.imuBias import ConstantBias
import numpy as np
from pyquaternion import Quaternion

from open_pacific.data.internal_dataclasses import (
    Pose, GlobalPose, Position, GlobalPosition, IMUBias
)


def Rot3_to_Quaternion(R: Rot3) -> Quaternion:
    """ Converts from GTSAM rotation object to a pyquaternion object.

    Args:
        R (Rot3): GSTAM rotation instance

    Returns:
        Quaternion: pyquaternion Quaternion instance.
    """
    q_from_gtsam = R.toQuaternion()
    return Quaternion(
        x=q_from_gtsam.x(),
        y=q_from_gtsam.y(),
        z=q_from_gtsam.z(),
        w=q_from_gtsam.w()
    )


def Pose3_to_Pose(
    p: Pose3, ts: float, cov: np.ndarray = None
) -> Pose:
    """ Converts GTSAM Pose3 to an internal pose type.

    Args:
        p (Pose3): GTSAM pose
        ts (float): Time stamp in seconds
        cov (np.ndarray, optional): Covariance.  Defaults to None

    Returns:
        Pose: Internal Pose
    """
    return Pose(
        ts=ts,
        position=p.translation(),
        orientation=Rot3_to_Quaternion(p.rotation()),
        cov=cov
    )


def Pose3_to_GlobalPose(
    p: Pose3, ts: float, cov: np.ndarray = None
) -> GlobalPose:
    """ Converts GTSAM Pose3 to an internal global pose type.
        Position information in Pose3 will be treated as an
        ECEF vector.

    Args:
        p (Pose3): GTSAM pose
        ts (float): Time stamp in seconds
        cov (np.ndarray, optional): Covariance.  Defaults to None

    Returns:
        GlobalPose: Internal Global Pose
    """
    return GlobalPose(
        ts=ts,
        position=GlobalPosition(ts=ts, ecef=p.translation()),
        orientation=Rot3_to_Quaternion(p.rotation()),
        cov=cov
    )


def Point3_to_Position(
    p: Point3, ts: float, cov: np.ndarray = None
) ->  Position:
    """ Convert GTSAM Point3 to a generic Position instance

    Args:
        p (Point3): GTSAM Point3 instance
        ts (float): Time stamp in seconds
        cov (np.ndarray, optional): Covariance.  Defaults to None

    Returns:
        Position: Corresponding Position instance
    """
    return Position(
        ts=ts,
        position=p.translation(),
        cov=cov
    )


def Point3_to_GlobalPosition(
    p: Point3, ts: float, cov: np.ndarray = None
) -> GlobalPosition:
    """ Convert GTSAM Point3 to a GlobalPosition instance.
        NOTE: This assumes that Point3 is an ECEF vector.

    Args:
        p (Point3): GTSAM Point3 instance
        ts (float): Time stamp in seconds
        cov (np.ndarray, optional): Covariance.  Defaults to None

    Returns:
        Position: Corresponding GlobalPosition instance
    """
    return GlobalPosition(
        ts=ts,
        ecef=p.translation(),
        cov=cov
    )


def ConstantBias_to_IMUBias(
    cb: ConstantBias, ts: float, cov: np.ndarray = None
) -> IMUBias:
    """ Converts GTSAM ConstantBias to IMUBias instance.

    Args:
        cb (ConstantBias): Bias estimate from bakend
        ts (float): Current time stamp
        cov (np.ndarray, optional): current covariance. Defaults to None.

    Returns:
        IMUBias: Corresponding interntal data type instance for IMU bias.
    """
    return IMUBias(
        ts=ts,
        a_bias=cb.accelerometer(),
        ω_bias=cb.gyroscope(),
        cov=cov
    )
