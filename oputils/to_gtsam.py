# Third Party
from gtsam import Rot3
from pyquaternion import Quaternion


def rpy_to_Rot3(r: float, p: float, y: float):
    """
    Create GTSAM Rot3 from YPR

    Args:
        r (float): roll in radians.
        p (float): pitch in radians.
        y (float): yaw in radians.

    Returns:
        gtsam.Rot3: GTSAM 3D rotation object instance.
    """
    init_rot = Rot3.RzRyRx(x=r, y=p, z=y)
    return init_rot


def quaternion_to_Rot3(q: Quaternion):
    """
    Convert pyquaternion quaternion to gtsam Rot3

    Args:
        q (Quaternion): rotation quaternion

    Returns:
        Rot3: corresponding GTSAM rotation instance.
    """
    return Rot3(x=q.x, y=q.y, z=q.z, w=q.w)
