import pytest
import numpy as np
from pyquaternion import Quaternion
from gtsam.utils.circlePose3 import circlePose3


import oputils.types as idc


def test_GlobalPosition_from_ECEF():
    """ Tests GlobalPosition construction with ECEF """
    # True LLA: 20, -180, 100 meters
    test_GP = idc.GlobalPosition(
        ts=0,
        ecef=np.asarray([1041182., -5904839., 2167731.]),
    )
    lat, lon, alt = test_GP.get_lla()
    utm_x, utm_y, utm_alt = test_GP.get_utm()
    zone = test_GP.get_zone()
    assert lat == pytest.approx(20., 1e-3)
    assert lon == pytest.approx(-80., 1e-3)
    assert alt == pytest.approx(100.221, 1e-2)
    assert utm_x == pytest.approx(604608.8985, 1e-3)
    assert utm_y == pytest.approx(2211793.4841, 1e-3)
    assert utm_alt == pytest.approx(100.221, 1e-2)
    assert zone == "17Q"


def test_GlobalPosition_from_LLA():
    """ Tests GlobalPosition construction with lat, lon, alt."""
    # True LLA: 20, -180, 100 meters
    test_GP = idc.GlobalPosition(
        ts=0,
        lat=20.,
        lon=-80.,
        alt=100.
    )
    ecef_x, ecef_y, ecef_z = test_GP.get_ecef()
    utm_x, utm_y, utm_alt = test_GP.get_utm()
    zone = test_GP.get_zone()
    assert ecef_x == pytest.approx(1041182., 1e-3)
    assert ecef_y == pytest.approx(-5904839., 1e-3)
    assert ecef_z == pytest.approx(2167731., 1e-3)
    assert utm_y == pytest.approx(2211793.56, 1e-3)
    assert utm_x == pytest.approx(604608.8985, 1e-3)
    assert utm_alt == pytest.approx(100.221, 1e-2)
    assert zone == "17Q"


def test_GlobalPosition_from_UTM():
    """ Tests GlobalPosition construction with UTM (x,y) and alt"""
    # True LLA: 20, -180, 100 meters
    test_GP = idc.GlobalPosition(
        ts=0,
        utm_x=604608.8985,
        utm_y=2211793.4841,
        alt=100.,
        zone="17Q"
    )
    ecef_x, ecef_y, ecef_z = test_GP.get_ecef()
    lat, lon, alt = test_GP.get_lla()
    assert ecef_x == pytest.approx(1041182., 1e-3)
    assert ecef_y == pytest.approx(-5904839., 1e-3)
    assert ecef_z == pytest.approx(2167731., 1e-3)
    assert lat == pytest.approx(20., 1e-3)
    assert lon == pytest.approx(-80., 1e-3)
    assert alt == pytest.approx(100.221, 1e-2)


def test_pose_between():
    """
    I am testing:

    1. If the between pose transformation is accurate for creating odometries
    2. If it matches the GTSAM implementation
    """

    # Use internal circle poses in GTSAM
    poses = circlePose3(numPoses=6, radius=3.0)
    p0    = poses.atPose3(0)
    p1    = poses.atPose3(1)
    delta = p0.between(p1)
    dq = delta.rotation().toQuaternion()
    dq = Quaternion(x=dq.x(), y=dq.y(), z=dq.z(), w=dq.w())

    # Create the internal implementation
    q0 = p0.rotation().toQuaternion()
    q0 = Quaternion(x=q0.x(), y=q0.y(), z=q0.z(), w=q0.w())
    q1 = p1.rotation().toQuaternion()
    q1 = Quaternion(x=q1.x(), y=q1.y(), z=q1.z(), w=q1.w())
    p0_internal = idc.Pose(ts=1.0, position=p0.translation(), orientation=q0)
    p1_internal = idc.Pose(ts=1.3, position=p1.translation(), orientation=q1)
    tform = p0_internal.transform_between(p1_internal)

    assert np.isclose(delta.translation(), tform.t, 1e-3).all()
    assert tform.orientation == dq
