# Third Party
import pytest
import numpy as np

# In House
from oputils.coordinate_utils import lla_to_ecef, lla_to_utm, ecef_to_lla, utm_to_lla, get_utm_to_lla_transform, ecef_to_ned_rotation, ecef_to_ned_rotation_using_ecef, calculate_ecef_gravity
from oputils.types import GlobalPosition

LLA = (
    (0., 0., 1000.),
    (0., 90., 1000.),
    (90., 0., 1000.),
    (-45., 30., 0.)
)

ECEF = (
    (6379137., 0., 0.),
    (0., 6379137., 0.),
    (0., 0., 6357752.),
    (3912348., 2258795., -4487348.)
)

# Easting, Northing, Up
UTM = (
    (166021.44, 0., 1000., "31N"),
    (166021.44, 0.,  1000., "46N"),
    (500000.0, 9997964.94,  1000., "31Z"),
    (263553.97, 5012670.50,  0., "36G")
)

GLOBALPOSITION_CONSTRUCTORS = (
    {"ecef": np.array([-2365824.33, -3357626.03, 4863281.6])},
    {"lat": 50.0060070, "lon": -125.1689924, "alt": 82.38},
    {"utm_x": 344580.48, "utm_y": 5541552.82, "alt": 82.37, "zone": "10U"}
)


@pytest.mark.parametrize("lla,ecef", zip(LLA, ECEF))
def test_lla_to_ecef(lla, ecef):
    test_ecef = lla_to_ecef(*lla)
    print(test_ecef)
    print(ecef)
    for i in range(3):
        assert test_ecef[i] == pytest.approx(ecef[i], abs=1e-0)


@pytest.mark.parametrize("lla,utm", zip(LLA, UTM))
def test_lla_to_utm(lla, utm):
    test_east, test_north, test_alt, test_zone = lla_to_utm(*lla)
    e, n, alt, zone = utm
    if test_east is None and test_north is None and test_zone is None:
        assert True
    else:
        assert test_north == pytest.approx(n)
        assert test_east == pytest.approx(e)
        assert test_alt == alt
        assert test_zone == zone


@pytest.mark.parametrize("ecef,lla", zip(ECEF, LLA))
def test_ecef_to_lla(ecef, lla):
    if lla[0] > 84. or lla[0] < -80.:
        assert True
    else:
        test_lat, test_lon, test_alt = ecef_to_lla(ecef)
        lat, lon, alt = lla
        print(test_lat, test_lon, test_alt)
        print(lat, lon, alt)
        assert test_lat == pytest.approx(lat, abs=1e-3)
        assert test_lon == pytest.approx(lon, abs=1e-3)
        assert test_alt == pytest.approx(alt, abs=1e0)


@pytest.mark.parametrize("utm,lla", zip(UTM, LLA))
def test_utm_to_lla(utm, lla):
    test_lat, test_lon, test_alt = utm_to_lla(*utm)
    lat, lon, alt = lla
    assert test_lat == pytest.approx(lat, abs=3.)
    assert test_lon == pytest.approx(lon, abs=3.)
    assert test_alt == alt


@pytest.mark.parametrize("build", GLOBALPOSITION_CONSTRUCTORS)
def test_GlobalPosition(build):
    test_gp = GlobalPosition(ts=12345., **build)
    lat = 50.006007
    lon = -125.168992
    ecef = np.array([-2365824.33, -3357626.03, 4863281.6])
    alt = 82.38
    zone = "10U"
    utm_north = 5541552.82
    utm_east = 344580.48

    assert test_gp.lat == pytest.approx(lat, abs=1e-1)
    assert test_gp.lon == pytest.approx(lon, abs=1e-1)
    assert test_gp.alt == pytest.approx(alt, abs=1e-1)
    assert test_gp.zone == zone
    assert test_gp.utm_x == pytest.approx(utm_east, abs=1e-1)
    assert test_gp.utm_y == pytest.approx(utm_north, abs=1e-1)
    for i in range(3):
        assert test_gp.ecef[i] == pytest.approx(ecef[i], abs=1e-1)

def test_zone_and_latlon_combinations():
    get_utm_to_lla_transform(20.34, -78.58, zone_str=None)
    get_utm_to_lla_transform(None, None, zone_str="17R")
    with pytest.raises(SystemExit):
        get_utm_to_lla_transform(None, None, zone_str=None)
    with pytest.raises(SystemExit):
        utm_to_lla(easting=30.4, northing=53.5, alt=50.3, zone=None)

def test_ned_to_ecef_pittsburgh():
    R_truth_ned_to_ecef = np.array([
        [-0.112684, 0.984795, -0.132214],
        [ 0.638797, 0.173719,  0.749507],
        [ 0.761079, 0.000000, -0.648659]])
    R_ecef_to_ned = ecef_to_ned_rotation(lat_rad=40.4406*np.pi/180, lon_rad=-79.9959*np.pi/180)
    R_ned_to_ecef = R_ecef_to_ned.T
    assert np.isclose(R_ecef_to_ned @ R_ned_to_ecef, np.eye(3), 1e-3).all()
    assert np.isclose(R_truth_ned_to_ecef, R_ned_to_ecef, 1e-3).all()

    # also test with ECEF directly
    pgh_ecef = GlobalPosition(lat_rad=40.4406*np.pi/180, lon_rad=-79.9959*np.pi/180, )
    R_ecef_to_ned2 = ecef_to_ned_rotation_using_ecef(pgh_ecef)

    for i, true in enumerate(R_truth_ned_to_ecef.flatten()):
        assert true == pytest.approx(R_ecef_to_ned2.flatten()[i])

def test_ecef_gravity_prime_meridian():
    # Point on prime meridian at equator (on Earth's surface)
    result = calculate_ecef_gravity(ecef_position=np.array([6378137.0, 0.0, 0.0]))
    print("Prime Meridian at Equator gravity:", result)
    print("Magnitude:", np.linalg.norm(result))
    # Expected: roughly [-9.78, 0.0, 0.0] m/s²
    assert np.isclose(result, np.array([-9.7643, 0.0, 0.0])).all()

def test_ecef_gravity_north_pole():
    # North Pole (on Earth's surface)
    result = calculate_ecef_gravity(ecef_position=np.array([0.0, 0.0, 6356752.3]))
    print("North Pole gravity:", result)
    print("Magnitude:", np.linalg.norm(result))
    # Expected: roughly [0.0, 0.0, -9.83] m/s²
    assert np.isclose(result, np.array([0.0, 0.0, -9.8643]), atol=1e-3).all()

def test_ecef_gravity_equator():
    # Point on equator at 90° East (on Earth's surface)
    result = calculate_ecef_gravity(ecef_position=np.array([0.0, 6378137.0, 0.0]))
    print("Equator at 90°E gravity:", result)
    print("Magnitude:", np.linalg.norm(result))
    # Expected: roughly [0.0, -9.7643, 0.0] m/s²
    assert np.isclose(result, np.array([0.0, -9.7643, 0.0]), atol=1e-3).all()

test_GlobalPosition({'alt': 82.37, 'utm_x': 344580.48, 'utm_y': 5541552.82, 'zone': '10U'})