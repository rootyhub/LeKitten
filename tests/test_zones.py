"""Zone geometry: play surfaces and the seated-person no-aim volume."""
import numpy as np

from lekitten.config import load_config
from lekitten.zones import Zones


def living(couch_play: bool):
    room = dict(load_config()["rooms"]["living_room"])
    room["couch_play"] = couch_play
    return Zones(room)


def test_couch_seat_is_a_surface_only_with_couch_play():
    seat = np.array([2.2, 1.2, 0.31])
    assert living(True).surface_at(seat).name == "couch_seat"
    assert living(False).surface_at(seat) is None


def test_beam_through_head_height_is_blocked():
    z = living(True)
    # From the desk, straight across at 1.0 m over the couch: through where heads are.
    assert z.beam_crosses_no_aim([-0.3, 0.7, 1.0], [2.2, 1.2, 1.0]) == "couch_people"


def test_low_beam_onto_seat_is_allowed_with_couch_play():
    z = living(True)
    assert z.beam_crosses_no_aim([-0.3, 0.7, 0.8], [2.2, 1.2, 0.31]) is None


def test_whole_couch_blocked_without_couch_play():
    z = living(False)
    assert z.beam_crosses_no_aim([-0.3, 0.7, 0.8], [2.2, 1.2, 0.31]) == "couch_people"


def test_rug_is_floor_surface():
    assert living(True).surface_at([0.8, 1.2, 0.0]).name == "floor"
