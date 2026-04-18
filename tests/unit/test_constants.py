from road_damage.common.constants import CLASS_ID_TO_NAME, assert_class_map_invariant


def test_class_map_invariant() -> None:
    assert CLASS_ID_TO_NAME == {0: "crack", 1: "pothole"}
    assert_class_map_invariant()
