from wc2026_groups import GROUPS


def test_twelve_groups_of_four():
    assert len(GROUPS) == 12
    assert all(len(v) == 4 for v in GROUPS.values())


def test_forty_eight_unique_teams():
    teams = [t for g in GROUPS.values() for t in g]
    assert len(teams) == 48
    assert len(set(teams)) == 48
