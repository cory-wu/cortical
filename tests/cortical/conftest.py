"""Shared fixtures: a 10x10 sheet with a scaled-down patch layout."""

import pytest

SMALL_L = 10


@pytest.fixture
def small_layout():
    return {
        "fixation": ((2.5, 7.5), 2),
        "ring1": ((2.5, 2.5), 4),
        "ring2": ((7.5, 7.5), 4),
        "rule": ((7.5, 2.5), 4),
    }
