"""Registry contract for the 100 high-priority framework functions."""

from __future__ import annotations

import pytest

from framework_functions import (
    FRAME_TITLES,
    FUNCTIONS,
    get_function,
    iter_functions,
    validate_registry,
)


def test_registry_has_exactly_one_hundred_contiguous_functions():
    validate_registry()
    assert len(FUNCTIONS) == 100
    assert [item.number for item in FUNCTIONS] == list(range(1, 101))


@pytest.mark.parametrize("frame", range(1, 11))
def test_each_frame_contributes_ten_functions(frame: int):
    assert len(list(iter_functions(frame))) == 10
    assert frame in FRAME_TITLES


def test_names_are_unique_and_resolvable_by_number_and_name():
    names = {item.name for item in FUNCTIONS}
    assert len(names) == 100
    for item in FUNCTIONS:
        assert get_function(item.number) is item
        assert get_function(item.name) is item
        assert item.frame_title == FRAME_TITLES[item.frame]


def test_unknown_keys_raise_key_error():
    with pytest.raises(KeyError):
        get_function(0)
    with pytest.raises(KeyError):
        get_function(101)
    with pytest.raises(KeyError):
        get_function("definitely_not_a_framework_function")


def test_all_registered_functions_are_importable_callables():
    for item in FUNCTIONS:
        assert callable(item.callable), f"{item.number}: {item.name}"
