"""Which tasks each collection serves, and where they come from."""

import pytest

from cogtasks import COLLECTIONS, list_tasks, make_task
from cogtasks.modcog import _tasks as modcog_tasks

COG20 = (
    "go", "rtgo", "dlygo", "anti", "rtanti", "dlyanti",
    "dm1", "dm2", "ctxdm1", "ctxdm2", "multidm",
    "dlydm1", "dlydm2", "ctxdlydm1", "ctxdlydm2", "multidlydm",
    "dms", "dnms", "dmc", "dnmc",
)  # fmt: skip


def test_exactly_cog20_and_modcog_are_registered():
    assert set(COLLECTIONS) == {"cog20", "modcog"}


def test_cog20_lists_the_twenty_tasks_in_paper_order():
    assert list_tasks("cog20") == COG20


def test_modcog_is_cog20_plus_22_int_and_40_seq_variants():
    tasks = list_tasks("modcog")
    assert len(set(tasks)) == len(tasks) == 82
    assert tasks[:20] == COG20
    assert sum(t.endswith(("intr", "intl")) for t in tasks) == 22
    assert sum(t.endswith(("seqr", "seql")) for t in tasks) == 40


@pytest.mark.parametrize("collection", ["cog20", "modcog"])
def test_every_task_is_the_modcog_implementation(collection):
    for name, factory in COLLECTIONS[collection].items():
        assert factory is getattr(modcog_tasks, name)


def test_cog20_and_modcog_share_the_same_base_task_factories():
    for name in COG20:
        assert COLLECTIONS["cog20"][name] is COLLECTIONS["modcog"][name]


def test_unknown_collection_raises():
    with pytest.raises(ValueError, match="Unknown collection"):
        list_tasks("yang19")


def test_unknown_task_raises_with_collection_name():
    with pytest.raises(ValueError, match="modcog"):
        make_task("modcog", "nope")
