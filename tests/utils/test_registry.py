"""
Tests for utils/registry.py.

Coverage
--------
register / get round-trip; case-insensitivity; duplicate raises; unknown name
raises (lists available); unknown kwargs warn and are dropped; describe/names;
membership and item access; **kwargs factories accept anything.
"""

import pytest

from utils.registry import Registry


def _fresh():
    r = Registry("Widget")

    @r.register("alpha", description="the first")
    def _alpha(scale=1.0):
        return ("alpha", scale)

    @r.register("Beta")
    def _beta(**kw):
        return ("beta", kw)

    return r


class TestRegistry:

    def test_get_round_trip(self):
        r = _fresh()
        assert r.get("alpha", scale=2.0) == ("alpha", 2.0)

    def test_case_insensitive(self):
        r = _fresh()
        assert r.get("ALPHA")[0] == "alpha"
        assert r.get("beta")[0] == "beta"

    def test_duplicate_raises(self):
        r = _fresh()
        with pytest.raises(ValueError, match="already registered"):
            @r.register("alpha")
            def _dup():
                ...

    def test_unknown_name_raises_with_available(self):
        r = _fresh()
        with pytest.raises(ValueError, match="not registered"):
            r.get("missing")

    def test_unknown_kwargs_warn_and_drop(self):
        r = _fresh()
        with pytest.warns(UserWarning, match="unknown kwargs"):
            out = r.get("alpha", bogus=123)
        assert out == ("alpha", 1.0)   # dropped, default used

    def test_var_keyword_factory_accepts_anything(self):
        r = _fresh()
        # _beta takes **kw -> no kwarg is "unknown"
        out = r.get("beta", anything=1, more=2)
        assert out == ("beta", {"anything": 1, "more": 2})

    def test_describe_and_names(self):
        r = _fresh()
        assert r.describe()["ALPHA"] == "the first"
        assert r.names() == ["ALPHA", "BETA"]

    def test_membership_and_getitem_and_len(self):
        r = _fresh()
        assert "alpha" in r and "ALPHA" in r and "zeta" not in r
        assert callable(r["alpha"])
        assert len(r) == 2
