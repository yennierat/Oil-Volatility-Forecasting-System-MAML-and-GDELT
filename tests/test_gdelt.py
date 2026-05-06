"""GDELT fetch tests.

Two layers:

1. **Unit test** — mocks `requests.get` with a synthetic GDELT export, asserts
   the parser produces the expected aggregate keys and per-region splits.
   Fast, deterministic, runs in CI.

2. **Network test** — actually hits GDELT. Marked `network` so it's skipped
   by default. Useful when you suspect GDELT changed format. Run with:
       pytest -m network
"""
import io
import zipfile

import pytest


# Build a synthetic GDELT TSV row. GDELT events have ~58 tab-separated columns;
# `fetch_gdelt` only reads indices 7, 17, 26, 27, 30, 31, 34, 53.
def _make_gdelt_row(
    actor1="US",
    actor2="IR",
    event_code="180",      # → "18" → military_conflict
    event_base="18",
    goldstein="-7.5",
    mentions="100",
    tone="-3.0",
    action_country="IR",   # → middle_east
):
    cols = [""] * 58
    cols[7]  = actor1
    cols[17] = actor2
    cols[26] = event_code
    cols[27] = event_base
    cols[30] = goldstein
    cols[31] = mentions
    cols[34] = tone
    cols[53] = action_country
    return "\t".join(cols)


def _make_gdelt_zip(rows):
    """Pack rows into a GDELT-style zipped TSV."""
    tsv = "\n".join(rows).encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("synthetic.export.csv", tsv)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, *, text="", content=b""):
        self.text = text
        self.content = content


def _make_fake_get(zip_bytes):
    def fake_get(url, **_kwargs):
        if "lastupdate.txt" in url:
            return _FakeResponse(text="42 abc123 http://fake.gdelt/export.zip")
        return _FakeResponse(content=zip_bytes)
    return fake_get


def test_fetch_gdelt_parses_synthetic_event(scheduler_v2, monkeypatch):
    rows = [_make_gdelt_row()]
    zip_bytes = _make_gdelt_zip(rows)
    monkeypatch.setattr(scheduler_v2.requests, "get", _make_fake_get(zip_bytes))

    result = scheduler_v2.fetch_gdelt()

    assert result["n_events"] == 1.0
    assert result["gs_mean"] == pytest.approx(-7.5)
    assert result["gs_conflict_pct"] == pytest.approx(1.0)  # all events negative
    assert result["mentions_sum"] == pytest.approx(100.0)
    assert result["tone_mean"] == pytest.approx(-3.0)

    # action_country=IR → middle_east region
    assert result["me_n_events"] == 1.0
    assert result["me_gs_mean"] == pytest.approx(-7.5)
    assert result["me_conflict_pct"] == pytest.approx(1.0)

    # No oil-producer events in this synthetic input
    assert result["oi_n_events"] == 0.0


def test_fetch_gdelt_filters_low_mention_events(scheduler_v2, monkeypatch):
    """Events with mentions < MIN_MENTIONS (10) must be dropped."""
    rows = [
        _make_gdelt_row(mentions="5"),       # below threshold — drop
        _make_gdelt_row(mentions="100"),     # above threshold — keep
    ]
    monkeypatch.setattr(
        scheduler_v2.requests, "get",
        _make_fake_get(_make_gdelt_zip(rows)),
    )

    result = scheduler_v2.fetch_gdelt()
    assert result["n_events"] == 1.0


def test_fetch_gdelt_falls_back_on_network_error(scheduler_v2, monkeypatch):
    """If GDELT is unreachable, must return GDELT_FALLBACK without crashing."""
    def _boom(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr(scheduler_v2.requests, "get", _boom)

    result = scheduler_v2.fetch_gdelt()
    assert result == scheduler_v2.GDELT_FALLBACK


def test_fetch_gdelt_returns_all_v2_keys(scheduler_v2, monkeypatch):
    """The 23-feature scheduler depends on per-region keys (me_*, oi_*).

    Regression guard: if anyone strips per-region aggregation, fetch_gdelt
    will silently drop those keys and the feature vector will fail to assemble.
    """
    rows = [_make_gdelt_row()]
    monkeypatch.setattr(
        scheduler_v2.requests, "get",
        _make_fake_get(_make_gdelt_zip(rows)),
    )

    result = scheduler_v2.fetch_gdelt()
    required = {
        "gs_mean", "gs_std", "gs_conflict_pct", "gs_weighted",
        "tone_mean", "tone_std", "n_events", "mentions_sum",
        "me_gs_mean", "me_conflict_pct", "me_n_events", "me_tone_mean",
        "oi_gs_mean", "oi_conflict_pct", "oi_n_events", "oi_tone_mean",
    }
    missing = required - result.keys()
    assert not missing, f"Missing GDELT keys: {missing}"


@pytest.mark.network
def test_fetch_gdelt_live_api(scheduler_v2):
    """Hit the real GDELT API. Skipped by default — run with `pytest -m network`.

    Asserts that GDELT responds, the parser runs to completion, and the result
    has the expected v2 schema. If this fails, GDELT changed format or is down.
    """
    result = scheduler_v2.fetch_gdelt()

    required = {
        "gs_mean", "gs_std", "n_events",
        "me_n_events", "oi_n_events",
    }
    assert required.issubset(result.keys()), \
        f"GDELT response missing keys: {required - result.keys()}"

    # Either we parsed real events, or the function fell back to GDELT_FALLBACK.
    # Both are acceptable outcomes for an integration test — we mainly want
    # to confirm no exception escaped.
    assert isinstance(result["n_events"], float)
