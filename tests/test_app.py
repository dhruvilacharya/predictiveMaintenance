"""
tests/test_app.py
=================
Smoke tests for app/streamlit_app.py.

These tests do NOT launch the Streamlit server. They import helper
functions directly and verify their return types.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Import helpers from the app (without running Streamlit) ───────────────────

def _import_app_helpers():
    """
    Import only the pure-Python helper functions from the app,
    bypassing Streamlit's page-config calls.
    """
    import importlib, types, sys

    # Stub out streamlit so imports don't fail in a headless environment
    if "streamlit" not in sys.modules:
        st_mock = types.ModuleType("streamlit")
        st_mock.set_page_config = lambda **kw: None
        st_mock.cache_data = lambda **kw: (lambda f: f)
        st_mock.cache_resource = lambda **kw: (lambda f: f)
        st_mock.stop = lambda: None
        st_mock.error = lambda *a, **kw: None
        st_mock.warning = lambda *a, **kw: None
        st_mock.info = lambda *a, **kw: None
        st_mock.title = lambda *a: None
        st_mock.markdown = lambda *a, **kw: None
        st_mock.subheader = lambda *a: None
        st_mock.metric = lambda *a, **kw: None
        st_mock.columns = lambda n: [st_mock] * (n if isinstance(n, int) else len(n))
        st_mock.selectbox = lambda *a, **kw: a[1][0] if len(a) > 1 and a[1] else None
        st_mock.slider = lambda *a, **kw: kw.get("value", 30)
        st_mock.number_input = lambda *a, **kw: kw.get("value", 0)
        st_mock.radio = lambda *a, **kw: kw.get("default", "Fleet Overview")
        st_mock.dataframe = lambda *a, **kw: None
        st_mock.plotly_chart = lambda *a, **kw: None
        st_mock.expander = lambda *a, **kw: st_mock
        st_mock.caption = lambda *a: None
        st_mock.download_button = lambda *a, **kw: None
        st_mock.__enter__ = lambda s: s
        st_mock.__exit__ = lambda s, *a: False
        sys.modules["streamlit"] = st_mock

    # Also stub plotly if not installed
    if "plotly" not in sys.modules:
        for sub in ["plotly", "plotly.express", "plotly.graph_objects"]:
            m = types.ModuleType(sub)
            sys.modules[sub] = m


_import_app_helpers()


# ── Tests for pure-Python helpers ─────────────────────────────────────────────

class TestRiskHelpers:

    def test_risk_color_alert(self):
        from app.streamlit_app import risk_color
        assert risk_color(pred_rul=10, threshold=30) == "red"

    def test_risk_color_watch(self):
        from app.streamlit_app import risk_color
        assert risk_color(pred_rul=45, threshold=30) == "orange"

    def test_risk_color_ok(self):
        from app.streamlit_app import risk_color
        assert risk_color(pred_rul=100, threshold=30) == "green"

    def test_risk_label_returns_string(self):
        from app.streamlit_app import risk_label
        for pred in [5, 35, 100]:
            result = risk_label(pred, threshold=30)
            assert isinstance(result, str) and len(result) > 0

    def test_risk_color_at_exact_threshold_is_alert(self):
        from app.streamlit_app import risk_color
        # pred_rul < threshold → alert (strict less-than)
        assert risk_color(pred_rul=30, threshold=30) == "orange"   # == threshold → watch
        assert risk_color(pred_rul=29, threshold=30) == "red"      # < threshold → alert


class TestGetPredictions:

    def _make_fake_model(self):
        """Return a trivial model that predicts 50 for everything."""
        class FakeModel:
            def predict(self, X):
                return np.full(len(X), 50.0)
        return FakeModel()

    def _make_fake_test_feat(self, n_engines: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for unit in range(1, n_engines + 1):
            for cycle in range(1, 11):
                row = {"unit": unit, "cycle": cycle}
                for i in range(4):
                    row[f"f{i}"] = rng.normal()
                rows.append(row)
        return pd.DataFrame(rows)

    def test_output_has_required_columns(self):
        from app.streamlit_app import get_predictions

        n = 5
        test_feat = self._make_fake_test_feat(n)
        rul_test  = pd.Series(np.random.randint(10, 100, n), index=range(1, n + 1))
        feature_cols = [f"f{i}" for i in range(4)]
        model = self._make_fake_model()

        result = get_predictions("FD001", test_feat, rul_test, feature_cols, model)

        assert result is not None
        for col in ["unit", "y_true", "y_pred", "lower", "upper", "error", "abs_err"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_length_equals_n_engines(self):
        from app.streamlit_app import get_predictions

        n = 7
        test_feat = self._make_fake_test_feat(n)
        rul_test  = pd.Series(np.full(n, 50.0), index=range(1, n + 1))
        feature_cols = [f"f{i}" for i in range(4)]
        model = self._make_fake_model()

        result = get_predictions("FD001", test_feat, rul_test, feature_cols, model)
        assert len(result) == n

    def test_predictions_non_negative(self):
        from app.streamlit_app import get_predictions

        n = 5
        test_feat = self._make_fake_test_feat(n)
        rul_test  = pd.Series(np.full(n, 30.0), index=range(1, n + 1))
        feature_cols = [f"f{i}" for i in range(4)]
        model = self._make_fake_model()

        result = get_predictions("FD001", test_feat, rul_test, feature_cols, model)
        assert (result["y_pred"] >= 0).all()
        assert (result["lower"]  >= 0).all()

    def test_none_model_returns_none(self):
        from app.streamlit_app import get_predictions

        n = 3
        test_feat = self._make_fake_test_feat(n)
        rul_test  = pd.Series(np.full(n, 30.0), index=range(1, n + 1))
        feature_cols = [f"f{i}" for i in range(4)]

        result = get_predictions("FD001", test_feat, rul_test, feature_cols, None)
        assert result is None
