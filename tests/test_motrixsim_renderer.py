"""Tests for GSRendererMotrixSim — CPU-only, no real motrixsim binary."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from conftest import BANANA_PLY

motrixsim_stub = sys.modules["motrixsim"]

from gaussian_renderer.gs_renderer_motrixsim import GSRendererMotrixSim  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mx_model(link_names):
    model = MagicMock()
    model.link_names = link_names
    return model


def _make_renderer(link_names, ply_path):
    models_dict = {name: str(ply_path) for name in link_names if name != "world"}
    mx_model = _make_mx_model(link_names)

    with patch("gaussian_renderer.core.gs_renderer.GSPLAT_AVAILABLE", True), \
            patch("torch.Tensor.cuda", lambda self: self):
        from gaussian_renderer.core.gs_renderer import GSRenderer

        renderer = GSRendererMotrixSim.__new__(GSRendererMotrixSim)
        GSRenderer.__init__(renderer, models_dict)
        renderer.init_renderer(mx_model)
    return renderer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_import_error_without_motrixsim():
    """ImportError raised when motrixsim is not installed."""
    import gaussian_renderer.gs_renderer_motrixsim as mod

    orig = mod.motrixsim
    mod.motrixsim = None
    mod._MOTRIXSIM_IMPORT_ERROR = ImportError("no motrixsim")
    try:
        with pytest.raises(ImportError, match="MotrixSim is not installed"):
            GSRendererMotrixSim.__new__(GSRendererMotrixSim).__init__({"banana": str(BANANA_PLY)}, MagicMock())
    finally:
        mod.motrixsim = orig
        mod._MOTRIXSIM_IMPORT_ERROR = None


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_init_renderer_maps_links():
    """init_renderer maps matching link names to Gaussian indices."""
    link_names = ["world", "banana"]
    renderer = _make_renderer(link_names, BANANA_PLY)

    assert len(renderer.gs_body_ids) == 1
    assert renderer.gs_body_ids[0] == 1  # "banana" is link index 1
    assert renderer.dynamic_mask is not None
    assert renderer.dynamic_mask.any()


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_init_renderer_no_matching_links():
    """No matching links → empty arrays, dynamic_mask all False."""
    link_names = ["world", "robot"]
    models_dict = {"banana": str(BANANA_PLY)}
    mx_model = _make_mx_model(link_names)

    with patch("gaussian_renderer.core.gs_renderer.GSPLAT_AVAILABLE", True), \
            patch("torch.Tensor.cuda", lambda self: self):
        from gaussian_renderer.core.gs_renderer import GSRenderer

        renderer = GSRendererMotrixSim.__new__(GSRendererMotrixSim)
        GSRenderer.__init__(renderer, models_dict)
        renderer.init_renderer(mx_model)

    assert len(renderer.gs_body_ids) == 0
    assert not renderer.dynamic_mask.any()


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_update_gaussians_passes_correct_slices():
    """update_gaussians extracts the right link rows and converts xyzw→wxyz."""
    link_names = ["world", "banana"]
    renderer = _make_renderer(link_names, BANANA_PLY)

    # link_poses: (2, 7) — pos(3) + quat_xyzw(4)
    link_poses = np.zeros((2, 7))
    link_poses[1] = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]  # xyzw identity at pos (1,2,3)
    renderer._mx_model.get_link_poses.return_value = link_poses

    with patch.object(renderer, "update_gaussian_properties") as mock_upd:
        renderer.update_gaussians(MagicMock())
        mock_upd.assert_called_once()
        pos_arg, quat_arg = mock_upd.call_args[0]
        kwargs = mock_upd.call_args[1]
        np.testing.assert_array_equal(pos_arg, link_poses[[1], :3])
        np.testing.assert_array_equal(quat_arg, link_poses[[1], 3:7])
        assert kwargs.get("scalar_first") is False


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_update_gaussians_empty_skips():
    """update_gaussians with no mapped links returns without error."""
    link_names = ["world"]
    models_dict = {"banana": str(BANANA_PLY)}
    mx_model = _make_mx_model(link_names)

    with patch("gaussian_renderer.core.gs_renderer.GSPLAT_AVAILABLE", True), \
            patch("torch.Tensor.cuda", lambda self: self):
        from gaussian_renderer.core.gs_renderer import GSRenderer

        renderer = GSRendererMotrixSim.__new__(GSRendererMotrixSim)
        GSRenderer.__init__(renderer, models_dict)
        renderer.init_renderer(mx_model)

    renderer.update_gaussians(MagicMock())  # should not raise
