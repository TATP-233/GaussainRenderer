"""Tests for GSRendererMuJoCo — all CPU-only, no CUDA, no real mujoco binary."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from conftest import BANANA_PLY

from gaussian_renderer.gs_renderer_mujoco import GSRendererMuJoCo

mujoco_stub = sys.modules["mujoco"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mj_model(body_names):
    model = MagicMock()
    model.nbody = len(body_names)
    model.cam_fovy = np.array([45.0])
    model.vis.global_.fovy = 45.0
    mujoco_stub.mj_id2name.side_effect = lambda m, t, i: body_names[i] if i < len(body_names) else ""
    return model


def _make_mj_data(n_bodies, n_cams=1):
    data = MagicMock()
    data.xpos = np.zeros((n_bodies, 3))
    data.xquat = np.tile([1.0, 0.0, 0.0, 0.0], (n_bodies, 1))
    data.cam_xpos = np.zeros((n_cams, 3))
    data.cam_xmat = np.tile(np.eye(3).flatten(), (n_cams, 1))
    return data


def _make_renderer(body_names, models_dict):
    mj_model = _make_mj_model(body_names)
    with patch("gaussian_renderer.core.gs_renderer.GSPLAT_AVAILABLE", True), \
            patch("torch.Tensor.cuda", lambda self: self), \
            patch("gaussian_renderer.gs_renderer_mujoco.mujoco", mujoco_stub):
        from gaussian_renderer.core.gs_renderer import GSRenderer

        renderer = GSRendererMuJoCo.__new__(GSRendererMuJoCo)
        GSRenderer.__init__(renderer, models_dict)
        renderer.init_renderer(mj_model)
    return renderer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_import_error_without_mujoco():
    """ImportError raised when mujoco is not installed."""
    import gaussian_renderer.gs_renderer_mujoco as mod

    orig, orig_err = mod.mujoco, mod._MUJOCO_IMPORT_ERROR
    mod.mujoco = None
    mod._MUJOCO_IMPORT_ERROR = ImportError("no mujoco")
    try:
        with pytest.raises(ImportError, match="MuJoCo is not installed"):
            GSRendererMuJoCo.__new__(GSRendererMuJoCo).__init__({"banana": str(BANANA_PLY)}, MagicMock())
    finally:
        mod.mujoco = orig
        mod._MUJOCO_IMPORT_ERROR = orig_err


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_init_renderer_maps_bodies():
    """init_renderer maps matching body names to Gaussian indices."""
    renderer = _make_renderer(["world", "banana"], {"banana": str(BANANA_PLY)})

    assert len(renderer.gs_body_ids) == 1
    assert renderer.gs_body_ids[0] == 1
    assert renderer.dynamic_mask.any()


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_init_renderer_no_matching_bodies():
    """No matching bodies → empty arrays, dynamic_mask all False."""
    renderer = _make_renderer(["world", "robot"], {"banana": str(BANANA_PLY)})

    assert len(renderer.gs_body_ids) == 0
    assert not renderer.dynamic_mask.any()


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_update_gaussians_passes_correct_slices():
    """update_gaussians extracts the right body rows from mj_data."""
    renderer = _make_renderer(["world", "banana"], {"banana": str(BANANA_PLY)})

    mj_data = _make_mj_data(n_bodies=2)
    mj_data.xpos[1] = [1.0, 2.0, 3.0]

    with patch.object(renderer, "update_gaussian_properties") as mock_upd:
        renderer.update_gaussians(mj_data)
        mock_upd.assert_called_once()
        pos_arg, quat_arg = mock_upd.call_args[0]
        np.testing.assert_array_equal(pos_arg, mj_data.xpos[[1]])
        np.testing.assert_array_equal(quat_arg, mj_data.xquat[[1]])


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_update_gaussians_empty_skips():
    """update_gaussians with no mapped bodies returns without error."""
    renderer = _make_renderer(["world"], {"banana": str(BANANA_PLY)})
    renderer.update_gaussians(_make_mj_data(n_bodies=1))  # should not raise
