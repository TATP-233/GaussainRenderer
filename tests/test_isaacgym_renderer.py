"""Tests for GSRendererIsaacGym — CPU-only, no real CUDA or Isaac Gym binary."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from conftest import BANANA_PLY

from gaussian_renderer.gs_renderer_isaacgym import GSRendererIsaacGym

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gym(body_name_to_id):
    gym = MagicMock()
    gym.get_actor_rigid_body_dict.return_value = dict(body_name_to_id)
    return gym


def _make_renderer(body_name_to_id, models_dict):
    gym = _make_gym(body_name_to_id)
    env = MagicMock()
    actor_handle = 0
    with (
        patch("gaussian_renderer.core.gs_renderer.GSPLAT_AVAILABLE", True),
        patch("torch.Tensor.cuda", lambda self: self),
    ):
        from gaussian_renderer.core.gs_renderer import GSRenderer

        renderer = GSRendererIsaacGym.__new__(GSRendererIsaacGym)
        GSRenderer.__init__(renderer, models_dict)
        renderer.init_renderer(gym, env, actor_handle)
    return renderer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_init_renderer_maps_bodies():
    """init_renderer maps matching body names to gaussian indices."""
    renderer = _make_renderer({"world": 0, "banana": 1}, {"banana": str(BANANA_PLY)})

    assert len(renderer.gs_body_ids) == 1
    assert renderer.gs_body_ids[0] == 1
    assert renderer.dynamic_mask.any()


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_init_renderer_queries_actor_rigid_body_dict():
    """init_renderer pulls body mapping from gym.get_actor_rigid_body_dict."""
    gym = _make_gym({"world": 0, "banana": 1})
    env = MagicMock()
    actor = 7
    with (
        patch("gaussian_renderer.core.gs_renderer.GSPLAT_AVAILABLE", True),
        patch("torch.Tensor.cuda", lambda self: self),
    ):
        from gaussian_renderer.core.gs_renderer import GSRenderer

        renderer = GSRendererIsaacGym.__new__(GSRendererIsaacGym)
        GSRenderer.__init__(renderer, {"banana": str(BANANA_PLY)})
        renderer.init_renderer(gym, env, actor)

    gym.get_actor_rigid_body_dict.assert_called_once_with(env, actor)


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_init_renderer_no_matching_bodies():
    """No matching bodies → empty arrays, dynamic_mask all False."""
    renderer = _make_renderer({"world": 0, "robot": 1}, {"banana": str(BANANA_PLY)})

    assert len(renderer.gs_body_ids) == 0
    assert not renderer.dynamic_mask.any()


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_update_gaussians_passes_correct_slices():
    """update_gaussians extracts the right body rows and forwards xyzw."""
    renderer = _make_renderer({"world": 0, "banana": 1}, {"banana": str(BANANA_PLY)})

    body_pos = np.zeros((2, 3))
    body_pos[1] = [1.0, 2.0, 3.0]
    body_quat = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))

    with patch.object(renderer, "update_gaussian_properties") as mock_upd:
        renderer.update_gaussians(body_pos, body_quat)
        mock_upd.assert_called_once()
        pos_arg, quat_arg = mock_upd.call_args[0]
        kwargs = mock_upd.call_args[1]
        np.testing.assert_array_equal(pos_arg, body_pos[[1]])
        np.testing.assert_array_equal(quat_arg, body_quat[[1]])
        assert kwargs.get("scalar_first") is False


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_update_gaussians_empty_skips():
    """update_gaussians with no mapped bodies returns without error."""
    renderer = _make_renderer({"world": 0}, {"banana": str(BANANA_PLY)})
    renderer.update_gaussians(np.zeros((1, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (1, 1)))  # should not raise


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_render_calls_render_batch_with_xyzw_to_xmat():
    """render() converts cam_quat (xyzw) to a flattened rotation matrix."""
    renderer = _make_renderer({"world": 0, "banana": 1}, {"banana": str(BANANA_PLY)})

    cam_pos = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    cam_quat = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    with patch.object(renderer, "render_batch", return_value=("rgb", "depth")) as mock_rb:
        rgb, depth = renderer.render(cam_pos, cam_quat, width=64, height=48, fovy=45.0)

    assert (rgb, depth) == ("rgb", "depth")
    mock_rb.assert_called_once()
    args, kwargs = mock_rb.call_args
    np.testing.assert_array_equal(args[0], cam_pos)
    expected_xmat = np.eye(3).reshape(1, 9)
    np.testing.assert_allclose(args[1], expected_xmat, atol=1e-6)
    assert args[2] == 48 and args[3] == 64
    np.testing.assert_array_equal(args[4], np.array([45.0], dtype=np.float32))
    assert kwargs.get("y_up") is True
