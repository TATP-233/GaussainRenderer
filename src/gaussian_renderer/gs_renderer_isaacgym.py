# SPDX-License-Identifier: MIT
#
# MIT License
#
# Copyright (c) 2025 Yufei Jia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import TYPE_CHECKING, Dict, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from isaacgym import gymapi  # type: ignore

from .core.gs_renderer import GSRenderer


class GSRendererIsaacGym(GSRenderer):
    def __init__(
        self,
        models_dict: Dict[str, str],
        gym: "gymapi.Gym",
        env: "gymapi.Env",
        actor_handle: int,
    ) -> None:
        super().__init__(models_dict)
        self.init_renderer(gym, env, actor_handle)

    def init_renderer(self, gym: "gymapi.Gym", env: "gymapi.Env", actor_handle: int) -> None:
        body_name_to_id: Dict[str, int] = gym.get_actor_rigid_body_dict(env, actor_handle)

        self.gs_idx_start = []
        self.gs_idx_end = []
        self.gs_body_ids = []

        objects_info = []
        for body_name, body_id in body_name_to_id.items():
            if body_name in self.gaussian_model_names:
                start_idx = self.gaussian_start_indices[body_name]
                end_idx = self.gaussian_end_indices[body_name]
                self.gs_idx_start.append(start_idx)
                self.gs_idx_end.append(end_idx)
                self.gs_body_ids.append(body_id)
                objects_info.append((body_name, start_idx, end_idx))

        self.gs_idx_start = np.array(self.gs_idx_start)
        self.gs_idx_end = np.array(self.gs_idx_end)
        self.gs_body_ids = np.array(self.gs_body_ids)

        self.set_objects_mapping(objects_info)

    def update_gaussians(self, body_pos: Union[np.ndarray, Tensor], body_quat: Union[np.ndarray, Tensor]) -> None:
        """Pull rigid body poses for the gaussian-bearing bodies and update.

        Args:
            body_pos: (Nbody, 3) world positions, indexable by body id.
            body_quat: (Nbody, 4) world quaternions in xyzw order (Isaac Gym convention).
        """
        if not hasattr(self, "gs_idx_start") or len(self.gs_idx_start) == 0:
            return

        pos_values = body_pos[self.gs_body_ids]
        quat_values = body_quat[self.gs_body_ids]

        self.update_gaussian_properties(pos_values, quat_values, scalar_first=False)

    def render(
        self,
        cam_pos: Union[np.ndarray, Tensor],
        cam_quat: Union[np.ndarray, Tensor],
        width: int,
        height: int,
        fovy: Union[float, np.ndarray],
        y_up: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """Render one batch of cameras given their world poses.

        Args:
            cam_pos: (Ncam, 3) camera world positions.
            cam_quat: (Ncam, 4) camera world quaternions in xyzw order.
            width, height: image size in pixels.
            fovy: scalar or (Ncam,) array of vertical FOV in degrees.
            y_up: forwarded to ``render_batch``; controls the OpenGL-style
                Y/Z flip applied to the camera rotation.
        Returns:
            ``(rgb, depth)`` tensors of shape ``(Ncam, H, W, 3)`` and
            ``(Ncam, H, W, 1)``.
        """
        cam_pos_arr = np.asarray(cam_pos).reshape(-1, 3)
        cam_quat_arr = np.asarray(cam_quat).reshape(-1, 4)
        cam_xmat = Rotation.from_quat(cam_quat_arr).as_matrix().reshape(-1, 9)
        fovy_arr = np.asarray(fovy, dtype=np.float32).reshape(-1)
        if fovy_arr.shape[0] == 1 and cam_pos_arr.shape[0] > 1:
            fovy_arr = np.broadcast_to(fovy_arr, (cam_pos_arr.shape[0],)).copy()

        return self.render_batch(cam_pos_arr, cam_xmat, height, width, fovy_arr, y_up=y_up)
