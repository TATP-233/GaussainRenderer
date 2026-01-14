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
from typing import Tuple, List, Union, Dict, Optional, Any, TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation
from torch import Tensor

try:
    import motrixsim  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    motrixsim = None  # type: ignore
    _MOTRIXSIM_IMPORT_ERROR = exc
else:
    _MOTRIXSIM_IMPORT_ERROR = None

if TYPE_CHECKING:
    import motrixsim
from .src.gs_renderer import GSRenderer

class GSRendererMotrixSim(GSRenderer):
    def __init__(self, models_dict: Dict[str, str], mx_model:"motrixsim.MotrixSimModel") -> None:
        if motrixsim is None:
            raise ImportError(
                "MotrixSim is not installed. Install the motrixsim package to use GSRendererMotrixSim."
            ) from _MOTRIXSIM_IMPORT_ERROR
        super().__init__(models_dict)
        self.init_renderer(mx_model)

    def init_renderer(self, mx_model:"motrixsim.MotrixSimModel") -> None:
        self._mx_model = mx_model

        self.gs_idx_start = []
        self.gs_idx_end = []
        self.gs_body_ids = []
        
        objects_info = []

        for i, link_name in enumerate(mx_model.link_names):
            if link_name in self.gaussian_model_names:
                start_idx = self.gaussian_start_indices[link_name]
                end_idx = self.gaussian_end_indices[link_name]
                self.gs_idx_start.append(start_idx)
                self.gs_idx_end.append(end_idx)
                self.gs_body_ids.append(i)
                objects_info.append((link_name, start_idx, end_idx))

        self.gs_idx_start = np.array(self.gs_idx_start)
        self.gs_idx_end = np.array(self.gs_idx_end)
        self.gs_body_ids = np.array(self.gs_body_ids)
        
        # Call the generic mapping method in base class
        self.set_objects_mapping(objects_info)

    def update_gaussians(self, mx_data: "motrixsim.SceneData") -> None:
        if not hasattr(self, 'gs_idx_start') or len(self.gs_idx_start) == 0:
            return
        
        if not hasattr(self, "_mx_model") or self._mx_model is None:
            raise RuntimeError("MotrixSim model is not initialized in the renderer, call init_renderer first.")

        link_poses = self._mx_model.get_link_poses(mx_data)

        # Batch extract position (N, 3)
        pos_values = link_poses[self.gs_body_ids, :3]
        
        # Batch extract quaternion (N, 4) - wxyz
        quat_values = link_poses[self.gs_body_ids, 3:7]
        
        # Call batch update interface
        self.update_gaussian_properties(
            pos_values,
            quat_values,
            scalar_first=False
        )

    def render(self, 
               mx_model: "motrixsim.MotrixSimModel",
               mx_data: "motrixsim.SceneData",
               cam_ids:Union[List[int], np.ndarray], 
               width:int, height:int) -> Dict[int, Tuple[Tensor, Tensor]]:
        if -1 in cam_ids:
            raise NotImplementedError("Free camera rendering not supported in MotrixSim bridge.")

        cam_pos_lst = []
        cam_xmat_lst = []
        fovy_lst = []
        for cid in cam_ids:
            # assert cid < len(mx_model.cameras), f"Camera ID {cid} out of range for MotrixSim model with {len(mx_model.cameras)} cameras."
            cam = mx_model.cameras[cid]
            cam_pose = cam.get_pose(mx_data)
            cam_pos_lst.append(cam_pose[:3])
            cam_xmat_lst.append(Rotation.from_quat(cam_pose[3:7]).as_matrix().flatten())
            fovy_lst.append(mx_model.cameras[cid].fovy) # TODO: get actual fovy from MotrixSim camera

        rgb_tensor, depth_tensor = self.render_batch(
            np.array(cam_pos_lst),
            np.array(cam_xmat_lst),
            height,
            width,
            np.array(fovy_lst)
        )
        
        batch_indices = {cid: i for i, cid in enumerate(cam_ids)}
        
        results = {}
        for cid, idx in batch_indices.items():
            results[cid] = (rgb_tensor[idx], depth_tensor[idx])
        
        return results
