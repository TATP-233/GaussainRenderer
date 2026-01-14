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

"""
Minimal batch Gaussian splat renderer that works with mjx state tensors.

Usage pattern (3 steps):
1) init: cfg = BatchSplatConfig(...); renderer = BatchSplatRenderer(cfg)
2) loop: gsb = renderer.batch_update_gaussians(body_pos, body_quat)
3) loop: rgb, depth = renderer.batch_env_render(gsb, cam_pos, cam_xmat, height, width, fovy)

Where body_pos/body_quat/cam_pos/cam_xmat can come from mjx (MuJoCo JAX) state via
`torch.utils.dlpack.from_dlpack(state.data.xxx)`.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from .src.gaussiandata import GaussianData, GaussianBatchData
from .src.util_gau import load_ply
from .src.batch_rasterization import (
    batch_env_render as _batch_env_render,
    batch_update_gaussians as _batch_update_gaussians,
)


@dataclass
class BatchSplatConfig:
    # Mapping from MuJoCo body name -> PLY path (local coordinate frame).
    body_gaussians: Dict[str, str]
    # Optional static/background PLY that is not attached to any body.
    background_ply: Optional[str] = None
    # Device to place tensors on; if None, will pick CUDA if available else CPU.
    device: Optional[torch.device] = None
    # Minibatch size for rendering
    minibatch: Optional[int] = None


class BatchSplatRenderer:
    def __init__(self, cfg: BatchSplatConfig, body_name_to_id: Optional[Dict[str, int]]) -> None:
        assert torch.cuda.is_available()
        device = cfg.device or torch.device("cuda")

        self.device = device
        self.minibatch = cfg.minibatch

        xyz_list: List[Tensor] = []
        rot_list: List[Tensor] = []
        scale_list: List[Tensor] = []
        opacity_list: List[Tensor] = []
        sh_list: List[Tensor] = []
        max_sh_dim = 0

        self.gs_idx_start: List[int] = []
        self.gs_idx_end: List[int] = []
        self.gs_body_ids: List[int] = []

        # Load per-body gaussians
        for body_name, ply_path in cfg.body_gaussians.items():
            g = load_ply(ply_path)
            max_sh_dim = max(max_sh_dim, g.sh.shape[1])
            start = len(torch.cat(xyz_list)) if xyz_list else 0
            end = start + len(g.xyz)
            self.gs_idx_start.append(start)
            self.gs_idx_end.append(end)
            if body_name not in body_name_to_id:
                raise ValueError(f"Body '{body_name}' not found in mj_model; available: {list(body_name_to_id.keys())}")
            self.gs_body_ids.append(body_name_to_id[body_name])
            xyz_list.append(torch.tensor(g.xyz, device=device, dtype=torch.float32))
            rot_list.append(torch.tensor(g.rot, device=device, dtype=torch.float32))
            scale_list.append(torch.tensor(g.scale, device=device, dtype=torch.float32))
            opacity_list.append(torch.tensor(g.opacity, device=device, dtype=torch.float32))
            sh_list.append(torch.tensor(g.sh, device=device, dtype=torch.float32))

        # Optional background/static gaussian
        if cfg.background_ply:
            g = load_ply(cfg.background_ply)
            max_sh_dim = max(max_sh_dim, g.sh.shape[1])
            xyz_list.append(torch.tensor(g.xyz, device=device, dtype=torch.float32))
            rot_list.append(torch.tensor(g.rot, device=device, dtype=torch.float32))
            scale_list.append(torch.tensor(g.scale, device=device, dtype=torch.float32))
            opacity_list.append(torch.tensor(g.opacity, device=device, dtype=torch.float32))
            sh_list.append(torch.tensor(g.sh, device=device, dtype=torch.float32))

        if sh_list:
            for i, sh in enumerate(sh_list):
                cur_dim = sh.shape[1]
                if cur_dim < max_sh_dim:
                    pad = torch.zeros(sh.shape[0], max_sh_dim - cur_dim, device=sh.device, dtype=sh.dtype)
                    sh_list[i] = torch.cat([sh, pad], dim=1)

        # Concatenate template
        xyz_all = torch.cat(xyz_list, dim=0)
        rot_all = torch.cat(rot_list, dim=0)
        scale_all = torch.cat(scale_list, dim=0)
        opacity_all = torch.cat(opacity_list, dim=0)
        sh_all = torch.cat(sh_list, dim=0).reshape(xyz_all.shape[0], -1, 3).contiguous()

        self.template = GaussianData(
            xyz=xyz_all,
            rot=rot_all,
            scale=scale_all,
            opacity=opacity_all,
            sh=sh_all,
        )

        self.gs_idx_start = torch.tensor(self.gs_idx_start, dtype=torch.long, device=device)
        self.gs_idx_end = torch.tensor(self.gs_idx_end, dtype=torch.long, device=device)
        self.gs_body_ids = torch.tensor(self.gs_body_ids, dtype=torch.long, device=device)

        # Precompute point-to-body mapping for vectorized update
        num_points = len(self.template)
        self.dynamic_mask = torch.zeros(num_points, dtype=torch.bool, device=device)
        self.point_to_body_idx = torch.zeros(num_points, dtype=torch.long, device=device)
        
        for i in range(len(self.gs_idx_start)):
            start = self.gs_idx_start[i]
            end = self.gs_idx_end[i]
            body_id = self.gs_body_ids[i]
            self.dynamic_mask[start:end] = True
            self.point_to_body_idx[start:end] = body_id

    def batch_update_gaussians(self, body_pos: Tensor, body_quat: Tensor, scalar_first=True) -> GaussianBatchData:
        """Update gaussians using body poses.

        Args:
            body_pos: (Nenv, Nbody, 3) torch tensor
            body_quat: (Nenv, Nbody, 4) torch tensor (wxyz)
        Returns:
            GaussianBatchData with per-env gaussians.
        """
        # Ensure device
        if not isinstance(body_pos, torch.Tensor):
            body_pos = torch.tensor(body_pos, device=self.device, dtype=torch.float32)
        if not isinstance(body_quat, torch.Tensor):
            body_quat = torch.tensor(body_quat, device=self.device, dtype=torch.float32)
        if not body_pos.device == self.device:
            body_pos = body_pos.to(self.device)
        if not body_quat.device == self.device:
            body_quat = body_quat.to(self.device)
        return _batch_update_gaussians(
            self.template,
            body_pos,
            body_quat,
            point_to_body_idx=self.point_to_body_idx,
            dynamic_mask=self.dynamic_mask,
            scalar_first=scalar_first
        )

    def batch_env_render(
        self,
        gsb: GaussianBatchData,
        cam_pos: Tensor,
        cam_xmat: Tensor,
        height: int,
        width: int,
        fovy: np.ndarray,
        bg_imgs: Optional[Tensor] = None,
    ):
        """Render RGBD for batch envs and cameras."""
        if not isinstance(cam_pos, torch.Tensor):
            cam_pos = torch.tensor(cam_pos, device=self.device, dtype=torch.float32)
        if not isinstance(cam_xmat, torch.Tensor):
            cam_xmat = torch.tensor(cam_xmat, device=self.device, dtype=torch.float32)
        if not cam_pos.device == self.device:
            cam_pos = cam_pos.to(self.device)
        if not cam_xmat.device == self.device:
            cam_xmat = cam_xmat.to(self.device)
        return _batch_env_render(gsb, cam_pos, cam_xmat, height, width, fovy, bg_imgs=bg_imgs, minibatch=self.minibatch)

if TYPE_CHECKING:
    import mujoco

class MjxBatchSplatRenderer(BatchSplatRenderer):
    def __init__(self, cfg: BatchSplatConfig, mj_model:"mujoco.MjModel") -> None:
        body_name_to_id = {}
        for i in range(mj_model.nbody):
            body_name = mj_model.body(i).name
            if body_name:
                body_name_to_id[body_name] = i
        super().__init__(cfg, body_name_to_id)

if TYPE_CHECKING:
    import motrixsim

class MtxBatchSplatRenderer(BatchSplatRenderer):
    def __init__(self, cfg, mx_model:"motrixsim.MotrixSimModel") -> None:
        body_name_to_id = {}
        for i, link_name in enumerate(mx_model.link_names):
            if link_name:
                body_name_to_id[link_name] = i
        super().__init__(cfg, body_name_to_id)
    
    def batch_update_gaussians(self, body_pos: Tensor, body_quat: Tensor) -> GaussianBatchData:
        return super().batch_update_gaussians(body_pos, body_quat, scalar_first=False)

__all__ = ["BatchSplatConfig", "BatchSplatRenderer", "MjxBatchSplatRenderer", "MtxBatchSplatRenderer"]