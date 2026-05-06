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

from .batch_splat import (
    BatchSplatConfig,
    BatchSplatRenderer,
    IsaacBatchSplatRenderer,
    MjxBatchSplatRenderer,
    MtxBatchSplatRenderer,
)
from .core.batch_rasterization import batch_env_render, batch_render, batch_update_gaussians
from .core.gaussiandata import GaussianBatchData, GaussianData
from .core.gs_renderer import GSRenderer
from .core.super_splat_loader import is_super_splat_format, load_super_splat_ply, save_super_splat_ply
from .core.util_gau import load_ply, save_ply, transform_shs
from .gs_renderer_isaacgym import GSRendererIsaacGym
from .gs_renderer_motrixsim import GSRendererMotrixSim
from .gs_renderer_mujoco import GSRendererMuJoCo

__version__ = "0.2.0"
__author__ = "Yufei Jia"
