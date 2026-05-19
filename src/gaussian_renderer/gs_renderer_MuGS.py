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
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

try:
    from motrixsim import SceneData, forward_kinematic, msd  # type: ignore
    from motrixsim.render import RenderApp, RenderSettings  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    SceneData = None  # type: ignore
    forward_kinematic = None  # type: ignore
    msd = None  # type: ignore
    RenderApp = None  # type: ignore
    RenderSettings = None  # type: ignore
    _MOTRIXSIM_IMPORT_ERROR = exc
else:
    _MOTRIXSIM_IMPORT_ERROR = None


MIX_SYSTEM_CAMERA_NAME = "mix_system_camera"
MIX_SYSTEM_CAMERA_BODY_NAME = "mix_system_camera_body"
MIX_SYSTEM_CAMERA_FOVY = 45.0
MIX_OCCLUSION_MARGIN = 0.08
MIX_INTERIOR_TILE_MARGIN = 0.2
MIX_CAMERA_MIN_CLEARANCE = -0.05
MIX_CAMERA_MAX_CLEARANCE = 3.5
MIX_CHROMA_KEY = np.array([255, 0, 255], dtype=np.uint8)
MIX_MASK_THRESHOLD = 24
MIX_MASK_MIN_COMPONENT = 8
MIX_MASK_ERODE_PIXELS = 1
MIX_MASK_FEATHER_PIXELS = 2.0
MIX_SEGMENT_ROBOT_MATERIAL = "mix_segment_robot_mat"
MIX_SEGMENT_MASK_THRESHOLD = 24


def parse_vec(text: str | None, length: int, default: float = 0.0) -> np.ndarray:
    if not text:
        return np.full(length, default, dtype=np.float32)
    values = np.fromstring(text, sep=" ", dtype=np.float32)
    if values.shape[0] != length:
        return np.full(length, default, dtype=np.float32)
    return values


class SceneInteriorGate:
    def __init__(self, floor_tiles: np.ndarray) -> None:
        self.floor_tiles = floor_tiles

    @property
    def enabled(self) -> bool:
        return self.floor_tiles.size > 0

    @classmethod
    def from_mjcf(cls, scene_file: str | Path) -> "SceneInteriorGate":
        try:
            root = ET.parse(scene_file).getroot()
        except ET.ParseError:
            return cls(np.empty((0, 5), dtype=np.float32))

        worldbody = root.find("worldbody")
        if worldbody is None:
            return cls(np.empty((0, 5), dtype=np.float32))

        tiles = []

        def visit_body(body_elem, parent_pos: np.ndarray) -> None:
            body_pos = parent_pos + parse_vec(body_elem.get("pos"), 3)
            for geom in body_elem.findall("geom"):
                if geom.get("type") != "box" or geom.get("group") != "3":
                    continue
                size = parse_vec(geom.get("size"), 3)
                if size[2] > 0.04:
                    continue
                center = body_pos + parse_vec(geom.get("pos"), 3)
                tiles.append(
                    [
                        center[0] - size[0],
                        center[0] + size[0],
                        center[1] - size[1],
                        center[1] + size[1],
                        center[2] + size[2],
                    ]
                )
            for child in body_elem.findall("body"):
                visit_body(child, body_pos)

        for body_elem in worldbody.findall("body"):
            visit_body(body_elem, np.zeros(3, dtype=np.float32))

        return cls(np.asarray(tiles, dtype=np.float32) if tiles else np.empty((0, 5), dtype=np.float32))

    def contains_camera(self, camera_pos: np.ndarray) -> bool:
        if not self.enabled:
            return True
        x, y, z = camera_pos[:3]
        tiles = self.floor_tiles
        within_xy = (
            (x >= tiles[:, 0] - MIX_INTERIOR_TILE_MARGIN)
            & (x <= tiles[:, 1] + MIX_INTERIOR_TILE_MARGIN)
            & (y >= tiles[:, 2] - MIX_INTERIOR_TILE_MARGIN)
            & (y <= tiles[:, 3] + MIX_INTERIOR_TILE_MARGIN)
        )
        clearance = z - tiles[:, 4]
        return bool(
            np.any(
                within_xy
                & (clearance >= MIX_CAMERA_MIN_CLEARANCE)
                & (clearance <= MIX_CAMERA_MAX_CLEARANCE)
            )
        )


def find_camera_id(model: Any, camera_name: str) -> int | None:
    for idx, camera in enumerate(model.cameras):
        if camera.name == camera_name or camera.name.endswith(camera_name):
            return idx
    return None


def body_link_indices(model: Any, body: Any) -> np.ndarray:
    start = int(body.base_link.index)
    stop = min(start + int(body.num_links), len(model.link_names))
    return np.arange(start, stop, dtype=np.int64)


def add_mix_system_camera(scene: Any, backdrop_rgb: np.ndarray = MIX_CHROMA_KEY) -> None:
    backdrop = np.asarray(backdrop_rgb, dtype=np.float32) / 255.0
    backdrop_rgb_text = " ".join(f"{float(x):.6f}" for x in backdrop)
    backdrop_rgba_text = f"{backdrop_rgb_text} 1"
    material_name = "mix_chroma_backdrop_mat" if np.any(backdrop > 0) else "mix_mask_backdrop_mat"
    mix_camera_mjcf = f"""<mujoco model="mix_system_camera">
  <asset>
    <material name="{material_name}" rgba="{backdrop_rgba_text}"
      emission="{backdrop_rgba_text}" specular="0" reflectance="0" castshadow="false" />
  </asset>
  <worldbody>
    <body name="{MIX_SYSTEM_CAMERA_BODY_NAME}">
      <inertial pos="0 0 0" mass="0.001" diaginertia="0.001 0.001 0.001" />
      <freejoint />
      <geom name="mix_chroma_backdrop" type="box" pos="0 0 -30" size="30 30 0.01"
        material="{material_name}" contype="0" conaffinity="0" group="4" />
      <camera name="{MIX_SYSTEM_CAMERA_NAME}" pos="0 0 0" />
    </body>
  </worldbody>
</mujoco>"""
    scene.attach(msd.from_str(mix_camera_mjcf))


def robot_segmentation_mjcf(robot_cls: Any) -> str:
    root = ET.parse(robot_cls.mjcf_path).getroot()
    asset = root.find("asset")
    if asset is None:
        worldbody = root.find("worldbody")
        asset = ET.Element("asset")
        if worldbody is None:
            root.insert(0, asset)
        else:
            root.insert(list(root).index(worldbody), asset)

    ET.SubElement(
        asset,
        "material",
        {
            "name": MIX_SEGMENT_ROBOT_MATERIAL,
            "rgba": "1 1 1 1",
            "emission": "1 1 1 1",
            "specular": "0",
            "reflectance": "0",
            "castshadow": "false",
        },
    )

    for geom in root.iter("geom"):
        geom.set("material", MIX_SEGMENT_ROBOT_MATERIAL)
        geom.set("rgba", "1 1 1 1")

    return ET.tostring(root, encoding="unicode")


def build_foreground_model(robot_cls: Any, width: int, height: int) -> tuple[Any, int | None]:
    fg_scene = msd.from_str('<mujoco model="mix_foreground"><worldbody /></mujoco>')
    fg_robot = msd.from_file(robot_cls.mjcf_path.as_posix())
    fg_scene.attach(fg_robot)
    add_mix_system_camera(fg_scene)
    fg_model = fg_scene.build()
    fg_camera_id = find_camera_id(fg_model, MIX_SYSTEM_CAMERA_NAME)
    if fg_camera_id is not None:
        fg_model.cameras[fg_camera_id].set_render_target("image", width, height)
    return fg_model, fg_camera_id


def build_segmentation_model(robot_cls: Any, width: int, height: int) -> tuple[Any, int | None]:
    seg_scene = msd.from_str('<mujoco model="mix_segmentation"><worldbody /></mujoco>')
    seg_robot = msd.from_str(robot_segmentation_mjcf(robot_cls), file_path=robot_cls.mjcf_path.as_posix())
    seg_scene.attach(seg_robot)
    add_mix_system_camera(seg_scene, np.array([0, 0, 0], dtype=np.uint8))
    seg_model = seg_scene.build()
    seg_camera_id = find_camera_id(seg_model, MIX_SYSTEM_CAMERA_NAME)
    if seg_camera_id is not None:
        seg_model.cameras[seg_camera_id].set_render_target("image", width, height)
    return seg_model, seg_camera_id


def segmentation_render_settings() -> Any:
    settings = RenderSettings.performance()
    settings.enable_shadow = False
    settings.enable_ssao = False
    settings.enable_ssgi = False
    return settings


def configure_foreground_visibility(render: Any) -> None:
    render.opt.set_group_vis(0, True)
    render.opt.set_group_vis(1, True)
    render.opt.set_group_vis(2, True)
    render.opt.set_group_vis(3, False)
    render.opt.set_group_vis(4, True)
    render.opt.set_group_vis(5, False)


def sync_robot_state(target_data: Any, target_model: Any, source_data: Any) -> None:
    target_qpos = target_data.dof_pos.copy()
    target_qvel = target_data.dof_vel.copy()
    src_qpos_len = min(source_data.dof_pos.shape[0], target_qpos.shape[0])
    src_qvel_len = min(source_data.dof_vel.shape[0], target_qvel.shape[0])
    target_qpos[:src_qpos_len] = source_data.dof_pos[:src_qpos_len]
    target_qvel[:src_qvel_len] = source_data.dof_vel[:src_qvel_len]
    target_data.set_dof_pos(target_qpos, target_model)
    target_data.set_dof_vel(target_qvel)


def set_camera_pose(target_data: Any, target_model: Any, camera_pose: np.ndarray) -> bool:
    camera_body = target_model.get_body(MIX_SYSTEM_CAMERA_BODY_NAME)
    if camera_body is None or camera_body.floatingbase is None:
        return False
    camera_body.floatingbase.set_translation(target_data, camera_pose[:3])
    camera_body.floatingbase.set_rotation(target_data, camera_pose[3:7])
    forward_kinematic(target_model, target_data)
    return True


def to_depth_np(depth_tensor: Any) -> np.ndarray:
    if hasattr(depth_tensor, "detach"):
        depth_np = depth_tensor.detach().cpu().numpy()
    else:
        depth_np = np.asarray(depth_tensor)
    depth_np = np.asarray(depth_np, dtype=np.float32)
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    return depth_np


def make_foreground_mask(foreground: np.ndarray) -> np.ndarray:
    rgb = foreground[..., :3].astype(np.int16)
    h, w = rgb.shape[:2]
    sample = np.concatenate(
        [
            rgb[: max(1, h // 20), :, :].reshape(-1, 3),
            rgb[-max(1, h // 20) :, :, :].reshape(-1, 3),
            rgb[:, : max(1, w // 20), :].reshape(-1, 3),
            rgb[:, -max(1, w // 20) :, :].reshape(-1, 3),
        ],
        axis=0,
    )
    background_color = np.median(sample, axis=0)
    color_delta = np.max(np.abs(rgb - background_color), axis=-1)
    chroma_delta = np.max(np.abs(rgb - MIX_CHROMA_KEY.astype(np.int16)), axis=-1)
    color_delta = np.minimum(color_delta, chroma_delta)
    mask = color_delta > MIX_MASK_THRESHOLD

    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= MIX_MASK_MIN_COMPONENT
        keep[0] = False
        mask = keep[labels]

    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype(np.uint8)


def make_foreground_alpha(mask: np.ndarray, erode_pixels: int = MIX_MASK_ERODE_PIXELS) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return mask_bool.astype(np.float32)

    if erode_pixels > 0:
        structure = np.ones((3, 3), dtype=bool)
        core = ndimage.binary_erosion(mask_bool, structure=structure, iterations=erode_pixels)
    else:
        core = mask_bool
    if not np.any(core):
        core = mask_bool

    distance = ndimage.distance_transform_edt(mask_bool)
    alpha = np.clip((distance - float(erode_pixels)) / MIX_MASK_FEATHER_PIXELS, 0.0, 1.0)
    alpha = np.where(core, alpha, 0.0)
    return alpha.astype(np.float32)


def alpha_from_foreground(foreground: np.ndarray) -> np.ndarray:
    if foreground.shape[-1] >= 4:
        alpha = foreground[..., 3].astype(np.float32) / 255.0
        if alpha.min() < 0.98 and alpha.max() > 0.0:
            alpha = np.where(alpha > (4.0 / 255.0), alpha, 0.0)
            return alpha.astype(np.float32)
    return make_foreground_alpha(make_foreground_mask(foreground))


def robot_mask_from_segmentation(segmentation: np.ndarray | None, width: int, height: int) -> np.ndarray:
    if segmentation is None or segmentation.ndim < 3 or segmentation.shape[-1] < 3:
        return np.zeros((height, width), dtype=np.uint8)
    rgb = segmentation[..., :3].astype(np.uint8)
    mask = np.max(rgb, axis=-1) > MIX_SEGMENT_MASK_THRESHOLD

    labels, count = ndimage.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        keep = sizes >= MIX_MASK_MIN_COMPONENT
        keep[0] = False
        mask = keep[labels]
    return mask.astype(np.uint8)


def remove_chroma_spill(foreground: np.ndarray, mask: np.ndarray) -> np.ndarray:
    cleaned = foreground[..., :3].astype(np.int16).copy()
    rgb = cleaned
    magenta_excess = np.minimum(rgb[..., 0], rgb[..., 2]) - rgb[..., 1]
    spill = (mask > 0) & (magenta_excess > 10)
    limit = np.clip(rgb[..., 1] + 10, 0, 255)
    rgb[..., 0] = np.where(spill, np.minimum(rgb[..., 0], limit), rgb[..., 0])
    rgb[..., 2] = np.where(spill, np.minimum(rgb[..., 2], limit), rgb[..., 2])
    return np.clip(cleaned, 0, 255).astype(np.uint8)


def robot_proxy_radius(robot_name: str, link_name: str | None) -> float:
    name = link_name or ""
    if robot_name in {"go1", "go2"}:
        if name == "base":
            return 0.32
        if "hip" in name:
            return 0.12
        if "thigh" in name:
            return 0.13
        if "calf" in name:
            return 0.11
        return 0.08

    if "torso" in name:
        return 0.28
    if "pelvis" in name or "waist" in name:
        return 0.22
    if "head" in name:
        return 0.16
    if "ankle" in name:
        return 0.10
    if "knee" in name or "hip" in name or "shoulder" in name:
        return 0.13
    if "elbow" in name or "wrist" in name:
        return 0.09
    return 0.12


def camera_view_points(camera_pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    camera_rot = Rotation.from_quat(camera_pose[3:7]).as_matrix()
    camera_mat = np.eye(4, dtype=np.float32)
    camera_mat[:3, :3] = camera_rot
    camera_mat[:3, 3] = camera_pose[:3]
    camera_mat[:3, 1] *= -1
    camera_mat[:3, 2] *= -1
    view_mat = np.linalg.inv(camera_mat)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (view_mat @ points_h.T).T[:, :3]


def estimate_robot_depth_map(
    camera_pose: np.ndarray,
    model: Any,
    data: Any,
    robot_link_ids: np.ndarray,
    robot_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    if robot_link_ids.size == 0:
        return depth_map

    link_poses = model.get_link_poses(data)
    link_points = link_poses[robot_link_ids, :3].astype(np.float32)
    camera_points = camera_view_points(camera_pose, link_points)
    focal = height / (2.0 * np.tan(np.radians(MIX_SYSTEM_CAMERA_FOVY) / 2.0))

    for link_id, point in zip(robot_link_ids, camera_points):
        z = float(point[2])
        if z <= 0.02:
            continue
        link_name = model.link_names[int(link_id)]
        radius = robot_proxy_radius(robot_name, link_name)
        u = (point[0] * focal / z) + width / 2.0
        v = (point[1] * focal / z) + height / 2.0
        px_radius = max(2, int(np.ceil(focal * radius / z)))
        u0 = max(0, int(np.floor(u - px_radius)))
        u1 = min(width, int(np.ceil(u + px_radius + 1)))
        v0 = max(0, int(np.floor(v - px_radius)))
        v1 = min(height, int(np.ceil(v + px_radius + 1)))
        if u0 >= u1 or v0 >= v1:
            continue

        ys, xs = np.ogrid[v0:v1, u0:u1]
        disk = (xs - u) ** 2 + (ys - v) ** 2 <= px_radius**2
        candidate_depth = max(0.0, z - min(radius * 0.1, 0.03))
        patch = depth_map[v0:v1, u0:u1]
        patch[disk] = np.minimum(patch[disk], candidate_depth)

    depth_map[~np.isfinite(depth_map)] = np.nan
    return depth_map


def fill_depth_under_mask(depth_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if depth_map.shape != mask.shape[:2]:
        return depth_map
    finite = np.isfinite(depth_map) & (depth_map > 0)
    if not np.any(finite):
        return depth_map
    missing = (mask > 0) & ~finite
    if not np.any(missing):
        return depth_map
    nearest = ndimage.distance_transform_edt(~finite, return_distances=False, return_indices=True)
    filled = depth_map[tuple(nearest)]
    return np.where(missing, filled, depth_map)


def composite_mix(
    background: np.ndarray,
    foreground: np.ndarray,
    robot_mask: np.ndarray | None = None,
    background_depth: np.ndarray | None = None,
    foreground_depth: np.ndarray | float | None = None,
) -> np.ndarray:
    if foreground.shape[:2] != background.shape[:2]:
        return background
    if robot_mask is not None and robot_mask.shape == background.shape[:2]:
        alpha = make_foreground_alpha(robot_mask, erode_pixels=0)
    else:
        alpha = alpha_from_foreground(foreground)
    mask = alpha > 0

    foreground_rgb = remove_chroma_spill(foreground, mask)

    if background_depth is not None and foreground_depth is not None and background_depth.shape == mask.shape:
        if np.isscalar(foreground_depth):
            foreground_depth_map = np.full(mask.shape, foreground_depth, dtype=np.float32)
        else:
            foreground_depth_map = np.asarray(foreground_depth, dtype=np.float32)
            foreground_depth_map = fill_depth_under_mask(foreground_depth_map, mask)
        valid_depth = (
            np.isfinite(background_depth)
            & (background_depth > 0)
            & np.isfinite(foreground_depth_map)
            & (foreground_depth_map > 0)
        )
        occluded = valid_depth & (background_depth < np.maximum(0.0, foreground_depth_map - MIX_OCCLUSION_MARGIN))
        alpha = alpha * (~occluded).astype(np.float32)

    alpha = alpha[..., None]
    return np.clip(background.astype(np.float32) * (1.0 - alpha) + foreground_rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def poll_capture_image(task: Any) -> tuple[np.ndarray | None, Any | None]:
    if task is None:
        return None, None
    if task.state == "pending":
        return None, task
    try:
        image = task.take_image()
    except RuntimeError:
        return None, None
    if image is None:
        return None, None
    pixels = np.asarray(image.pixels)
    if pixels.ndim == 3 and pixels.shape[-1] >= 3:
        return np.ascontiguousarray(pixels), None
    return None, None


def capture_now(render: Any, camera: Any, data: Any) -> np.ndarray | None:
    render.sync(data, wait=True)
    task = camera.capture()
    render.sync(data, wait=True)
    image, task = poll_capture_image(task)
    if image is not None:
        return image
    if task is not None:
        render.sync(data, wait=True)
        image, _ = poll_capture_image(task)
    return image


class GSRendererMuGS:
    def __init__(
        self,
        robot_cls: Any,
        robot_name: str,
        scene_file: str | Path,
        model: Any,
        data: Any,
        robot_body: Any,
        width: int = 480,
        height: int = 360,
    ) -> None:
        if _MOTRIXSIM_IMPORT_ERROR is not None:
            raise ImportError("MotrixSim is not installed. Install motrixsim to use GSRendererMuGS.") from _MOTRIXSIM_IMPORT_ERROR

        self.robot_name = robot_name
        self.scene_file = Path(scene_file)
        self.model = model
        self.width = width
        self.height = height
        self.robot_link_ids = body_link_indices(model, robot_body)
        self.interior_gate = SceneInteriorGate.from_mjcf(scene_file)
        self.ready = False

        self.fg_render = None
        self.fg_data = None
        self.fg_camera = None
        self.seg_render = None
        self.seg_data = None
        self.seg_camera = None

        self.fg_model, fg_camera_id = build_foreground_model(robot_cls, width, height)
        self.seg_model, seg_camera_id = build_segmentation_model(robot_cls, width, height)

        fg_camera_body = self.fg_model.get_body(MIX_SYSTEM_CAMERA_BODY_NAME)
        seg_camera_body = self.seg_model.get_body(MIX_SYSTEM_CAMERA_BODY_NAME)
        if (
            fg_camera_id is None
            or seg_camera_id is None
            or fg_camera_body is None
            or fg_camera_body.floatingbase is None
            or seg_camera_body is None
            or seg_camera_body.floatingbase is None
        ):
            return

        self.fg_render = RenderApp(headless=True)
        self.fg_render.__enter__()
        self.fg_render.launch(self.fg_model, render_settings=RenderSettings.performance())
        configure_foreground_visibility(self.fg_render)
        self.fg_data = SceneData(self.fg_model)
        sync_robot_state(self.fg_data, self.fg_model, data)
        self.fg_camera = self.fg_render.get_camera(fg_camera_id)

        self.seg_render = RenderApp(headless=True)
        self.seg_render.__enter__()
        self.seg_render.launch(self.seg_model, render_settings=segmentation_render_settings())
        configure_foreground_visibility(self.seg_render)
        self.seg_data = SceneData(self.seg_model)
        sync_robot_state(self.seg_data, self.seg_model, data)
        self.seg_camera = self.seg_render.get_camera(seg_camera_id)
        self.ready = True

    def sync(self, data: Any) -> None:
        if self.fg_data is not None:
            sync_robot_state(self.fg_data, self.fg_model, data)
        if self.seg_data is not None:
            sync_robot_state(self.seg_data, self.seg_model, data)

    def reset(self, data: Any) -> None:
        if self.fg_data is not None:
            self.fg_data.reset(self.fg_model)
        if self.seg_data is not None:
            self.seg_data.reset(self.seg_model)
        self.sync(data)

    def composite(
        self,
        background_rgb: np.ndarray,
        background_depth: Any,
        data: Any,
        system_camera: Any,
    ) -> np.ndarray:
        if not self.ready:
            return background_rgb

        camera_pose = np.asarray(system_camera.pose, dtype=np.float32)
        if not self.interior_gate.contains_camera(camera_pose[:3]):
            return background_rgb

        set_camera_pose(self.fg_data, self.fg_model, camera_pose)
        set_camera_pose(self.seg_data, self.seg_model, camera_pose)
        foreground = capture_now(self.fg_render, self.fg_camera, self.fg_data)
        segmentation = capture_now(self.seg_render, self.seg_camera, self.seg_data)
        if foreground is None:
            return background_rgb

        robot_mask = robot_mask_from_segmentation(segmentation, self.width, self.height) if segmentation is not None else None
        foreground_depth = estimate_robot_depth_map(
            camera_pose,
            self.model,
            data,
            self.robot_link_ids,
            self.robot_name,
            self.width,
            self.height,
        )
        return composite_mix(
            background_rgb,
            foreground,
            robot_mask,
            to_depth_np(background_depth),
            foreground_depth,
        )

    def close(self) -> None:
        if self.fg_render is not None:
            self.fg_render.__exit__(None, None, None)
            self.fg_render = None
        if self.seg_render is not None:
            self.seg_render.__exit__(None, None, None)
            self.seg_render = None

    def __enter__(self) -> "GSRendererMuGS":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False
