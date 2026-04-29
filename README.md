# Gaussian Renderer

A Gaussian Splatting Renderer and Tools package.

[中文文档](README_zh.md)

This repository primarily provides rendering capabilities for Gaussian Splatting models. It was first developed as the Gaussian rendering component for **DISCOVERSE**, the first-generation simulation work.

Building on DISCOVERSE, GaussianRenderer is now extended for [gs_playground](https://github.com/discoverse-dev/gs_playground), the second-generation work, where it provides the Gaussian rendering components related to Rigid-Link Gaussian Kinematics.

For detailed usage within the simulation environment, see: [https://github.com/TATP-233/DISCOVERSE](https://github.com/TATP-233/DISCOVERSE)

## Requirements

Python >= 3.10

## Installation

```bash
uv add gaussian-renderer
# or: pip install gaussian-renderer
```

From source:
```bash
git clone https://github.com/TATP-233/GaussainRenderer.git
cd GaussainRenderer
uv pip install .
# or: pip install .
```

### Optional extras

```bash
uv add "gaussian-renderer[viewer]"   # OpenGL viewer (glfw, PyOpenGL)
uv add "gaussian-renderer[mujoco]"   # MuJoCo integration
uv add "gaussian-renderer[motrix]"   # MotrixSim integration

# Combine as needed
uv add "gaussian-renderer[viewer,mujoco]"
# or: pip install ".[viewer,mujoco]"
```

## Usage

### Command-line tools

**`gs-viewer`** — OpenGL viewer for `.ply` models
```bash
gs-viewer path/to/model.ply
```
Controls: Left mouse = rotate, Right/Middle = pan, Scroll = zoom, Up/Down = SH degree, Drag & drop = load file

**`gs-compress`** — Compress 3DGS PLY to SuperSplat format
```bash
gs-compress input.ply
gs-compress input.ply -o output.ply
gs-compress models/          # batch
```

**`gs-transform`** — Apply translation/rotation/scale to a model
```bash
gs-transform input.ply -o output.ply -t 0 1 0 -s 2.0
gs-transform input.ply -r 0 0 0 1   # rotation quaternion xyzw
# --compress: save as compressed PLY
```

### Python API

```bash
uv run python -m gaussian_renderer.simple_viewer path/to/model.ply
uv run python -m gaussian_renderer.supersplat_compress input.ply
uv run python -m gaussian_renderer.transform_gs_model input.ply
```

## Development

```bash
uv pip install ".[dev]"
# or: pip install ".[dev]"
make lint       # ruff check
make format     # ruff format
make typecheck  # mypy
make test       # pytest
make ci         # all of the above
```

## Citation

```bibtex
@article{jia2025discoverse,
      title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
      author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
      journal={arXiv preprint arXiv:2507.21981},
      year={2025},
      url={https://arxiv.org/abs/2507.21981}
}

@article{jia2026gsplayground,
      title={GS-Playground: A High-Throughput Photorealistic Simulator for Vision-Informed Robot Learning},
      author={Yufei Jia and Heng Zhang and Ziheng Zhang and Junzhe Wu and Mingrui Yu and Zifan Wang and Dixuan Jiang and Zheng Li and Chenyu Cao and Zhuoyuan Yu and Xun Yang and Haizhou Ge and Yuchi Zhang and Jiayuan Zhang and Zhenbiao Huang and Tianle Liu and Shenyu Chen and Jiacheng Wang and Bin Xie and Xuran Yao and Xiwa Deng and Guangyu Wang and Jinzhi Zhang and Lei Hao and Zhixing Chen and Yuxiang Chen and Anqi Wang and Hongyun Tian and Yiyi Yan and Zhanxiang Cao and Yizhou Jiang and Hanyang Shao and Yue Li and Lu Shi and Bokui Chen and Wei Sui and Hanqing Cui and Yusen Qin and Ruqi Huang and Lei Han and Tiancai Wang and Guyue Zhou},
      journal={arXiv preprint arXiv:2604.25459},
      year={2026},
      url={https://arxiv.org/abs/2604.25459}
}
```
