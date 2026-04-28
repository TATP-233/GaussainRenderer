# Gaussian Renderer

高斯泼溅（Gaussian Splatting）渲染器及工具包。

本仓库主要提供高斯泼溅模型的渲染功能，是 **DISCOVERSE** 项目的一部分。

在 DISCOVERSE 的基础上，GaussianRenderer 也支持 [gs_playground](https://github.com/discoverse-dev/gs_playground) 的相关工作，包含 Rigid-Link Gaussian Kinematics 所需的高斯渲染组件。

关于在仿真环境中的详细使用方法，请参考：[https://github.com/TATP-233/DISCOVERSE](https://github.com/TATP-233/DISCOVERSE)

## 环境要求

Python >= 3.10

## 安装

```bash
uv add gaussian-renderer
# 或：pip install gaussian-renderer
```

从源码安装：
```bash
git clone https://github.com/TATP-233/GaussainRenderer.git
cd GaussainRenderer
uv pip install .
# 或：pip install .
```

### 可选依赖

```bash
uv add "gaussian-renderer[viewer]"   # OpenGL 查看器（glfw, PyOpenGL）
uv add "gaussian-renderer[mujoco]"   # MuJoCo 集成
uv add "gaussian-renderer[motrix]"   # MotrixSim 集成

# 按需组合
uv add "gaussian-renderer[viewer,mujoco]"
# 或：pip install ".[viewer,mujoco]"
```

## 使用

### 命令行工具

**`gs-viewer`** — OpenGL 查看器，支持 `.ply` 模型
```bash
gs-viewer path/to/model.ply
```
控制：左键旋转，右键/中键平移，滚轮缩放，上/下键调整 SH 阶数，拖放加载文件

**`gs-compress`** — 将 3DGS PLY 压缩为 SuperSplat 格式
```bash
gs-compress input.ply
gs-compress input.ply -o output.ply
gs-compress models/          # 批量处理
```

**`gs-transform`** — 对模型应用平移/旋转/缩放
```bash
gs-transform input.ply -o output.ply -t 0 1 0 -s 2.0
gs-transform input.ply -r 0 0 0 1   # 旋转四元数 xyzw
# --compress: 保存为压缩 PLY
```

### Python API

```bash
uv run python -m gaussian_renderer.simple_viewer path/to/model.ply
uv run python -m gaussian_renderer.supersplat_compress input.ply
uv run python -m gaussian_renderer.transform_gs_model input.ply
```

## 开发

```bash
uv pip install ".[dev]"
# 或：pip install ".[dev]"
make lint       # ruff 检查
make format     # ruff 格式化
make typecheck  # mypy
make test       # pytest
make ci         # 以上全部
```

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@article{jia2025discoverse,
    title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
    author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
    journal={arXiv preprint arXiv:2507.21981},
    year={2025},
    url={https://arxiv.org/abs/2507.21981}
}
```
