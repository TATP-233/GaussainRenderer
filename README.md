# Gaussian Renderer

A Gaussian Splatting Renderer and Tools package.

[中文文档](README_zh.md)

This repository primarily provides rendering capabilities for Gaussian Splatting models. It is developed as a component of the **DISCOVERSE** project.

For detailed usage of the renderer within the simulation environment, please refer to: [https://github.com/TATP-233/DISCOVERSE](https://github.com/TATP-233/DISCOVERSE)


## Installation

### Basic Installation
To install the core package (without viewer dependencies):

```bash
pip install gaussian-renderer
```

Or from source:
```bash
git clone https://github.com/TATP-233/GaussainRenderer.git
cd GaussainRenderer
pip install .
```

### With Viewer Support
To install with viewer dependencies (glfw, PyOpenGL):

```bash
pip install ".[viewer]"
```

### With MuJoCo Support (Optional)
To use MuJoCo-related functionality (e.g., modules in `gaussian_renderer/gs_renderer_mujoco.py`), install the optional MuJoCo extra:

```bash
pip install ".[mujoco]"
```

### With MotrixSim Support (Optional)
To enable MotrixSim integration, install the optional extra:

```bash
pip install ".[motrix]"
```

### Combine Extras
You can combine extras as needed, for example:

```bash
# Viewer + MuJoCo
pip install ".[viewer,mujoco]"

# Viewer + MotrixSim
pip install ".[viewer,motrix]"
```

## Usage

This package provides three command-line tools:

### 1. Simple Viewer (`gs-viewer`)
A simple OpenGL-based viewer for Gaussian Splatting models (.ply files).

**Usage:**
```bash
# Open a specific model
gs-viewer path/to/model.ply

# Or just run and drag-drop files
gs-viewer
```

**Controls:**
- **Left Mouse**: Rotate
- **Right/Middle Mouse**: Pan
- **Scroll**: Zoom
- **Up/Down Key**: Adjust SH Degree
- **Drag & Drop**: Load PLY file

### 2. SuperSplat Compressor (`gs-compress`)
Compress 3DGS PLY models to the SuperSplat format.

**Usage:**
```bash
# Compress a single file
gs-compress input.ply

# Specify output file
gs-compress input.ply -o output.ply

# Batch compress a directory
gs-compress models/
```

### 3. Model Transformer (`gs-transform`)
Apply transformations (translation, rotation, scaling) to Gaussian Splatting models.

**Usage:**
```bash
# Basic transformation
gs-transform input.ply -o output.ply -t 0 1 0 -s 2.0

# Rotate (quaternion xyzw)
gs-transform input.ply -r 0 0 0 1

# Options:
# -t x y z      : Translation vector
# -r x y z w    : Rotation quaternion
# -s scale      : Scale factor
# --compress    : Save as compressed PLY
```

### Python API

You can also use the tools as Python modules:

```bash
python -m gaussian_renderer.simple_viewer path/to/model.ply
python -m gaussian_renderer.supersplat_compress input.ply
python -m gaussian_renderer.transform_gs_model input.ply
```
## Citation

If you find this package helpful, please consider citing our work:

```bibtex
@article{jia2025discoverse,
      title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
      author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
      journal={arXiv preprint arXiv:2507.21981},
      year={2025},
      url={https://arxiv.org/abs/2507.21981}
}
```
