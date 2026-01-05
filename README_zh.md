# Gaussian Renderer

高斯泼溅（Gaussian Splatting）渲染器及工具包。

本仓库主要提供高斯泼溅模型的渲染功能，是 **DISCOVERSE** 项目的一部分。

关于在仿真环境中渲染器的详细使用方法，请参考：[https://github.com/TATP-233/DISCOVERSE](https://github.com/TATP-233/DISCOVERSE)

## 安装

### 基础安装
安装核心包（不包含查看器依赖）：

```bash
pip install gaussian-renderer
```

从源码安装：
```bash
git clone https://github.com/TATP-233/GaussainRenderer.git
cd GaussainRenderer
pip install .
```

### 带查看器支持
安装包含查看器依赖（glfw, PyOpenGL）的版本：

```bash
pip install ".[viewer]"
```

## 命令行使用

本包提供了三个命令行工具：

### 1. 简易查看器 (`gs-viewer`)
一个基于 OpenGL 的简单高斯泼溅模型（.ply 文件）查看器。

**用法：**
```bash
# 打开特定模型
gs-viewer path/to/model.ply

# 或者直接运行并拖放文件
gs-viewer
```

**控制：**
- **左键**：旋转
- **右键/中键**：平移
- **滚轮**：缩放
- **上/下键**：调整 SH 阶数
- **拖放**：加载 PLY 文件

### 2. SuperSplat 压缩工具 (`gs-compress`)
将 3DGS PLY 模型压缩为 SuperSplat 格式。

**用法：**
```bash
# 压缩单个文件
gs-compress input.ply

# 指定输出文件
gs-compress input.ply -o output.ply

# 批量压缩目录
gs-compress models/
```

### 3. 模型变换工具 (`gs-transform`)
对高斯泼溅模型应用变换（平移、旋转、缩放）。

**用法：**
```bash
# 基础变换
gs-transform input.ply -o output.ply -t 0 1 0 -s 2.0

# 旋转 (四元数 xyzw)
gs-transform input.ply -r 0 0 0 1

# 选项：
# -t x y z      : 平移向量
# -r x y z w    : 旋转四元数
# -s scale      : 缩放因子
# --compress    : 保存为压缩的 PLY
```

### Python API

您也可以将这些工具作为 Python 模块使用：

```bash
python -m gaussian_renderer.simple_viewer path/to/model.ply
python -m gaussian_renderer.supersplat_compress input.ply
python -m gaussian_renderer.transform_gs_model input.ply
```

## 引用

如果您发现DISCOVERSE对您的研究有帮助，请考虑引用我们的工作：

```bibtex
@article{jia2025discoverse,
    title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
    author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
    journal={arXiv preprint arXiv:2507.21981},
    year={2025},
    url={https://arxiv.org/abs/2507.21981}
}
```