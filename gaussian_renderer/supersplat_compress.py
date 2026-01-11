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
将3DGS PLY模型压缩为SuperSplat格式
支持单个文件压缩和目录批量压缩

使用方法:
    # 单个文件
    python supersplat_compress.py input.ply
    python supersplat_compress.py input.ply -o output.ply
    
    # 批量处理目录
    python supersplat_compress.py models/
    python supersplat_compress.py models/ --backup
"""

import argparse
import sys
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional
from functools import partial
import time

# 添加项目根目录到Python路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from .src.util_gau import load_ply
from .src.super_splat_loader import save_super_splat_ply, is_super_splat_format
from plyfile import PlyData


def find_ply_files(directory: Path, recursive: bool = True) -> List[Path]:
    """查找目录中的所有PLY文件"""
    if recursive:
        ply_files = list(directory.rglob("*.ply"))
    else:
        ply_files = list(directory.glob("*.ply"))
    return sorted(ply_files)


def check_supersplat_format(ply_path: Path) -> bool:
    """检查PLY文件是否已经是SuperSplat格式"""
    try:
        plydata = PlyData.read(str(ply_path))
        return is_super_splat_format(plydata)
    except Exception:
        return False


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def compress_single_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    backup: bool = False,
    save_sh_degree: Optional[int] = None,
    force: bool = False,
    verbose: bool = True
) -> Tuple[bool, str, int, int, int, Path]:
    """
    压缩单个PLY文件
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径 (如果为None，则替换原文件或生成默认名)
        backup: 是否备份原文件 (仅当原位替换时有效)
        save_sh_degree: SH阶数
        force: 是否强制覆盖/重新压缩
        verbose: 是否打印详细信息
        
    Returns:
        (成功标志, 消息, 原始大小, 压缩后大小, 点数, 最终文件路径)
    """
    try:
        # 检查输入文件
        if not input_path.exists():
            return False, f"输入文件不存在: {input_path}", 0, 0, 0, input_path
            
        # 检查是否已经是SuperSplat格式
        if not force and check_supersplat_format(input_path):
            return False, "已是SuperSplat格式", 0, 0, 0, input_path

        # 确定输出路径
        is_inplace = False
        if output_path is None:
            # 如果是批量处理模式（通常output_path为None），默认行为是原位替换
            # 但为了安全，我们先生成临时文件
            is_inplace = True
            temp_output = input_path.with_suffix('.ply.tmp')
        else:
            # 指定了输出路径
            if output_path.exists() and not force:
                return False, f"输出文件已存在: {output_path}", 0, 0, 0, output_path
            temp_output = output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # 记录原始大小
        original_size = input_path.stat().st_size
        
        # 备份处理 (仅在原位替换模式下)
        backup_path = None
        if is_inplace and backup:
            backup_path = input_path.with_suffix('.ply.bak')
            shutil.copy2(input_path, backup_path)

        # 加载模型
        try:
            gaussian_data = load_ply(str(input_path))
            num_points = len(gaussian_data.xyz)
        except Exception as e:
            if backup_path and backup_path.exists():
                backup_path.unlink()
            return False, f"加载失败: {e}", 0, 0, 0, input_path

        if verbose:
            print(f"正在压缩: {input_path.name} ({num_points} points)")

        # 压缩并保存
        try:
            save_super_splat_ply(gaussian_data, str(temp_output), save_sh_degree=save_sh_degree)
        except Exception as e:
            if is_inplace and temp_output.exists():
                temp_output.unlink()
            if backup_path and backup_path.exists():
                backup_path.unlink()
            return False, f"压缩失败: {e}", 0, 0, 0, input_path

        # 如果是原位替换，移动临时文件覆盖原文件
        final_path = temp_output
        if is_inplace:
            temp_output.replace(input_path)
            final_path = input_path
            # 如果不需要备份，删除备份
            if not backup and backup_path and backup_path.exists():
                backup_path.unlink()

        compressed_size = final_path.stat().st_size
        return True, "压缩成功", original_size, compressed_size, num_points, final_path

    except Exception as e:
        return False, f"错误: {e}", 0, 0, 0, input_path


def process_directory(args):
    """处理目录批量压缩"""
    dir_path = Path(args.input)
    
    # 确定并行进程数
    if args.jobs == 0:
        num_workers = mp.cpu_count()
    else:
        num_workers = max(1, args.jobs)

    print(f"正在搜索PLY文件...")
    recursive = not args.no_recursive
    ply_files = find_ply_files(dir_path, recursive=recursive)
    
    if not ply_files:
        print(f"未找到任何PLY文件")
        return 0
    
    print(f"找到 {len(ply_files)} 个PLY文件")
    
    # 过滤
    files_to_compress = []
    files_skipped = []
    
    print("\n检查文件格式...")
    for i, ply_file in enumerate(ply_files, 1):
        rel_path = ply_file.relative_to(dir_path)
        # 简单检查，不打印太多
        if not args.force and check_supersplat_format(ply_file):
            files_skipped.append(ply_file)
        else:
            files_to_compress.append(ply_file)
            
    print(f"  - 需要压缩: {len(files_to_compress)}")
    print(f"  - 跳过: {len(files_skipped)}")
    
    if not files_to_compress:
        print("\n无需处理")
        return 0

    # Dry run
    if args.dry_run:
        print("\n[DRY RUN] 将要压缩的文件:")
        for ply_file in files_to_compress:
            print(f"  - {ply_file.relative_to(dir_path)}")
        return 0

    # 确认
    if not args.yes:
        print(f"\n警告: 将原位替换 {len(files_to_compress)} 个文件")
        if args.backup:
            print("  已启用备份 (.ply.bak)")
        else:
            print("  未启用备份，原文件将被覆盖")
        
        if input("\n是否继续? [y/N]: ").strip().lower() not in ['y', 'yes']:
            print("操作已取消")
            return 0

    # 执行
    print(f"\n开始批量压缩 (进程数: {num_workers})...")
    start_time = time.time()
    
    success_count = 0
    failed_count = 0
    total_orig = 0
    total_comp = 0

    # 包装函数以适配 pool.map
    # 注意：output_path=None 表示原位替换
    compress_func = partial(
        compress_single_file,
        output_path=None,
        backup=args.backup,
        save_sh_degree=args.sh_degree,
        force=args.force,
        verbose=False
    )

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = []
            for f in files_to_compress:
                results.append((f, pool.apply_async(compress_func, (f,))))
            
            for i, (f, res) in enumerate(results, 1):
                rel = f.relative_to(dir_path)
                try:
                    success, msg, orig, comp, pts, _ = res.get(timeout=600)
                    if success:
                        success_count += 1
                        total_orig += orig
                        total_comp += comp
                        ratio = (1 - comp/orig)*100 if orig > 0 else 0
                        print(f"[{i}/{len(files_to_compress)}] ✓ {rel} ({format_size(orig)} -> {format_size(comp)}, -{ratio:.1f}%)")
                    else:
                        failed_count += 1
                        print(f"[{i}/{len(files_to_compress)}] ✗ {rel}: {msg}")
                except Exception as e:
                    failed_count += 1
                    print(f"[{i}/{len(files_to_compress)}] ✗ {rel}: {e}")
    else:
        for i, f in enumerate(files_to_compress, 1):
            rel = f.relative_to(dir_path)
            success, msg, orig, comp, pts, _ = compress_func(f)
            if success:
                success_count += 1
                total_orig += orig
                total_comp += comp
                ratio = (1 - comp/orig)*100 if orig > 0 else 0
                print(f"[{i}/{len(files_to_compress)}] ✓ {rel} ({format_size(orig)} -> {format_size(comp)}, -{ratio:.1f}%)")
            else:
                failed_count += 1
                print(f"[{i}/{len(files_to_compress)}] ✗ {rel}: {msg}")

    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"完成! 耗时: {elapsed:.1f}s")
    print(f"成功: {success_count}, 失败: {failed_count}, 跳过: {len(files_skipped)}")
    if success_count > 0:
        saved = total_orig - total_comp
        ratio = (1 - total_comp/total_orig)*100 if total_orig > 0 else 0
        print(f"总空间节省: {format_size(saved)} ({ratio:.1f}%)")
    
    return 0 if failed_count == 0 else 1


def process_single_file(args):
    """处理单个文件压缩"""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    # 如果没有指定输出路径，且不是原位替换模式（通常单文件模式用户期望生成新文件，除非明确覆盖）
    # 这里逻辑：如果用户没给 -o，默认生成 .compressed.ply，而不是直接覆盖原文件（除非像 batch 模式那样明确是原位）
    # 但为了统一，如果用户想覆盖，可以指定 -o input.ply --force
    # 或者我们保持 compress_to_supersplat.py 的行为：默认生成 .compressed.ply
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}.compressed{input_path.suffix}"
    
    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    
    success, msg, orig, comp, pts, final_path = compress_single_file(
        input_path,
        output_path=output_path,
        backup=args.backup, # 单文件模式通常不备份，除非用户指定（虽然 argparse 里有）
        save_sh_degree=args.sh_degree,
        force=args.force,
        verbose=True
    )
    
    if success:
        ratio = (1 - comp/orig)*100 if orig > 0 else 0
        print(f"\n成功! {msg}")
        print(f"原始大小: {format_size(orig)}")
        print(f"压缩后:   {format_size(comp)}")
        print(f"节省空间: {ratio:.2f}%")
        return 0
    else:
        print(f"\n失败: {msg}", file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='3DGS模型 SuperSplat 压缩工具 (支持单文件和目录批量处理)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='输入文件路径 (.ply) 或 目录路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文件路径 (仅单文件模式有效)。默认: filename.compressed.ply'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='备份原文件 (仅在原位替换/批量模式下有效)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='不递归搜索子目录 (仅批量模式有效)'
    )
    
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=0,
        help='并行进程数 (仅批量模式有效, 0=自动)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制压缩 (即使已是SuperSplat格式或输出已存在)'
    )
    
    parser.add_argument(
        '--sh-degree',
        type=int,
        default=None,
        help='指定SH阶数 (0-3)。默认保持原样'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅显示将要处理的文件 (仅批量模式有效)'
    )
    
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help='跳过确认 (仅批量模式有效)'
    )

    args = parser.parse_args()
    
    path = Path(args.input)
    
    if not path.exists():
        print(f"错误: 路径不存在: {path}", file=sys.stderr)
        return 1

    if path.is_file():
        if path.suffix.lower() != '.ply':
            print(f"错误: 输入文件必须是 .ply 格式", file=sys.stderr)
            return 1
        return process_single_file(args)
    elif path.is_dir():
        if args.output:
            print("警告: 批量模式下 --output 参数将被忽略 (总是原位替换)", file=sys.stderr)
        return process_directory(args)
    else:
        print(f"错误: 无效的路径类型", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
