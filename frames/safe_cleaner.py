# -*- coding: utf-8 -*-
"""
@Time ： 2026/3/5 14:43
@Auth ： chensonglin
@IDE ：PyCharm
  Describe:

"""
# safe_cleaner.py
import os
import glob
from pathlib import Path


def safe_delete_all_except_self():
    """安全删除当前目录下除脚本自身外的所有文件"""

    # 获取当前脚本的文件名
    current_script = Path(__file__).name
    current_dir = Path('.')

    print(f"当前脚本: {current_script}")
    print(f"当前目录: {current_dir.absolute()}")
    print("-" * 50)

    # 获取所有文件（不包括子目录中的文件）
    all_files = [f for f in current_dir.iterdir() if f.is_file()]

    # 筛选出要删除的文件（排除当前脚本）
    files_to_delete = [f for f in all_files if f.name != current_script]

    if not files_to_delete:
        print("没有需要删除的文件")
        return

    print(f"找到 {len(files_to_delete)} 个文件将被删除:")
    for file in files_to_delete:
        size = file.stat().st_size
        print(f"  - {file.name:<30} {size:>10} bytes")

    print("-" * 50)
    response = input(f"确定要删除这 {len(files_to_delete)} 个文件吗? (y/N): ")

    if response.lower() == 'y':
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"✓ 已删除: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ 删除失败 {file_path.name}: {e}")

        print(f"\n成功删除 {deleted_count} 个文件")
    else:
        print("取消删除操作")


if __name__ == '__main__':
    safe_delete_all_except_self()
