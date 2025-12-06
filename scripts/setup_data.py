#!/usr/bin/env python
"""
数据设置脚本
============

从主项目复制必要的数据文件到release/data目录
"""

import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SCRIPT_DIR)

# 源数据目录 (主项目)
SOURCE_DATA_DIR = os.path.join(os.path.dirname(_BASE_DIR), 'data')

# 目标数据目录 (release)
TARGET_DATA_DIR = os.path.join(_BASE_DIR, 'data')

# 需要的文件列表
REQUIRED_FILES = [
    # LIHC
    'TCGA-LIHC.star_fpkm.tsv.gz',
    'TCGA-LIHC.survival.tsv.gz',
    'TCGA-LIHC_embeddings_v5.pt',
    'LIRI-JP.star_fpkm.tsv.gz',
    'LIRI-JP.survival.tsv.gz',
    'LIRI-JP_embeddings_v5.pt',
    
    # BRCA
    'TCGA-BRCA_symbols.star_fpkm.tsv.gz',
    'TCGA-BRCA.survival.tsv.gz',
    'TCGA-BRCA_embeddings_v5.pt',
    'GSE20685.expression.tsv.gz',
    'GSE20685.survival.tsv.gz',
    'GSE20685_embeddings_v5.pt',
    
    # OV
    'TCGA-OV.star_fpkm.tsv.gz',
    'TCGA-OV.survival.tsv.gz',
    'TCGA-OV_embeddings_v5.pt',
    'OV-AU.star_fpkm.tsv.gz',
    'OV-AU.survival.tsv.gz',
    'OV-AU_embeddings_v5.pt',
    
    # PAAD
    'TCGA-PAAD.star_fpkm.tsv.gz',
    'TCGA-PAAD.survival.tsv.gz',
    'TCGA-PAAD_embeddings_v5.pt',
    'PACA-CA.star_fpkm.tsv.gz',
    'PACA-CA.survival.tsv.gz',
    'PACA-CA_embeddings_v5.pt',
    
    # PRAD
    'TCGA-PRAD.star_fpkm.tsv.gz',
    'TCGA-PRAD.survival.tsv.gz',
    'TCGA-PRAD_embeddings_v5.pt',
    'PRAD-CA.star_fpkm.tsv.gz',
    'PRAD-CA.survival.tsv.gz',
    'PRAD-CA_embeddings_v5.pt',
    
    # 基因ID映射
    'gene_id_mapping.csv',
]


def setup_data(source_dir=None, target_dir=None, use_symlink=False):
    """
    设置数据文件
    
    Args:
        source_dir: 源数据目录
        target_dir: 目标数据目录
        use_symlink: 是否使用符号链接(节省空间)
    """
    if source_dir is None:
        source_dir = SOURCE_DATA_DIR
    if target_dir is None:
        target_dir = TARGET_DATA_DIR
    
    print("="*60)
    print("数据设置")
    print("="*60)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"使用符号链接: {use_symlink}")
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 检查和复制文件
    success = 0
    failed = 0
    skipped = 0
    
    for filename in REQUIRED_FILES:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        # 检查目标是否已存在
        if os.path.exists(target_path):
            print(f"  ✓ 已存在: {filename}")
            skipped += 1
            continue
        
        # 检查源文件
        if not os.path.exists(source_path):
            print(f"  ✗ 源文件不存在: {filename}")
            failed += 1
            continue
        
        try:
            if use_symlink:
                os.symlink(source_path, target_path)
                print(f"  → 链接: {filename}")
            else:
                shutil.copy2(source_path, target_path)
                print(f"  → 复制: {filename}")
            success += 1
        except Exception as e:
            print(f"  ✗ 失败: {filename} ({e})")
            failed += 1
    
    # 汇总
    print("\n" + "="*60)
    print(f"完成: {success} 复制, {skipped} 已存在, {failed} 失败")
    
    if failed > 0:
        print("\n⚠️ 部分文件缺失，可能影响某些癌种的评估")
    else:
        print("\n✅ 所有数据文件准备就绪!")
    
    return failed == 0


def check_data():
    """检查数据完整性"""
    print("="*60)
    print("数据完整性检查")
    print("="*60)
    
    missing = []
    found = []
    
    for filename in REQUIRED_FILES:
        filepath = os.path.join(TARGET_DATA_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.1f}KB"
            found.append((filename, size_str))
        else:
            missing.append(filename)
    
    print(f"\n找到 {len(found)}/{len(REQUIRED_FILES)} 个文件:\n")
    
    for filename, size in found:
        print(f"  ✓ {filename} ({size})")
    
    if missing:
        print(f"\n缺失 {len(missing)} 个文件:\n")
        for filename in missing:
            print(f"  ✗ {filename}")
    
    return len(missing) == 0


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup data files')
    parser.add_argument('--check', action='store_true', help='Check data only')
    parser.add_argument('--symlink', action='store_true', help='Use symlinks instead of copying')
    parser.add_argument('--source', type=str, default=None, help='Source data directory')
    
    args = parser.parse_args()
    
    if args.check:
        check_data()
    else:
        setup_data(source_dir=args.source, use_symlink=args.symlink)



