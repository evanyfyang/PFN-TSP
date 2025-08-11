#!/usr/bin/env python3
"""
检查TSP数据集结构，特别是fix_group_nodes_same_size策略
"""

import pickle
import numpy as np
import torch
from collections import defaultdict

def check_dataset_structure(dataset_path):
    """检查数据集的结构和特性"""
    print(f"检查数据集: {dataset_path}")
    print("=" * 60)
    
    # 加载数据集
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"数据集类型: {type(dataset)}")
    print(f"数据集键: {list(dataset.keys())}")
    
    # 分析每个节点数的数据
    for num_nodes in sorted(dataset.keys()):
        instances = dataset[num_nodes]
        print(f"\n=== 节点数 {num_nodes} ===")
        print(f"实例总数: {len(instances)}")
        
        if len(instances) == 0:
            print("  无实例")
            continue
        
        # 检查第一个实例的结构
        first_instance = instances[0]
        print(f"实例类型: {type(first_instance)}")
        print(f"实例键: {list(first_instance.keys()) if isinstance(first_instance, dict) else 'N/A'}")
        
        if isinstance(first_instance, dict):
            print(f"坐标数量: {len(first_instance.get('coords', []))}")
            print(f"路径长度: {len(first_instance.get('tour', []))}")
            print(f"候选信息: {type(first_instance.get('candidate_info', {}))}")
        
        # 检查是否有分组结构
        if isinstance(instances[0], list):
            print(f"  数据以列表形式组织，每个列表包含 {len(instances[0])} 个实例")
            print(f"  总组数: {len(instances)}")
            
            # 检查每组的大小
            group_sizes = [len(group) for group in instances]
            print(f"  组大小统计:")
            print(f"    最小: {min(group_sizes)}")
            print(f"    最大: {max(group_sizes)}")
            print(f"    平均: {np.mean(group_sizes):.1f}")
            print(f"    标准差: {np.std(group_sizes):.1f}")
            
            # 检查是否有重复
            check_duplicates_in_groups(instances, num_nodes)
            
        elif isinstance(instances[0], dict):
            # 检查是否有group_id字段
            has_group_id = 'group_id' in instances[0]
            print(f"  数据以字典形式组织，包含group_id: {has_group_id}")
            
            if has_group_id:
                # 按group_id分组
                groups = defaultdict(list)
                for instance in instances:
                    group_id = instance.get('group_id', 0)
                    groups[group_id].append(instance)
                
                print(f"  检测到 {len(groups)} 个组")
                group_sizes = [len(group) for group in groups.values()]
                print(f"  组大小统计:")
                print(f"    最小: {min(group_sizes)}")
                print(f"    最大: {max(group_sizes)}")
                print(f"    平均: {np.mean(group_sizes):.1f}")
                print(f"    标准差: {np.std(group_sizes):.1f}")
                
                # 检查重复
                check_duplicates_in_dict_groups(groups, num_nodes)
            else:
                print("  没有group_id字段，数据可能是随机排列的")
        
        # 检查坐标重复情况
        check_coordinate_duplicates(instances, num_nodes)

def check_duplicates_in_groups(groups, num_nodes):
    """检查列表形式分组中的重复情况"""
    print(f"  检查重复情况:")
    
    # 检查组内重复
    for group_idx, group in enumerate(groups[:3]):  # 只检查前3组
        print(f"    组 {group_idx}:")
        
        # 提取所有坐标
        all_coords = []
        for instance in group:
            if isinstance(instance, dict):
                coords = instance.get('coords', [])
                all_coords.extend(coords)
        
        # 检查坐标重复
        coord_tuples = [tuple(coord) for coord in all_coords]
        unique_coords = set(coord_tuples)
        total_coords = len(coord_tuples)
        unique_count = len(unique_coords)
        
        print(f"      总坐标数: {total_coords}")
        print(f"      唯一坐标数: {unique_count}")
        print(f"      重复率: {(total_coords - unique_count) / total_coords * 100:.1f}%")
        
        # 检查是否来自1.4倍节点数的采样
        expected_unique = int(num_nodes * 1.4)
        print(f"      期望唯一坐标数 (1.4 * {num_nodes}): {expected_unique}")
        print(f"      实际唯一坐标数: {unique_count}")
        print(f"      是否接近1.4倍: {abs(unique_count - expected_unique) <= 2}")

def check_duplicates_in_dict_groups(groups, num_nodes):
    """检查字典形式分组中的重复情况"""
    print(f"  检查重复情况:")
    
    for group_id, group in list(groups.items())[:3]:  # 只检查前3组
        print(f"    组 {group_id}:")
        
        # 提取所有坐标
        all_coords = []
        for instance in group:
            coords = instance.get('coords', [])
            all_coords.extend(coords)
        
        # 检查坐标重复
        coord_tuples = [tuple(coord) for coord in all_coords]
        unique_coords = set(coord_tuples)
        total_coords = len(coord_tuples)
        unique_count = len(unique_coords)
        
        print(f"      总坐标数: {total_coords}")
        print(f"      唯一坐标数: {unique_count}")
        print(f"      重复率: {(total_coords - unique_count) / total_coords * 100:.1f}%")
        
        # 检查是否来自1.4倍节点数的采样
        expected_unique = int(num_nodes * 1.4)
        print(f"      期望唯一坐标数 (1.4 * {num_nodes}): {expected_unique}")
        print(f"      实际唯一坐标数: {unique_count}")
        print(f"      是否接近1.4倍: {abs(unique_count - expected_unique) <= 2}")

def check_coordinate_duplicates(instances, num_nodes):
    """检查坐标重复情况"""
    print(f"  坐标重复分析:")
    
    # 收集所有坐标
    all_coords = []
    for instance in instances:
        if isinstance(instance, dict):
            coords = instance.get('coords', [])
            all_coords.extend(coords)
        elif isinstance(instance, list):
            for sub_instance in instance:
                if isinstance(sub_instance, dict):
                    coords = sub_instance.get('coords', [])
                    all_coords.extend(coords)
    
    if not all_coords:
        print("    无坐标数据")
        return
    
    # 统计重复
    coord_tuples = [tuple(coord) for coord in all_coords]
    unique_coords = set(coord_tuples)
    total_coords = len(coord_tuples)
    unique_count = len(unique_coords)
    
    print(f"    总坐标数: {total_coords}")
    print(f"    唯一坐标数: {unique_count}")
    print(f"    重复率: {(total_coords - unique_count) / total_coords * 100:.1f}%")
    
    # 检查是否来自1.4倍节点数的采样
    expected_unique = int(num_nodes * 1.4)
    print(f"    期望唯一坐标数 (1.4 * {num_nodes}): {expected_unique}")
    print(f"    实际唯一坐标数: {unique_count}")
    print(f"    是否接近1.4倍: {abs(unique_count - expected_unique) <= 2}")

def simulate_batch_creation(dataset_path, batch_size=2, seq_len=5):
    """模拟批次创建过程"""
    print(f"\n=== 模拟批次创建 (batch_size={batch_size}, seq_len={seq_len}) ===")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # 选择一个节点数进行测试
    test_num_nodes = list(dataset.keys())[0]
    instances = dataset[test_num_nodes]
    
    print(f"测试节点数: {test_num_nodes}")
    print(f"实例总数: {len(instances)}")
    
    if isinstance(instances[0], list):
        # 列表形式
        print("数据以列表形式组织")
        
        # 模拟批次创建
        total_instances_needed = batch_size * seq_len
        print(f"每个批次需要 {total_instances_needed} 个实例")
        
        # 检查第一组是否有足够的实例
        first_group = instances[0]
        print(f"第一组实例数: {len(first_group)}")
        
        if len(first_group) >= total_instances_needed:
            batch_instances = first_group[:total_instances_needed]
            print(f"可以从第一组创建批次")
        else:
            print(f"第一组实例不足，需要跨组")
            batch_instances = []
            for group in instances:
                batch_instances.extend(group)
                if len(batch_instances) >= total_instances_needed:
                    batch_instances = batch_instances[:total_instances_needed]
                    break
        
        # 分析批次中的坐标
        all_batch_coords = []
        for instance in batch_instances:
            if isinstance(instance, dict):
                coords = instance.get('coords', [])
                all_batch_coords.extend(coords)
        
        coord_tuples = [tuple(coord) for coord in all_batch_coords]
        unique_coords = set(coord_tuples)
        total_coords = len(coord_tuples)
        unique_count = len(unique_coords)
        
        print(f"批次中总坐标数: {total_coords}")
        print(f"批次中唯一坐标数: {unique_count}")
        print(f"批次中重复率: {(total_coords - unique_count) / total_coords * 100:.1f}%")
        
        # 检查是否来自1.4倍节点数的采样
        expected_unique = int(test_num_nodes * 1.4)
        print(f"期望唯一坐标数 (1.4 * {test_num_nodes}): {expected_unique}")
        print(f"实际唯一坐标数: {unique_count}")
        print(f"是否接近1.4倍: {abs(unique_count - expected_unique) <= 2}")

if __name__ == "__main__":
    dataset_path = "/local-scratchg/yifan/2025/PFNs/datasets/TSP_51_60_5120_0730__fix_group_nodes_same_size.pkl"
    
    try:
        check_dataset_structure(dataset_path)
        simulate_batch_creation(dataset_path, batch_size=2, seq_len=5)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc() 