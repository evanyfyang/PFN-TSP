#!/usr/bin/env python3
"""
测试脚本：验证LKH3算法得到的tour的edges是否都在candidate edges里面
"""

import sys
import os
import numpy as np
import torch
from typing import List, Dict, Tuple, Set

# 添加路径以导入模块
sys.path.append('2025/PFNs')
from pfns.priors.tsp_data_loader import TSPDataLoader
from pfns.priors.lkh3_wrapper import LKH3Wrapper


def get_tour_edges(tour: List[int]) -> Set[Tuple[int, int]]:
    """
    从tour中提取所有的edges（无向边）
    
    Args:
        tour: 节点访问顺序列表
        
    Returns:
        包含所有edges的集合，每个edge表示为(min_node, max_node)的元组
    """
    edges = set()
    n = len(tour)
    
    for i in range(n):
        current_node = tour[i]
        next_node = tour[(i + 1) % n]  # 最后一个节点连接到第一个节点
        
        # 确保edge的表示是标准化的（较小的节点在前）
        edge = (min(current_node, next_node), max(current_node, next_node))
        edges.add(edge)
    
    return edges


def get_candidate_edges(candidate_info: Dict) -> Set[Tuple[int, int]]:
    """
    从candidate信息中提取所有的candidate edges
    
    Args:
        candidate_info: LKH3返回的candidate信息字典
        
    Returns:
        包含所有candidate edges的集合
    """
    edges = set()
    
    if 'candidates' not in candidate_info:
        return edges
    
    for node_id, candidates in candidate_info['candidates'].items():
        for neighbor_id, alpha_value in candidates:
            # 转换为0-based索引（如果需要）并标准化edge表示
            node_0based = node_id - 1 if node_id > 0 else node_id
            neighbor_0based = neighbor_id - 1 if neighbor_id > 0 else neighbor_id
            
            edge = (min(node_0based, neighbor_0based), max(node_0based, neighbor_0based))
            edges.add(edge)
    
    return edges


def validate_tour_against_candidates(tour: List[int], candidate_info: Dict) -> Tuple[bool, List[Tuple[int, int]], Dict]:
    """
    验证tour中的所有edges是否都在candidate edges中
    
    Args:
        tour: 节点访问顺序列表
        candidate_info: LKH3返回的candidate信息字典
        
    Returns:
        Tuple of (is_valid, missing_edges, validation_info)
        - is_valid: 是否所有tour edges都在candidates中
        - missing_edges: 不在candidates中的tour edges列表
        - validation_info: 包含详细验证信息的字典
    """
    tour_edges = get_tour_edges(tour)
    candidate_edges = get_candidate_edges(candidate_info)
    
    # 找出不在candidate edges中的tour edges
    missing_edges = []
    for edge in tour_edges:
        if edge not in candidate_edges:
            missing_edges.append(edge)
    
    is_valid = len(missing_edges) == 0
    
    validation_info = {
        'total_tour_edges': len(tour_edges),
        'total_candidate_edges': len(candidate_edges),
        'missing_edges_count': len(missing_edges),
        'coverage_percentage': (len(tour_edges) - len(missing_edges)) / len(tour_edges) * 100 if tour_edges else 0,
        'tour_edges': tour_edges,
        'candidate_edges': candidate_edges
    }
    
    return is_valid, missing_edges, validation_info


def test_single_instance():
    """测试单个TSP实例"""
    print("=== 测试单个TSP实例 ===")
    
    # 创建一个简单的测试用例
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0], 
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5]
    ])
    
    print(f"测试坐标:\n{coords}")
    
    # 初始化LKH3 wrapper
    wrapper = LKH3Wrapper()
    
    # 测试不同的参数设置
    test_configs = [
        {'max_candidates': 5, 'alpha': None},
        {'max_candidates': 10, 'alpha': 0.5},
        {'max_candidates': 15, 'alpha': 1.0},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n--- 配置 {i+1}: {config} ---")
        
        try:
            tour, candidate_info = wrapper.solve_tsp_with_candidates(
                coords, 
                max_candidates=config['max_candidates'],
                alpha=config['alpha'],
                cleanup=True
            )
            
            print(f"最优tour: {tour}")
            print(f"图维度: {candidate_info['dimension']}")
            
            # 验证tour
            is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
            
            print(f"验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
            print(f"Tour edges总数: {validation_info['total_tour_edges']}")
            print(f"Candidate edges总数: {validation_info['total_candidate_edges']}")
            print(f"覆盖率: {validation_info['coverage_percentage']:.1f}%")
            
            if not is_valid:
                print(f"缺失的edges: {missing_edges}")
                print("详细分析:")
                print(f"  Tour edges: {validation_info['tour_edges']}")
                print(f"  Candidate edges: {validation_info['candidate_edges']}")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()


def test_dataloader_batch():
    """测试TSPDataLoader生成的批次"""
    print("\n=== 测试TSPDataLoader批次 ===")
    
    # 创建dataloader
    def eval_pos_sampler():
        return 0, 1  # 简单的评估位置采样器
    
    dataloader = TSPDataLoader(
        num_steps=1,  # 只测试一个批次
        batch_size=3,  # 小批次大小
        eval_pos_seq_len_sampler=eval_pos_sampler,
        seq_len_maximum=2,  # 每个批次2个图
        device='cpu',
        num_nodes_range=(5, 8),  # 5-8个节点
        num_processes=2,  # 减少进程数以便调试
        use_lkh3=True,
        max_candidates=10,
        alpha=1.0
    )
    
    print("开始生成批次...")
    
    # 获取一个批次
    batch = next(iter(dataloader))
    
    print(f"批次形状:")
    print(f"  x: {batch.x.shape}")
    print(f"  y: {batch.y.shape}")
    print(f"  candidate_info长度: {len(batch.candidate_info)}")
    
    # 验证批次中的每个实例
    total_instances = 0
    valid_instances = 0
    all_missing_edges = []
    
    seq_len, batch_size, num_nodes, _ = batch.x.shape
    
    for g in range(seq_len):
        for b in range(batch_size):
            instance_idx = g * batch_size + b
            
            # 获取坐标和tour
            coords = batch.x[g, b].numpy()
            tour = batch.y[g, b].tolist()
            candidate_info = batch.candidate_info[instance_idx]
            
            print(f"\n--- 实例 {instance_idx} (图{g}, 批次{b}) ---")
            print(f"节点数: {num_nodes}")
            print(f"Tour: {tour}")
            
            # 验证tour
            is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
            
            total_instances += 1
            if is_valid:
                valid_instances += 1
            else:
                all_missing_edges.extend(missing_edges)
            
            print(f"验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
            print(f"覆盖率: {validation_info['coverage_percentage']:.1f}%")
            
            if not is_valid:
                print(f"缺失的edges: {missing_edges}")
    
    # 总结
    print(f"\n=== 批次验证总结 ===")
    print(f"总实例数: {total_instances}")
    print(f"通过验证的实例数: {valid_instances}")
    print(f"通过率: {valid_instances/total_instances*100:.1f}%")
    
    if all_missing_edges:
        print(f"所有缺失edges总数: {len(all_missing_edges)}")
        print(f"唯一缺失edges: {set(all_missing_edges)}")


def analyze_candidate_structure():
    """分析candidate结构，帮助理解可能的问题"""
    print("\n=== 分析Candidate结构 ===")
    
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    
    wrapper = LKH3Wrapper()
    
    try:
        tour, candidate_info = wrapper.solve_tsp_with_candidates(
            coords, max_candidates=10, alpha=1.0, cleanup=True
        )
        
        print(f"坐标: {coords}")
        print(f"Tour: {tour}")
        print(f"Candidate info keys: {candidate_info.keys()}")
        print(f"维度: {candidate_info['dimension']}")
        print(f"MST parents: {candidate_info['mst_parents']}")
        
        print("\nCandidate edges详细信息:")
        for node_id, candidates in candidate_info['candidates'].items():
            print(f"节点 {node_id}: {candidates}")
            
        # 检查索引问题
        print(f"\nTour节点范围: {min(tour)} - {max(tour)}")
        candidate_nodes = list(candidate_info['candidates'].keys())
        print(f"Candidate节点范围: {min(candidate_nodes)} - {max(candidate_nodes)}")
        
        # 验证
        is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
        print(f"\n验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
        
        if not is_valid:
            print("问题分析:")
            print(f"Tour edges: {validation_info['tour_edges']}")
            print(f"Candidate edges: {validation_info['candidate_edges']}")
            print(f"缺失edges: {missing_edges}")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("LKH3 Tour验证测试")
    print("=" * 50)
    
    try:
        # 测试1: 单个实例
        test_single_instance()
        
        # 测试2: 分析candidate结构
        analyze_candidate_structure()
        
        # 测试3: dataloader批次
        test_dataloader_batch()
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 