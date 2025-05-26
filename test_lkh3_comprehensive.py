#!/usr/bin/env python3
"""
全面测试脚本：深入验证LKH3算法的candidate edges和tour一致性
"""

import sys
import os
import numpy as np
import torch
import time
from typing import List, Dict, Tuple, Set

# 添加路径以导入模块
sys.path.append('2025/PFNs')
from pfns.priors.tsp_data_loader import TSPDataLoader
from pfns.priors.lkh3_wrapper import LKH3Wrapper


def get_tour_edges(tour: List[int]) -> Set[Tuple[int, int]]:
    """从tour中提取所有的edges（无向边）"""
    edges = set()
    n = len(tour)
    
    for i in range(n):
        current_node = tour[i]
        next_node = tour[(i + 1) % n]
        edge = (min(current_node, next_node), max(current_node, next_node))
        edges.add(edge)
    
    return edges


def get_candidate_edges(candidate_info: Dict) -> Set[Tuple[int, int]]:
    """从candidate信息中提取所有的candidate edges"""
    edges = set()
    
    if 'candidates' not in candidate_info:
        return edges
    
    for node_id, candidates in candidate_info['candidates'].items():
        for neighbor_id, alpha_value in candidates:
            # LKH3使用1-based索引，转换为0-based
            node_0based = node_id - 1
            neighbor_0based = neighbor_id - 1
            
            edge = (min(node_0based, neighbor_0based), max(node_0based, neighbor_0based))
            edges.add(edge)
    
    return edges


def validate_tour_against_candidates(tour: List[int], candidate_info: Dict) -> Tuple[bool, List[Tuple[int, int]], Dict]:
    """验证tour中的所有edges是否都在candidate edges中"""
    tour_edges = get_tour_edges(tour)
    candidate_edges = get_candidate_edges(candidate_info)
    
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


def test_various_sizes():
    """测试不同大小的TSP实例"""
    print("=== 测试不同大小的TSP实例 ===")
    
    wrapper = LKH3Wrapper()
    
    # 测试不同的实例大小
    test_sizes = [4, 6, 8, 10, 15, 20]
    
    for size in test_sizes:
        print(f"\n--- 测试 {size} 个节点 ---")
        
        # 生成随机坐标
        coords = np.random.uniform(0, 1, (size, 2))
        
        try:
            start_time = time.time()
            tour, candidate_info = wrapper.solve_tsp_with_candidates(
                coords, 
                max_candidates=min(15, size-1),  # 确保不超过节点数-1
                alpha=1.0,
                cleanup=True
            )
            solve_time = time.time() - start_time
            
            # 验证tour
            is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
            
            print(f"求解时间: {solve_time:.2f}s")
            print(f"Tour长度: {len(tour)}")
            print(f"验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
            print(f"Tour edges: {validation_info['total_tour_edges']}")
            print(f"Candidate edges: {validation_info['total_candidate_edges']}")
            print(f"覆盖率: {validation_info['coverage_percentage']:.1f}%")
            
            if not is_valid:
                print(f"缺失的edges: {missing_edges}")
                
        except Exception as e:
            print(f"错误: {e}")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    wrapper = LKH3Wrapper()
    
    # 测试用例1: 最小TSP (3个节点)
    print("\n--- 测试用例1: 3个节点的三角形 ---")
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, 0.866]  # 等边三角形
    ])
    
    try:
        tour, candidate_info = wrapper.solve_tsp_with_candidates(
            coords, max_candidates=5, alpha=1.0, cleanup=True
        )
        
        is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
        print(f"Tour: {tour}")
        print(f"验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
        print(f"覆盖率: {validation_info['coverage_percentage']:.1f}%")
        
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试用例2: 共线点
    print("\n--- 测试用例2: 共线点 ---")
    coords = np.array([
        [0.0, 0.0],
        [0.25, 0.0],
        [0.5, 0.0],
        [0.75, 0.0],
        [1.0, 0.0]
    ])
    
    try:
        tour, candidate_info = wrapper.solve_tsp_with_candidates(
            coords, max_candidates=8, alpha=1.0, cleanup=True
        )
        
        is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
        print(f"Tour: {tour}")
        print(f"验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
        print(f"覆盖率: {validation_info['coverage_percentage']:.1f}%")
        
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试用例3: 正方形网格
    print("\n--- 测试用例3: 正方形网格 ---")
    coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]
    ])
    
    try:
        tour, candidate_info = wrapper.solve_tsp_with_candidates(
            coords, max_candidates=12, alpha=1.0, cleanup=True
        )
        
        is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
        print(f"Tour: {tour}")
        print(f"验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
        print(f"覆盖率: {validation_info['coverage_percentage']:.1f}%")
        
    except Exception as e:
        print(f"错误: {e}")


def test_different_alpha_values():
    """测试不同的alpha值对candidate edges的影响"""
    print("\n=== 测试不同Alpha值的影响 ===")
    
    wrapper = LKH3Wrapper()
    
    # 固定坐标
    coords = np.random.uniform(0, 1, (10, 2))
    np.random.seed(42)  # 确保可重复性
    coords = np.random.uniform(0, 1, (10, 2))
    
    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for alpha in alpha_values:
        print(f"\n--- Alpha = {alpha} ---")
        
        try:
            tour, candidate_info = wrapper.solve_tsp_with_candidates(
                coords, max_candidates=15, alpha=alpha, cleanup=True
            )
            
            is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
            
            print(f"Tour: {tour}")
            print(f"验证结果: {'✓ 通过' if is_valid else '✗ 失败'}")
            print(f"Candidate edges总数: {validation_info['total_candidate_edges']}")
            print(f"覆盖率: {validation_info['coverage_percentage']:.1f}%")
            
            if not is_valid:
                print(f"缺失的edges: {missing_edges}")
                
        except Exception as e:
            print(f"错误: {e}")


def test_large_batch():
    """测试大批次的数据"""
    print("\n=== 测试大批次数据 ===")
    
    def eval_pos_sampler():
        return 0, 1
    
    dataloader = TSPDataLoader(
        num_steps=1,
        batch_size=5,  # 更大的批次
        eval_pos_seq_len_sampler=eval_pos_sampler,
        seq_len_maximum=3,  # 更多的图
        device='cpu',
        num_nodes_range=(8, 12),  # 更大的节点范围
        num_processes=4,
        use_lkh3=True,
        max_candidates=15,
        alpha=1.0
    )
    
    print("生成大批次数据...")
    start_time = time.time()
    
    batch = next(iter(dataloader))
    generation_time = time.time() - start_time
    
    print(f"批次生成时间: {generation_time:.2f}s")
    print(f"批次形状: x={batch.x.shape}, y={batch.y.shape}")
    
    # 验证所有实例
    total_instances = 0
    valid_instances = 0
    total_edges = 0
    total_candidate_edges = 0
    
    seq_len, batch_size, num_nodes, _ = batch.x.shape
    
    for g in range(seq_len):
        for b in range(batch_size):
            instance_idx = g * batch_size + b
            
            tour = batch.y[g, b].tolist()
            candidate_info = batch.candidate_info[instance_idx]
            
            is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
            
            total_instances += 1
            if is_valid:
                valid_instances += 1
            
            total_edges += validation_info['total_tour_edges']
            total_candidate_edges += validation_info['total_candidate_edges']
    
    print(f"\n=== 大批次验证总结 ===")
    print(f"总实例数: {total_instances}")
    print(f"通过验证的实例数: {valid_instances}")
    print(f"通过率: {valid_instances/total_instances*100:.1f}%")
    print(f"平均tour edges: {total_edges/total_instances:.1f}")
    print(f"平均candidate edges: {total_candidate_edges/total_instances:.1f}")


def analyze_candidate_distribution():
    """分析candidate edges的分布特性"""
    print("\n=== 分析Candidate Edges分布 ===")
    
    wrapper = LKH3Wrapper()
    
    # 生成多个实例进行统计分析
    num_instances = 10
    node_count = 10
    
    all_candidate_counts = []
    all_coverage_rates = []
    
    for i in range(num_instances):
        coords = np.random.uniform(0, 1, (node_count, 2))
        
        try:
            tour, candidate_info = wrapper.solve_tsp_with_candidates(
                coords, max_candidates=15, alpha=1.0, cleanup=True
            )
            
            is_valid, missing_edges, validation_info = validate_tour_against_candidates(tour, candidate_info)
            
            all_candidate_counts.append(validation_info['total_candidate_edges'])
            all_coverage_rates.append(validation_info['coverage_percentage'])
            
        except Exception as e:
            print(f"实例 {i} 错误: {e}")
    
    if all_candidate_counts:
        print(f"Candidate edges统计 (基于{len(all_candidate_counts)}个实例):")
        print(f"  平均: {np.mean(all_candidate_counts):.1f}")
        print(f"  最小: {np.min(all_candidate_counts)}")
        print(f"  最大: {np.max(all_candidate_counts)}")
        print(f"  标准差: {np.std(all_candidate_counts):.1f}")
        
        print(f"\n覆盖率统计:")
        print(f"  平均: {np.mean(all_coverage_rates):.1f}%")
        print(f"  最小: {np.min(all_coverage_rates):.1f}%")
        print(f"  最大: {np.max(all_coverage_rates):.1f}%")
        
        # 理论上完全图的边数
        complete_graph_edges = node_count * (node_count - 1) // 2
        avg_sparsity = np.mean(all_candidate_counts) / complete_graph_edges * 100
        print(f"\n图的稀疏性:")
        print(f"  完全图边数: {complete_graph_edges}")
        print(f"  平均candidate edges占比: {avg_sparsity:.1f}%")


def main():
    """主函数"""
    print("LKH3 全面验证测试")
    print("=" * 60)
    
    try:
        # 测试1: 不同大小的实例
        test_various_sizes()
        
        # 测试2: 边界情况
        test_edge_cases()
        
        # 测试3: 不同alpha值
        test_different_alpha_values()
        
        # 测试4: 大批次数据
        test_large_batch()
        
        # 测试5: candidate分布分析
        analyze_candidate_distribution()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 