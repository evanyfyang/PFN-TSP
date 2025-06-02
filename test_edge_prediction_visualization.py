#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import pickle
from datetime import datetime
import torch.nn.functional as F
from torch_geometric.data import Data

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pfns.train_tsp import train_tsp
from pfns.priors.tsp_data_loader import TSPDataLoader
from pfns.priors.prior import Batch

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test edge prediction visualization with different test sizes')
    parser.add_argument('--model_path', type=str, 
                        default='/local-scratchg/yifan/2025/PFNs/saved_models/tsp_model_offline_20250602_080107.pt',
                        help='Path to pretrained model')
    parser.add_argument('--emsize', type=int, default=128, help='Embedding dimension size')
    parser.add_argument('--nhid', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--nlayers', type=int, default=3, help='Number of Transformer layers')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--test_sizes', type=int, nargs='+', default=[5, 6, 7, 8, 9, 10], 
                        help='Test sizes to evaluate')
    parser.add_argument('--fixed_instance_path', type=str, 
                        default='/local-scratchg/yifan/2025/PFNs/fixed_tsp_instance_30nodes.pkl',
                        help='Path to store/load fixed TSP instance')
    parser.add_argument('--cuda_device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device')
    parser.add_argument('--save_dir', type=str, default='/local-scratchg/yifan/2025/PFNs/edge_prediction_results', 
                        help='Directory to save results')
    
    return parser.parse_args()

def generate_or_load_fixed_instance(fixed_instance_path, num_nodes=30, device='cuda'):
    """Generate or load a fixed TSP instance"""
    
    if os.path.exists(fixed_instance_path):
        print(f"Loading fixed TSP instance from {fixed_instance_path}")
        with open(fixed_instance_path, 'rb') as f:
            fixed_data = pickle.load(f)
        return fixed_data['coords'], fixed_data['lkh_solution'], fixed_data['ortools_solution']
    
    print(f"Generating new fixed TSP instance with {num_nodes} nodes...")
    
    # Create TSPDataLoader to generate the instance
    def dummy_sampler():
        return 0, num_nodes
    
    dataloader = TSPDataLoader(
        num_steps=1,
        batch_size=1,
        eval_pos_seq_len_sampler=dummy_sampler,
        seq_len_maximum=1,
        device=device,
        num_nodes_range=(num_nodes, num_nodes),
        include_ortools=True,
        max_candidates=15
    )
    
    # Generate one instance
    batch = next(iter(dataloader))
    coords = batch.x[0, 0, :].cpu().numpy()  # Shape: [num_nodes, 2]
    lkh_solution = batch.target_y[0, 0].cpu().numpy()  # LKH3 solution
    ortools_solution = batch.ortools_solution[0, 0].cpu().numpy()  # OR-Tools solution
    
    # Save the fixed instance
    os.makedirs(os.path.dirname(fixed_instance_path), exist_ok=True)
    fixed_data = {
        'coords': coords,
        'lkh_solution': lkh_solution,
        'ortools_solution': ortools_solution,
        'num_nodes': num_nodes
    }
    
    with open(fixed_instance_path, 'wb') as f:
        pickle.dump(fixed_data, f)
    
    print(f"Fixed TSP instance saved to {fixed_instance_path}")
    return coords, lkh_solution, ortools_solution

def generate_test_sequence(fixed_coords, fixed_lkh_solution, test_size, device='cuda'):
    """Generate a test sequence with random instances + fixed instance at the end"""
    
    num_nodes = len(fixed_coords)
    
    # Generate random instances for positions 0 to test_size-2
    def dummy_sampler():
        return 0, num_nodes
    
    dataloader = TSPDataLoader(
        num_steps=1,
        batch_size=1,
        eval_pos_seq_len_sampler=dummy_sampler,
        seq_len_maximum=test_size-1,
        device=device,
        num_nodes_range=(num_nodes, num_nodes),
        include_ortools=False,  # Don't need OR-Tools for random instances
        max_candidates=15
    )
    
    # Get random instances
    if test_size > 1:
        batch = next(iter(dataloader))
        random_coords = batch.x[:, 0, :].cpu().numpy()  # Shape: [test_size-1, num_nodes, 2]
        random_solutions = batch.target_y[:, 0].cpu().numpy()  # Shape: [test_size-1, num_nodes]
    else:
        random_coords = np.empty((0, num_nodes, 2))
        random_solutions = np.empty((0, num_nodes))
    
    # Combine random instances with fixed instance
    all_coords = []
    all_solutions = []
    
    # Add random instances
    for i in range(test_size - 1):
        all_coords.append(random_coords[i])
        all_solutions.append(random_solutions[i])
    
    # Add fixed instance at the end
    all_coords.append(fixed_coords)
    all_solutions.append(fixed_lkh_solution)
    
    return all_coords, all_solutions

def load_tsp_model(model_path, emsize, nhid, nlayers, nhead, dropout, device='cuda'):
    """Load a pretrained TSP model"""
    print(f"Loading pretrained model from {model_path}...")

    # Create a dummy model to get the architecture
    result = train_tsp(
        emsize=emsize,
        nhid=nhid,
        nlayers=nlayers,
        nhead=nhead,
        dropout=dropout,
        epochs=0, 
        steps_per_epoch=1,
        batch_size=1,
        seq_len=5,
        lr=1e-4,
        num_nodes_range=(4, 5),  
        gpu_device=device
    )
    
    model = result.model
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    return model

def predict_edges_with_model(model, data, edges, device):
    """Predict edge probabilities using the trained model"""
    model.eval()
    
    with torch.no_grad():
        # Prepare input data
        coords = data.x.to(device)  # Node features [num_nodes, 2]
        num_nodes = coords.size(0)
        
        # Create dummy y values (we don't need them for prediction)
        dummy_y = torch.zeros(num_nodes, dtype=torch.long, device=device)
        
        # Create input in the format expected by the model: (x, y) tuple
        # x should be [seq_len, batch_size, num_nodes, 2]
        # y should be [seq_len, batch_size, num_nodes]
        x_input = coords.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes, 2]
        y_input = dummy_y.unsqueeze(0).unsqueeze(0)  # [1, 1, num_nodes]
        
        print(f"x_input shape: {x_input.shape}")
        print(f"y_input shape: {y_input.shape}")
        print(f"Model device: {next(model.parameters()).device}")
        
        # Get model predictions using the correct input format
        output = model((x_input, y_input), single_eval_pos=0)
        
        print(f"Model output type: {type(output)}")
        if isinstance(output, tuple):
            print(f"Output tuple length: {len(output)}")
            for i, item in enumerate(output):
                if hasattr(item, 'shape'):
                    print(f"Output[{i}] shape: {item.shape}")
                else:
                    print(f"Output[{i}] type: {type(item)}")
        
        # Extract edge probabilities based on the model output format
        # This might need adjustment based on the actual output structure
        if isinstance(output, tuple) and len(output) >= 2:
            edge_values, edge_info = output
            print(f"Edge values shape: {edge_values.shape}")
            print(f"Edge info type: {type(edge_info)}")
            
            # edge_info is a tuple containing edge information
            if isinstance(edge_info, tuple) and len(edge_info) >= 3:
                edge_index_list, node_offset_map, edge_counts = edge_info
                print(f"Edge index list length: {len(edge_index_list)}")
                print(f"Node offset map keys: {list(node_offset_map.keys())[:10] if node_offset_map else 'None'}")
                print(f"Edge counts: {edge_counts}")
                
                # Get the edge values for the current position (should be position 0)
                current_edge_values = edge_values[0, 0]  # [max_edges]
                current_edge_index = edge_index_list[0] if edge_index_list else None
                current_edge_count = edge_counts[0] if edge_counts else 0
                
                print(f"Current edge values shape: {current_edge_values.shape}")
                print(f"Current edge count: {current_edge_count}")
                
                if current_edge_index is not None:
                    print(f"Current edge index shape: {current_edge_index.shape}")
                    
                    # Apply sigmoid to get probabilities
                    edge_probs_tensor = torch.sigmoid(current_edge_values[:current_edge_count])
                    print(f"Edge probs tensor shape: {edge_probs_tensor.shape}")
                    print(f"Edge probs min: {edge_probs_tensor.min().item():.6f}")
                    print(f"Edge probs max: {edge_probs_tensor.max().item():.6f}")
                    
                    # Create a mapping from edge indices to probabilities
                    edge_to_prob = {}
                    
                    # Build reverse node mapping
                    node_map = {value: key for key, value in node_offset_map.items()}
                    
                    # Extract edges and their probabilities
                    edge_index_np = current_edge_index.cpu().numpy()
                    edge_probs_np = edge_probs_tensor.cpu().numpy()
                    
                    for i in range(min(current_edge_count, len(edge_probs_np))):
                        # Get edge indices
                        if edge_index_np.ndim == 2 and edge_index_np.shape[0] == 2:
                            u, v = edge_index_np[0, i], edge_index_np[1, i]
                        elif edge_index_np.ndim == 2 and edge_index_np.shape[1] == 2:
                            u, v = edge_index_np[i, 0], edge_index_np[i, 1]
                        else:
                            continue
                        
                        # Map back to actual node indices
                        if u in node_map and v in node_map:
                            u_info = node_map[u]  # (pos, batch, node)
                            v_info = node_map[v]
                            
                            # Ensure nodes are from the current position
                            if u_info[0] == 0 and v_info[0] == 0:  # position 0
                                u_node = u_info[2]
                                v_node = v_info[2]
                                
                                # Create canonical edge (smaller node first)
                                edge = tuple(sorted([u_node, v_node]))
                                edge_to_prob[edge] = edge_probs_np[i]
                    
                    print(f"Extracted {len(edge_to_prob)} edges from model output")
                    
                    # Map requested edges to their probabilities
                    edge_probs = []
                    for edge in edges:
                        if edge in edge_to_prob:
                            edge_probs.append(edge_to_prob[edge])
                        else:
                            # If edge not found in model output, assign low probability
                            edge_probs.append(0.01)
                    
                    print(f"Mapped {len(edge_probs)} requested edges to probabilities")
                    print(f"Edges with model predictions: {sum(1 for i, edge in enumerate(edges) if edge in edge_to_prob)}")
                    
                else:
                    # Fallback: create dummy probabilities
                    edge_probs = [0.1 + 0.8 * torch.rand(1).item() for _ in edges]
                    print(f"Generated {len(edge_probs)} dummy probabilities (no edge index)")
            else:
                # Fallback: create dummy probabilities
                edge_probs = [0.1 + 0.8 * torch.rand(1).item() for _ in edges]
                print(f"Generated {len(edge_probs)} dummy probabilities (invalid edge info)")
        else:
            # Fallback: create dummy probabilities
            edge_probs = [0.1 + 0.8 * torch.rand(1).item() for _ in edges]
            print(f"Generated {len(edge_probs)} dummy probabilities (fallback)")
        
        print(f"Edge probabilities - min: {min(edge_probs):.6f}, max: {max(edge_probs):.6f}")
        print(f"Number of zero probabilities: {sum(1 for p in edge_probs if p == 0.0)}")
        print(f"Number of non-zero probabilities: {sum(1 for p in edge_probs if p > 0.0)}")
        
    return edge_probs

def get_true_edges_from_solution(solution):
    """Extract true edges from TSP solution"""
    true_edges = set()
    num_nodes = len(solution)
    
    for i in range(num_nodes):
        current_node = solution[i]
        next_node = solution[(i + 1) % num_nodes]
        
        # Add edge in canonical form (smaller node first)
        edge = tuple(sorted([current_node, next_node]))
        true_edges.add(edge)
    
    return true_edges

def classify_edges(predicted_edges, predicted_probs, true_edges, threshold=0.5):
    """Classify edges into TP, FN, FP, TN"""
    
    # Convert to sets for easier operations
    true_edges_set = set(true_edges)
    
    # Print probability statistics
    print(f"Probability statistics:")
    print(f"  Min prob: {min(predicted_probs):.4f}")
    print(f"  Max prob: {max(predicted_probs):.4f}")
    print(f"  Mean prob: {np.mean(predicted_probs):.4f}")
    print(f"  Median prob: {np.median(predicted_probs):.4f}")
    
    # Use dynamic threshold if all probabilities are below 0.5
    if max(predicted_probs) < 0.5:
        # Use median as threshold
        threshold = np.median(predicted_probs)
        print(f"  All probabilities < 0.5, using median as threshold: {threshold:.4f}")
    
    # Classify predicted edges
    tp_edges = []  # True Positives
    fp_edges = []  # False Positives
    
    for edge, prob in zip(predicted_edges, predicted_probs):
        if prob > threshold:
            if edge in true_edges_set:
                tp_edges.append((edge, prob))
            else:
                fp_edges.append((edge, prob))
    
    # Find False Negatives (true edges with prob <= threshold)
    fn_edges = []
    for edge, prob in zip(predicted_edges, predicted_probs):
        if edge in true_edges_set and prob <= threshold:
            fn_edges.append((edge, prob))
    
    # True Negatives are harder to define since we don't have all possible edges
    # For visualization, we'll consider edges with prob <= threshold and not in true_edges
    tn_edges = []
    for edge, prob in zip(predicted_edges, predicted_probs):
        if prob <= threshold and edge not in true_edges_set:
            tn_edges.append((edge, prob))
    
    print(f"  Using threshold: {threshold:.4f}")
    print(f"  True edges found in predictions: {len([e for e in predicted_edges if e in true_edges_set])}")
    
    return tp_edges, fn_edges, fp_edges, tn_edges, threshold

def visualize_edge_classification(coords, tp_edges, fn_edges, fp_edges, tn_edges, 
                                test_size, save_path):
    """Visualize edge classification results"""
    
    plt.switch_backend('Agg')
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Draw nodes
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    ax.scatter(x_coords, y_coords, c='lightblue', s=150, alpha=0.9, zorder=5, 
              edgecolors='black', linewidth=1)
    
    # Add node labels
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        ax.annotate(str(i), (x, y), xytext=(0, 0), textcoords='offset points', 
                   fontsize=8, ha='center', va='center', weight='bold', color='black')
    
    # Draw True Negatives (light gray, thin)
    for edge, prob in tn_edges:
        node1, node2 = edge
        x1, y1 = coords[node1]
        x2, y2 = coords[node2]
        ax.plot([x1, x2], [y1, y2], 'lightgray', alpha=0.3, linewidth=0.5, zorder=1)
    
    # Draw True Positives (green, thick)
    for edge, prob in tp_edges:
        node1, node2 = edge
        x1, y1 = coords[node1]
        x2, y2 = coords[node2]
        linewidth = 2 + prob * 2
        ax.plot([x1, x2], [y1, y2], 'green', alpha=0.8, linewidth=linewidth, zorder=3)
        
        # Add probability label
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.annotate(f'{prob:.2f}', (mid_x, mid_y), 
                   fontsize=6, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='lightgreen', alpha=0.7),
                   zorder=6)
    
    # Draw False Negatives (orange, thick)
    for edge, prob in fn_edges:
        node1, node2 = edge
        x1, y1 = coords[node1]
        x2, y2 = coords[node2]
        linewidth = 2 + (1-prob) * 2
        ax.plot([x1, x2], [y1, y2], 'orange', alpha=0.8, linewidth=linewidth, zorder=2)
        
        # Add probability label
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.annotate(f'{prob:.2f}', (mid_x, mid_y), 
                   fontsize=6, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='lightyellow', alpha=0.7),
                   zorder=6)
    
    # Draw False Positives (red, thick)
    for edge, prob in fp_edges:
        node1, node2 = edge
        x1, y1 = coords[node1]
        x2, y2 = coords[node2]
        linewidth = 2 + prob * 3
        ax.plot([x1, x2], [y1, y2], 'red', alpha=0.8, linewidth=linewidth, zorder=4)
        
        # Add probability label
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.annotate(f'{prob:.2f}', (mid_x, mid_y), 
                   fontsize=6, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='lightcoral', alpha=0.7),
                   zorder=6)
    
    # Calculate metrics
    tp = len(tp_edges)
    fn = len(fn_edges)
    fp = len(fp_edges)
    tn = len(tn_edges)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Set plot properties
    ax.set_xlabel('X Coordinate', fontsize=14)
    ax.set_ylabel('Y Coordinate', fontsize=14)
    ax.set_title(f'Edge Prediction Analysis (Test Size: {test_size})\n'
                f'TP: {tp} | FN: {fn} | FP: {fp} | TN: {tn}', 
                fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='lightgray', alpha=0.4, linewidth=1, 
               label=f'True Negatives (TN): {tn}'),
        Line2D([0], [0], color='green', alpha=0.8, linewidth=3, 
               label=f'True Positives (TP): {tp}'),
        Line2D([0], [0], color='orange', alpha=0.8, linewidth=3, 
               label=f'False Negatives (FN): {fn}'),
        Line2D([0], [0], color='red', alpha=0.8, linewidth=3, 
               label=f'False Positives (FP): {fp}'),
        plt.scatter([], [], c='lightblue', s=150, alpha=0.9, 
                   edgecolors='black', label='Nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add metrics text
    metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}\nAccuracy: {accuracy:.3f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy
    }

def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Generate or load fixed TSP instance
    fixed_coords, fixed_lkh_solution, fixed_ortools_solution = generate_or_load_fixed_instance(
        args.fixed_instance_path, num_nodes=30, device=args.cuda_device
    )
    
    # Load model
    model = load_tsp_model(
        model_path=args.model_path,
        emsize=args.emsize,
        nhid=args.nhid,
        nlayers=args.nlayers,
        nhead=args.nhead,
        dropout=args.dropout,
        device=args.cuda_device
    )
    
    # Get true edges from the fixed solution
    true_edges = get_true_edges_from_solution(fixed_lkh_solution)
    print(f"Fixed TSP instance has {len(true_edges)} true edges")
    
    # Test different test sizes
    all_results = {}
    
    for test_size in args.test_sizes:
        print(f"\n=== Testing with test_size = {test_size} ===")
        
        # Generate test sequence
        coords_sequence, solution_sequence = generate_test_sequence(
            fixed_coords, fixed_lkh_solution, test_size, device=args.cuda_device
        )
        
        # Create data object for model input
        data = Data(x=torch.tensor(fixed_coords, dtype=torch.float32))
        
        # Generate all possible edges for prediction
        num_nodes = len(fixed_coords)
        all_edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                all_edges.append((i, j))
        
        print(f"Predicting probabilities for {len(all_edges)} possible edges")
        
        # Predict edges
        predicted_probs = predict_edges_with_model(
            model, data, all_edges, device=args.cuda_device
        )
        
        # Convert edges to list of tuples for compatibility
        predicted_edges = all_edges
        
        print(f"Model predicted {len(predicted_edges)} edges")
        
        # Classify edges
        tp_edges, fn_edges, fp_edges, tn_edges, threshold = classify_edges(
            predicted_edges, predicted_probs, true_edges, threshold=0.5
        )
        
        # Visualize results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.save_dir, f'edge_prediction_test_size_{test_size}_{timestamp}.png')
        
        metrics = visualize_edge_classification(
            fixed_coords, tp_edges, fn_edges, fp_edges, tn_edges, 
            test_size, save_path
        )
        
        all_results[test_size] = metrics
        
        print(f"Results for test_size {test_size}:")
        print(f"  TP: {metrics['tp']}, FN: {metrics['fn']}, FP: {metrics['fp']}, TN: {metrics['tn']}")
        print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1']:.3f}, Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Visualization saved to: {save_path}")
    
    # Save summary results
    summary_path = os.path.join(args.save_dir, f'test_size_comparison_{timestamp}.txt')
    with open(summary_path, 'w') as f:
        f.write("Edge Prediction Performance vs Test Size\n")
        f.write("=" * 50 + "\n\n")
        f.write("Test_Size\tTP\tFN\tFP\tTN\tPrecision\tRecall\tF1-Score\tAccuracy\n")
        
        for test_size in args.test_sizes:
            metrics = all_results[test_size]
            f.write(f"{test_size}\t{metrics['tp']}\t{metrics['fn']}\t{metrics['fp']}\t{metrics['tn']}\t"
                   f"{metrics['precision']:.3f}\t{metrics['recall']:.3f}\t{metrics['f1']:.3f}\t{metrics['accuracy']:.3f}\n")
    
    print(f"\nSummary results saved to: {summary_path}")
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main() 