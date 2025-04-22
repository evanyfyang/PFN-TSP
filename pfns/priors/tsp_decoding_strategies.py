#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch

def greedy_decode(adj_list, num_nodes):
    """
    Greedy decoding strategy starting from node 0.
    """
    current_node = 0
    tour = [current_node]
    visited = set([current_node])
    
    while len(tour) < num_nodes:
        neighbors = adj_list[current_node]
        
        valid_neighbors = [(node, prob) for node, prob in neighbors if node not in visited]
        
        # If there are no valid neighbors, choose the first unvisited node
        if not valid_neighbors:
            unvisited = list(set(range(num_nodes)) - visited)
            if unvisited:
                next_node = unvisited[0]
            else:
                break
        else:
            next_node = max(valid_neighbors, key=lambda x: x[1])[0]
        
        tour.append(next_node)
        visited.add(next_node)
        current_node = next_node
    
    return tour

def greedy_all_decode(adj_list, num_nodes):
    """
    Try greedy decoding starting from each node and select the best path.
    """
    best_tour = None
    best_tour_prob = -float('inf')
    
    for start_node in range(num_nodes):
        current_node = start_node
        tour = [current_node]
        visited = set([current_node])
        tour_prob = 0.0
        
        while len(tour) < num_nodes:
            neighbors = adj_list[current_node]
            valid_neighbors = [(node, prob) for node, prob in neighbors if node not in visited]
            
            if not valid_neighbors:
                unvisited = list(set(range(num_nodes)) - visited)
                if unvisited:
                    next_node = unvisited[0]
                    tour_prob += 0.0  # Penalty probability
                else:
                    break
            else:
                next_node, prob = max(valid_neighbors, key=lambda x: x[1])
                tour_prob += prob
            
            tour.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        # Add the edge back to the starting point
        if len(tour) == num_nodes:
            last_node = tour[-1]
            first_node = tour[0]
            for node, prob in adj_list[last_node]:
                if node == first_node:
                    tour_prob += prob
                    break
            
            if tour_prob > best_tour_prob:
                best_tour_prob = tour_prob
                best_tour = tour
    
    # If all attempts fail, use the default greedy starting from node 0
    if best_tour is None:
        best_tour = greedy_decode(adj_list, num_nodes)
    
    return best_tour

def beam_search_decode(adj_list, num_nodes, beam_width=5):
    """
    Beam search decoding strategy starting from node 0.
    Optimizes for shortest path length instead of highest probability.
    """
    initial_path = [0]
    initial_visited = set([0])
    
    # Use path length instead of probability as scoring criteria (initial length = 0)
    beam = [(initial_path, initial_visited, 0.0)]  
    
    while beam and len(beam[0][0]) < num_nodes:
        new_candidates = []
        
        for path, visited, path_length in beam:
            current_node = path[-1]
            neighbors = adj_list[current_node]
            valid_neighbors = [(node, prob) for node, prob in neighbors if node not in visited]
            
            if not valid_neighbors:
                unvisited = list(set(range(num_nodes)) - visited)
                for next_node in unvisited[:beam_width]:
                    new_path = path + [next_node]
                    new_visited = visited.copy()
                    new_visited.add(next_node)
                    # Use a default distance as penalty
                    new_candidates.append((new_path, new_visited, path_length + 10.0))
            else:
                for next_node, edge_prob in valid_neighbors:
                    new_path = path + [next_node]
                    new_visited = visited.copy()
                    new_visited.add(next_node)
                    # Use 1/prob as distance metric (higher probability means shorter distance)
                    distance = 1.0 / edge_prob if edge_prob > 0 else 10.0
                    new_candidates.append((new_path, new_visited, path_length + distance))

        if not new_candidates:
            break
        # Note: When sorting by path length, reverse=False means selecting the shortest path
        beam = sorted(new_candidates, key=lambda x: x[2], reverse=False)[:beam_width]
    
    if beam:
        tour = beam[0][0]
    else:
        tour = list(range(num_nodes))
    
    return tour

def beam_search_all_decode(adj_list, num_nodes, beam_width=5):
    """
    Try beam search decoding starting from each node and select the shortest path.
    Optimizes for shortest path length instead of highest probability.
    """
    best_tour = None
    best_tour_length = float('inf')  # Initialize to infinity, looking for minimum value
    
    for start_node in range(num_nodes):
        initial_path = [start_node]
        initial_visited = set([start_node])
        
        # Use path length instead of probability
        beam = [(initial_path, initial_visited, 0.0)]
        
        while beam and len(beam[0][0]) < num_nodes:
            new_candidates = []
            
            for path, visited, path_length in beam:
                current_node = path[-1]
                neighbors = adj_list[current_node]
                valid_neighbors = [(node, prob) for node, prob in neighbors if node not in visited]
                
                if not valid_neighbors:
                    unvisited = list(set(range(num_nodes)) - visited)
                    for next_node in unvisited[:beam_width]:
                        new_path = path + [next_node]
                        new_visited = visited.copy()
                        new_visited.add(next_node)
                        # Use a default distance as penalty
                        new_candidates.append((new_path, new_visited, path_length + 10.0))
                else:
                    for next_node, edge_prob in valid_neighbors:
                        new_path = path + [next_node]
                        new_visited = visited.copy()
                        new_visited.add(next_node)
                        # Use 1/prob as distance metric
                        distance = 1.0 / edge_prob if edge_prob > 0 else 10.0
                        new_candidates.append((new_path, new_visited, path_length + distance))
            if not new_candidates:
                break
            # Sort by path length (ascending)
            beam = sorted(new_candidates, key=lambda x: x[2], reverse=False)[:beam_width]
        
        if beam:
            tour = beam[0][0]
            tour_length = beam[0][2]
            
            # Select the shortest path
            if tour_length < best_tour_length:
                best_tour_length = tour_length
                best_tour = tour
    
    # If all attempts fail, use default beam search
    if best_tour is None:
        best_tour = beam_search_decode(adj_list, num_nodes, beam_width)
    
    return best_tour

def mcmc_decode(adj_list, node_map, edge_index, edge_values, num_nodes, num_iterations=1000, temperature=1.0):
    """
    MCMC decoding strategy using 2-opt local search.
    Optimizes for shortest path length instead of highest probability.
    """
    # First use greedy method to get an initial solution
    current_node = 0
    initial_tour = [current_node]
    visited = set([current_node])
    
    while len(initial_tour) < num_nodes:
        neighbors = adj_list[current_node]
        valid_neighbors = [(node, prob) for node, prob in neighbors if node not in visited]
        
        if not valid_neighbors:
            unvisited = list(set(range(num_nodes)) - visited)
            if unvisited:
                next_node = unvisited[0]
            else:
                break
        else:
            next_node = max(valid_neighbors, key=lambda x: x[1])[0]
        
        initial_tour.append(next_node)
        visited.add(next_node)
        current_node = next_node
    
    # Convert probabilities to distances (1/probability)
    distance_lookup = {}
    for i, (u, v) in enumerate(edge_index):
        u_real = node_map[u]
        v_real = node_map[v]
        # Convert probability to distance, higher probability means shorter distance
        distance = 1.0 / edge_values[i] if edge_values[i] > 0 else 10.0
        distance_lookup[(u_real, v_real)] = distance
        distance_lookup[(v_real, u_real)] = distance  # Undirected graph
    
    def calculate_tour_length(tour):
        total_length = 0
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i+1) % len(tour)]
            edge = (u, v)
            # If edge doesn't exist, use a larger default distance
            distance = distance_lookup.get(edge, 10.0)
            total_length += distance
        return total_length
    
    current_tour = initial_tour
    current_length = calculate_tour_length(current_tour)
    best_tour = current_tour.copy()
    best_length = current_length
    
    for _ in range(num_iterations):
        i, j = sorted(np.random.choice(range(num_nodes), 2, replace=False))
        if i == 0 and j == num_nodes - 1:
            continue  
        
        # 2-opt swap
        new_tour = current_tour.copy()
        new_tour[i:j+1] = reversed(current_tour[i:j+1])
        
        new_length = calculate_tour_length(new_tour)
        
        # Metropolis-Hastings (note the negative sign, because we want to minimize length)
        acceptance_ratio = np.exp((current_length - new_length) / temperature)
        if np.random.random() < acceptance_ratio:
            current_tour = new_tour
            current_length = new_length
            
            # Update best path
            if current_length < best_length:
                best_tour = current_tour.copy()
                best_length = current_length
    
    return best_tour 