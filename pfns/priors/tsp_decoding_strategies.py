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
    """
    initial_path = [0]
    initial_visited = set([0])
    
    beam = [(initial_path, initial_visited, 1.0)]
    
    while beam and len(beam[0][0]) < num_nodes:
        new_candidates = []
        
        for path, visited, path_prob in beam:
            current_node = path[-1]
            neighbors = adj_list[current_node]
            valid_neighbors = [(node, prob) for node, prob in neighbors if node not in visited]
            
            if not valid_neighbors:
                unvisited = list(set(range(num_nodes)) - visited)
                for next_node in unvisited[:beam_width]:
                    new_path = path + [next_node]
                    new_visited = visited.copy()
                    new_visited.add(next_node)
                    new_candidates.append((new_path, new_visited, path_prob * 0.5))
            else:
                for next_node, edge_prob in valid_neighbors:
                    new_path = path + [next_node]
                    new_visited = visited.copy()
                    new_visited.add(next_node)
                    new_candidates.append((new_path, new_visited, path_prob * edge_prob))
        if not new_candidates:
            break
        beam = sorted(new_candidates, key=lambda x: x[2], reverse=True)[:beam_width]
    
    if beam:
        tour = beam[0][0]
    else:
        tour = list(range(num_nodes))
    
    return tour

def beam_search_all_decode(adj_list, num_nodes, beam_width=5):
    """
    Try beam search decoding starting from each node and select the best path.
    """
    best_tour = None
    best_tour_prob = -float('inf')
    
    for start_node in range(num_nodes):
        initial_path = [start_node]
        initial_visited = set([start_node])
        
        beam = [(initial_path, initial_visited, 1.0)]
        
        while beam and len(beam[0][0]) < num_nodes:
            new_candidates = []
            
            for path, visited, path_prob in beam:
                current_node = path[-1]
                neighbors = adj_list[current_node]
                valid_neighbors = [(node, prob) for node, prob in neighbors if node not in visited]
                
                if not valid_neighbors:
                    unvisited = list(set(range(num_nodes)) - visited)
                    for next_node in unvisited[:beam_width]:
                        new_path = path + [next_node]
                        new_visited = visited.copy()
                        new_visited.add(next_node)
                        new_candidates.append((new_path, new_visited, path_prob * 0.5))
                else:
                    for next_node, edge_prob in valid_neighbors:
                        new_path = path + [next_node]
                        new_visited = visited.copy()
                        new_visited.add(next_node)
                        new_candidates.append((new_path, new_visited, path_prob * edge_prob))
            if not new_candidates:
                break
            beam = sorted(new_candidates, key=lambda x: x[2], reverse=True)[:beam_width]
        
        if beam:
            tour = beam[0][0]
            tour_prob = beam[0][2]
            
            if tour_prob > best_tour_prob:
                best_tour_prob = tour_prob
                best_tour = tour
    
    # If all attempts fail, use the default beam search starting from node 0
    if best_tour is None:
        best_tour = beam_search_decode(adj_list, num_nodes, beam_width)
    
    return best_tour

def mcmc_decode(adj_list, node_map, edge_index, edge_values, num_nodes, num_iterations=1000, temperature=1.0):
    """
    MCMC decoding strategy using 2-opt local search.
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
    
    prob_lookup = {}
    for i, (u, v) in enumerate(edge_index):
        u_real = node_map[u]
        v_real = node_map[v]
        prob_lookup[(u_real, v_real)] = edge_values[i]
        prob_lookup[(v_real, u_real)] = edge_values[i]  # Undirected graph
    
    def calculate_tour_probability(tour):
        total_log_prob = 0
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i+1) % len(tour)]
            edge = (u, v)
            prob = prob_lookup.get(edge, 0.01)
            total_log_prob += np.log(prob)
        return total_log_prob
    
    current_tour = initial_tour
    current_prob = calculate_tour_probability(current_tour)
    best_tour = current_tour.copy()
    best_prob = current_prob
    
    for _ in range(num_iterations):
        i, j = sorted(np.random.choice(range(num_nodes), 2, replace=False))
        if i == 0 and j == num_nodes - 1:
            continue  
        
        # 2-opt swap
        new_tour = current_tour.copy()
        new_tour[i:j+1] = reversed(current_tour[i:j+1])
        
        new_prob = calculate_tour_probability(new_tour)
        
        # Metropolis-Hastings
        acceptance_ratio = np.exp((new_prob - current_prob) / temperature)
        if np.random.random() < acceptance_ratio:
            current_tour = new_tour
            current_prob = new_prob
            
            if current_prob > best_prob:
                best_tour = current_tour.copy()
                best_prob = current_prob
    
    return best_tour 