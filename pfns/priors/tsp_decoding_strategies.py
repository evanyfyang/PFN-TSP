#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import random
import math
import time
from typing import List, Tuple


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

def greedy_edge_decode(adj_list, num_nodes):
    """
    Edge-based greedy decoding strategy using union-find and degree counting.
    """
    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.components = n
        
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
        
        def union(self, x, y):
            px, py = self.find(x), self.find(y)
            if px == py:
                return False
            
            if self.rank[px] < self.rank[py]:
                px, py = py, px
            self.parent[py] = px
            if self.rank[px] == self.rank[py]:
                self.rank[px] += 1
            self.components -= 1
            return True
        
        def connected(self, x, y):
            return self.find(x) == self.find(y)
    
    edges = []
    for node in range(num_nodes):
        for neighbor, prob in adj_list[node]:
            if node < neighbor:
                edges.append((prob, node, neighbor))
    
    edges.sort(reverse=True)
    
    uf = UnionFind(num_nodes)
    degree = [0] * num_nodes
    selected_edges = []
    
    for prob, u, v in edges:
        if degree[u] >= 2 or degree[v] >= 2:
            continue
        
        if uf.connected(u, v) and len(selected_edges) < num_nodes - 1:
            continue
        
        selected_edges.append((u, v))
        degree[u] += 1
        degree[v] += 1
        uf.union(u, v)
        
        if len(selected_edges) == num_nodes:
            break
    
    if len(selected_edges) < num_nodes:
        endpoints = [i for i in range(num_nodes) if degree[i] == 1]
        
        if len(endpoints) == 2:
            u, v = endpoints
            selected_edges.append((u, v))
        else:
            return greedy_decode(adj_list, num_nodes)
    
    def edges_to_tour(edges, num_nodes):
        graph = {i: [] for i in range(num_nodes)}
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        
        tour = [0]
        visited = {0}
        current = 0
        
        while len(tour) < num_nodes:
            next_node = None
            for neighbor in graph[current]:
                if neighbor not in visited:
                    next_node = neighbor
                    break
            
            if next_node is None:
                break
            
            tour.append(next_node)
            visited.add(next_node)
            current = next_node
        
        return tour
    
    tour = edges_to_tour(selected_edges, num_nodes)
    
    if len(tour) < num_nodes:
        return greedy_decode(adj_list, num_nodes)
    
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




def mcts_decode(
    node_positions: np.ndarray,
    distance_matrix: np.ndarray,
    candidate_mask: np.ndarray,
    prob_matrix: np.ndarray,
    neighbor_lists: List[List[Tuple[int, float]]],
):
    N = int(node_positions.shape[0])
    Virtual_City_Num = N
    City_Num = N
    Null = -1
    Inf_Cost = 10**18

    # ---- missing items defined here ----
    Start_City = 0            # canonical name (used in helpers)
    Start_city = Start_City   # alias to satisfy references that use this casing

    class Node:
        # __slots__ = ("Next_City", "Pre_City", "Salesman")
        __slots__ = ("Next_City", "Pre_City")
        def __init__(self):
            self.Next_City = Null
            self.Pre_City = Null
            # self.Salesman = 0

    All_Node = [Node() for _ in range(N)]
    Best_All_Node = [Node() for _ in range(N)]   # canonical
    Best_All_nodes = Best_All_Node               # alias to satisfy references using plural/snake

    # Solution array used by helpers 
    Solution = np.full(Virtual_City_Num, Null, dtype=np.int32)
    # -------------------------------------

    Alpha = 0.85
    Beta = 1.0
    Param_T = 0.02
    Param_H = 3
    Max_Depth = min(16, N)

    Edge_Heatmap = np.clip(prob_matrix.astype(np.float64), 0.0, 1.0)
    Weight = np.zeros((N, N), dtype=np.float64)
    Chosen_Times = np.zeros((N, N), dtype=np.int32)

    Candidate = [[] for _ in range(N)]
    Candidate_Num = np.zeros(N, dtype=np.int32)
    for u in range(N):
        cand = np.flatnonzero(candidate_mask[u]).tolist()
        if neighbor_lists and len(neighbor_lists[u]) > 0:
            order = {v: i for i, (v, _) in enumerate(neighbor_lists[u])}
            cand.sort(key=lambda v: order.get(v, len(cand)))
        Candidate[u] = cand
        Candidate_Num[u] = len(cand)

    City_Sequence = np.full(2 * (Max_Depth + 2), Null, dtype=np.int32)
    Temp_City_Sequence = np.full_like(City_Sequence, Null)
    Gain = np.zeros(Max_Depth + 2, dtype=np.float64)
    Real_Gain = np.zeros_like(Gain)
    Pair_City_Num = 0
    Temp_Pair_Num = 0

    Promising_City = np.full(N, Null, dtype=np.int32)
    Promising_City_Num = 0
    Probabilistic = np.zeros(N, dtype=np.int32)

    Avg_Weight = 0.0
    Total_Simulation_Times = 0
    Current_Instance_Begin_Time = time.time()
    Current_Instance_Best_Distance = float(Inf_Cost)

    rng = np.random.default_rng()

    # ---------- helpers (matching DIMES C names) ----------
    Distance = np.full((N, N), Inf_Cost, dtype=np.float64)

    def Get_Random_Int(Divide_Num):
        return int(rng.integers(0, max(1, Divide_Num)))

    def Calculate_Int_Distance(First_City, Second_City):
        dx = float(node_positions[First_City, 0] - node_positions[Second_City, 0])
        dy = float(node_positions[First_City, 1] - node_positions[Second_City, 1])
        return int(0.5 + math.sqrt(dx * dx + dy * dy))

    def Calculate_Double_Distance(First_City, Second_City):
        dx = float(node_positions[First_City, 0] - node_positions[Second_City, 0])
        dy = float(node_positions[First_City, 1] - node_positions[Second_City, 1])
        return math.sqrt(dx * dx + dy * dy)

    def Calculate_All_Pair_Distance():
        for i in range(Virtual_City_Num):
            for j in range(Virtual_City_Num):
                if i == j:
                    Distance[i, j] = Inf_Cost
                else:
                    if candidate_mask[i, j]:
                        d = float(distance_matrix[i, j])
                        if d <= 0.0:
                            d = Calculate_Int_Distance(i, j)
                        Distance[i, j] = d
                    else:
                        Distance[i, j] = Inf_Cost

    def Get_Distance(First_City, Second_City):
        return float(Distance[First_City, Second_City])

    def Convert_Solution_To_All_Node():
        # Cur_Salesman = 0
        for i in range(Virtual_City_Num):
            Temp_Cur_City = int(Solution[i])
            Temp_Pre_City = int(Solution[(i - 1 + Virtual_City_Num) % Virtual_City_Num])
            Temp_Next_City = int(Solution[(i + 1) % Virtual_City_Num])

            # if Temp_Cur_City >= City_Num:
            #     Cur_Salesman += 1

            All_Node[Temp_Cur_City].Pre_City = Temp_Pre_City
            All_Node[Temp_Cur_City].Next_City = Temp_Next_City
            # All_Node[Temp_Cur_City].Salesman = Cur_Salesman

    def Convert_All_Node_To_Solution():
        for i in range(Virtual_City_Num):
            Solution[i] = Null

        Cur_Index = 0
        Solution[Cur_Index] = Start_City

        Cur_City = Start_City
        while True:
            Cur_Index += 1
            Cur_City = All_Node[Cur_City].Next_City
            if Cur_City == Null or Cur_Index >= Virtual_City_Num:
                return False
            Solution[Cur_Index] = Cur_City
            if All_Node[Cur_City].Next_City == Start_City:
                break
        return True

    def Check_Solution_Feasible():
        Cur_City = Start_City
        Visited_City_Num = 0
        while True:
            Cur_City = All_Node[Cur_City].Next_City
            if Cur_City == Null:
                return False
            Visited_City_Num += 1
            if Visited_City_Num > Virtual_City_Num:
                return False
            if Cur_City == Start_City and Visited_City_Num == Virtual_City_Num:
                return True

    def Get_Solution_Total_Distance():
        Solution_Total_Distance = 0.0
        for i in range(Virtual_City_Num):
            Temp_Next_City = All_Node[i].Next_City
            if Temp_Next_City != Null:
                Solution_Total_Distance += Get_Distance(i, Temp_Next_City)
            else:
                return Inf_Cost
        return Solution_Total_Distance

    def Reverse_Sub_Path(First_City, Second_City):
        Cur_City = First_City
        Temp_Next_City = All_Node[Cur_City].Next_City
        while True:
            Temp_City = All_Node[Cur_City].Pre_City
            All_Node[Cur_City].Pre_City = All_Node[Cur_City].Next_City
            All_Node[Cur_City].Next_City = Temp_City
            if Cur_City == Second_City:
                break
            Cur_City = Temp_Next_City
            Temp_Next_City = All_Node[Cur_City].Next_City

    def Store_Best_Solution():
        for i in range(Virtual_City_Num):
            # Best_All_Node[i].Salesman = All_Node[i].Salesman
            Best_All_Node[i].Next_City = All_Node[i].Next_City
            Best_All_Node[i].Pre_City = All_Node[i].Pre_City

    def Restore_Best_Solution():
        for i in range(Virtual_City_Num):
            # All_Node[i].Salesman = Best_All_Node[i].Salesman
            All_Node[i].Next_City = Best_All_Node[i].Next_City
            All_Node[i].Pre_City = Best_All_Node[i].Pre_City
    # -----------------------------------------------------

    def MCTS_Init():
        nonlocal Total_Simulation_Times
        for i in range(Virtual_City_Num):
            for j in range(Virtual_City_Num):
                Weight[i, j] = Edge_Heatmap[i, j] * 100.0
                Chosen_Times[i, j] = 0
        Total_Simulation_Times = 0

    def Get_Avg_Weight(Cur_City):
        total = 0.0
        for i in range(Virtual_City_Num):
            if i == Cur_City:
                continue
            total += Weight[Cur_City, i]
        return total / float(Virtual_City_Num - 1) if Virtual_City_Num > 1 else 0.0

    def Get_Potential(First_City, Second_City):
        denom = float(Chosen_Times[First_City, Second_City] + 1)
        expl = Alpha * math.sqrt(max(0.0, math.log(Total_Simulation_Times + 1.0) / (1.0 * denom)))
        base = Weight[First_City, Second_City] / (Avg_Weight if Avg_Weight > 0 else 1.0)
        return base + expl

    Promising_City = np.full(N, Null, dtype=np.int32)
    Promising_City_Num = 0
    Probabilistic = np.zeros(N, dtype=np.int32)
    Avg_Weight = 0.0

    def Identify_Promising_City(Cur_City, Begin_City):
        nonlocal Promising_City_Num
        Promising_City_Num = 0
        for i in range(Candidate_Num[Cur_City]):
            Temp_City = Candidate[Cur_City][i]
            if Temp_City == Begin_City:
                continue
            if Temp_City == All_Node[Cur_City].Next_City:
                continue
            if Get_Potential(Cur_City, Temp_City) < 1.0:
                continue
            Promising_City[Promising_City_Num] = Temp_City
            Promising_City_Num += 1

    def Get_Probabilistic(Cur_City):
        if Promising_City_Num == 0:
            return False
        total_p = 0.0
        for i in range(Promising_City_Num):
            total_p += Get_Potential(Cur_City, int(Promising_City[i]))
        if total_p <= 0.0:
            return False
        s = 0
        for i in range(Promising_City_Num):
            pot = Get_Potential(Cur_City, int(Promising_City[i]))
            if i < Promising_City_Num - 1:
                s += int(1000.0 * pot / total_p)
                Probabilistic[i] = s
            else:
                Probabilistic[i] = 1000
        return True

    def Probabilistic_Get_City_To_Connect():
        Random_Num = Get_Random_Int(1000)
        for i in range(Promising_City_Num):
            if Random_Num < int(Probabilistic[i]):
                return int(Promising_City[i])
        return Null

    def Choose_City_To_Connect(Cur_City, Begin_City):
        nonlocal Avg_Weight
        Avg_Weight = Get_Avg_Weight(Cur_City)
        Identify_Promising_City(Cur_City, Begin_City)
        Get_Probabilistic(Cur_City)
        return Probabilistic_Get_City_To_Connect()

    Pair_City_Num = 0
    Temp_Pair_Num = 0

    def Get_Simulated_Action_Delta(Begin_City):
        nonlocal Pair_City_Num
        if Convert_All_Node_To_Solution() is False:
            return -Inf_Cost

        Next_City = All_Node[Begin_City].Next_City
        All_Node[Begin_City].Next_City = Null
        All_Node[Next_City].Pre_City = Null

        City_Sequence[0] = Begin_City
        City_Sequence[1] = Next_City

        Gain[0] = Get_Distance(Begin_City, Next_City)
        Real_Gain[0] = Gain[0] - Get_Distance(Next_City, Begin_City)
        Pair_City_Num = 1

        If_Changed = False
        Cur_City = Next_City

        while True:
            Next_City_To_Connect = Choose_City_To_Connect(Cur_City, Begin_City)
            if Next_City_To_Connect == Null:
                break

            Chosen_Times[Cur_City, Next_City_To_Connect] += 1
            Chosen_Times[Next_City_To_Connect, Cur_City] += 1

            Next_City_To_Disconnect = All_Node[Next_City_To_Connect].Pre_City

            City_Sequence[2 * Pair_City_Num] = Next_City_To_Connect
            City_Sequence[2 * Pair_City_Num + 1] = Next_City_To_Disconnect

            Gain[Pair_City_Num] = (
                Gain[Pair_City_Num - 1]
                - Get_Distance(Cur_City, Next_City_To_Connect)
                + Get_Distance(Next_City_To_Connect, Next_City_To_Disconnect)
            )
            Real_Gain[Pair_City_Num] = (
                Gain[Pair_City_Num] - Get_Distance(Next_City_To_Disconnect, Begin_City)
            )
            Pair_City_Num += 1

            Reverse_Sub_Path(Cur_City, Next_City_To_Disconnect)
            All_Node[Cur_City].Next_City = Next_City_To_Connect
            All_Node[Next_City_To_Connect].Pre_City = Cur_City
            All_Node[Next_City_To_Disconnect].Pre_City = Null
            If_Changed = True

            Cur_City = Next_City_To_Disconnect

            if Real_Gain[Pair_City_Num - 1] > 0 or Pair_City_Num > Max_Depth:
                break

        if If_Changed:
            Convert_Solution_To_All_Node()
        else:
            All_Node[Begin_City].Next_City = Next_City
            All_Node[Next_City].Pre_City = Begin_City

        Max_Real_Gain = -Inf_Cost
        Best_Index = 1
        for i in range(1, Pair_City_Num):
            if Real_Gain[i] > Max_Real_Gain:
                Max_Real_Gain = Real_Gain[i]
                Best_Index = i

        Pair_City_Num = Best_Index + 1
        return Max_Real_Gain

    def Back_Propagation(Before_Simulation_Distance, Action_Delta):
        if Action_Delta <= 0:
            return
        Increase_Rate = Beta * (math.exp(float(Action_Delta) / float(Before_Simulation_Distance)) - 1.0)
        for i in range(Pair_City_Num):
            Second_City = int(City_Sequence[2 * i + 1])
            Third_City = int(City_Sequence[2 * i + 2]) if i < Pair_City_Num - 1 else int(City_Sequence[0])
            Weight[Second_City, Third_City] += Increase_Rate
            Weight[Third_City, Second_City] += Increase_Rate

    def Simulation(Max_Simulation_Times):
        nonlocal Temp_Pair_Num, Total_Simulation_Times, Pair_City_Num
        Best_Action_Delta = -Inf_Cost
        for _ in range(Max_Simulation_Times):
            Begin_City = Get_Random_Int(Virtual_City_Num)
            Action_Delta = Get_Simulated_Action_Delta(Begin_City)
            Total_Simulation_Times += 1

            if Action_Delta > Best_Action_Delta:
                Best_Action_Delta = Action_Delta
                Temp_Pair_Num = Pair_City_Num
                Temp_City_Sequence[: 2 * Pair_City_Num] = City_Sequence[: 2 * Pair_City_Num]

            if Best_Action_Delta > 0:
                break

        Pair_City_Num = Temp_Pair_Num
        City_Sequence[: 2 * Pair_City_Num] = Temp_City_Sequence[: 2 * Pair_City_Num]
        return Best_Action_Delta

    def Execute_Best_Action():
        Begin_City = int(City_Sequence[0])
        Cur_City = int(City_Sequence[1])
        All_Node[Begin_City].Next_City = Null
        All_Node[Cur_City].Pre_City = Null
        for i in range(1, Pair_City_Num):
            Next_City_To_Connect = int(City_Sequence[2 * i])
            Next_City_To_Disconnect = int(City_Sequence[2 * i + 1])

            Reverse_Sub_Path(Cur_City, Next_City_To_Disconnect)

            All_Node[Cur_City].Next_City = Next_City_To_Connect
            All_Node[Next_City_To_Connect].Pre_City = Cur_City
            All_Node[Next_City_To_Disconnect].Pre_City = Null

            Cur_City = Next_City_To_Disconnect

        All_Node[Begin_City].Next_City = Cur_City
        All_Node[Cur_City].Pre_City = Begin_City

        return Check_Solution_Feasible()

    def MCTS():
        nonlocal Current_Instance_Best_Distance
        while (time.time() - Current_Instance_Begin_Time) < (Param_T * Virtual_City_Num):
            Before_Simulation_Distance = Get_Solution_Total_Distance()
            Best_Delta = Simulation(Param_H * Virtual_City_Num)
            Back_Propagation(Before_Simulation_Distance, Best_Delta)

            if Best_Delta > 0:
                if Execute_Best_Action():
                    Cur_Solution_Total_Distance = Get_Solution_Total_Distance()
                    if Cur_Solution_Total_Distance < Current_Instance_Best_Distance:
                        Current_Instance_Best_Distance = Cur_Solution_Total_Distance
                        Store_Best_Solution()
                else:
                    break
            else:
                break

    # ---- run: distances, initial tour, MCTS ----
    Calculate_All_Pair_Distance()

    used = np.zeros(N, dtype=bool)
    tour0 = [Start_City]
    used[Start_City] = True
    for _ in range(1, N):
        u = tour0[-1]
        next_candidates = [v for v in Candidate[u] if not used[v]]
        if len(next_candidates) == 0:
            pool = [v for v in range(N) if not used[v] and v != u]
            if not pool:
                break
            v = min(pool, key=lambda x: Get_Distance(u, x))
            if Get_Distance(u, v) >= Inf_Cost:
                v = min(pool, key=lambda x: Calculate_Double_Distance(u, x))
        else:
            v = max(next_candidates, key=lambda x: (prob_matrix[u, x], -Get_Distance(u, x)))
        tour0.append(v)
        used[v] = True

    for i in range(N):
        Solution[i] = tour0[i]
    Convert_Solution_To_All_Node()

    if not Check_Solution_Feasible():
        for u in range(N):
            All_Node[u].Next_City = (u + 1) % N
            All_Node[(u + 1) % N].Pre_City = u

    MCTS_Init()
    Current_Instance_Best_Distance = Get_Solution_Total_Distance()
    Store_Best_Solution()
    MCTS()
    Restore_Best_Solution()

    tour = []
    cur = Start_City
    seen = set()
    for _ in range(N):
        tour.append(cur)
        seen.add(cur)
        cur = All_Node[cur].Next_City
        if cur in seen or cur == Null:
            break
    if len(tour) != N:
        tour = list(range(N))
    return tour
