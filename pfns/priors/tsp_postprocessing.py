from __future__ import annotations
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import time
from scipy.spatial import Delaunay
import math
from typing import List


#use or-tools to solve tsp, but initial solution is given
def solve_tsp_static_with_or_tools_and_initial_solutions(initial_solution: list, coords: np.ndarray, time_limit = 1):
    num_nodes = len(coords)
    distance_matrix = np.zeros((num_nodes+1, num_nodes+1))#add more stop as fake depot for initial solution assignment
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                distance_matrix[i][j] = 0
            else:
                distance_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    
    manager = pywrapcp.RoutingIndexManager(num_nodes+1, 1, num_nodes)#n nodes to be visited + 1 depot. Depot is num_nodes, nodes to be visited is from 0 to num_nodes -1
    routing = pywrapcp.RoutingModel(manager)
    

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)
        
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()#DO not use OR-tools to generate initial solution
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(time_limit)
    routing.CloseModelWithParameters(search_parameters)
    
    

    # Convert route to indices used by routing model
    route_indices = [manager.NodeToIndex(node) for node in initial_solution]
    assignment = routing.ReadAssignmentFromRoutes([route_indices],True)
    if assignment is None:
        print("Assignment Fail")

    
    solution = routing.SolveFromAssignmentWithParameters(assignment,search_parameters)
    
    
    if solution:
        index = routing.Start(0)
        tour = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            tour.append(node)
            index = solution.Value(routing.NextVar(index))
        return tour[1:]#need to eliminate fake depot
    else:
        return list(range(num_nodes))
    

#use 2-=opt to further improve the solution
def solve_tsp_static_with_2opt_and_initial_solutions(initial_solution: List[int],
                                                     coords: np.ndarray,
                                                     time_limit: float = 1.0) -> List[int]:
    """
    Optimize a TSP tour using 2-opt local search starting from 'initial_solution'.
    - initial_solution: list of node indices forming a Hamiltonian cycle order.
    - coords: np.ndarray of shape [N, 2] with XY coordinates.
    - time_limit: maximum wall-clock time (seconds) for local search.
    Returns: optimized tour (list of node indices in visiting order; implicitly cycles back to start).
    """

    # -------------------------------
    # Global-like state (C-style)
    # -------------------------------
    Null = -1
    Inf_Cost = 10**18
    Virtual_City_Num = int(coords.shape[0])
    City_Num = Virtual_City_Num
    Start_City = int(initial_solution[0]) if len(initial_solution) > 0 else 0

    # Node structure from C (doubly-linked cycle)
    class Node:
        # __slots__ = ("Next_City", "Pre_City", "Salesman")
        __slots__ = ("Next_City", "Pre_City")
        def __init__(self):
            self.Next_City = Null
            self.Pre_City = Null
            # self.Salesman = 0

    All_Node = [Node() for _ in range(Virtual_City_Num)]
    Best_All_Node = [Node() for _ in range(Virtual_City_Num)]

    # Distance matrix (double / float64)
    Distance = np.full((Virtual_City_Num, Virtual_City_Num), Inf_Cost, dtype=np.float64)

    # Candidate sets (like DIMES: Candidate[i][*], Candidate_Num[i])
    Candidate: List[List[int]] = [[] for _ in range(Virtual_City_Num)]
    Candidate_Num = np.zeros(Virtual_City_Num, dtype=np.int32)

    # These exist in the original DIMES 2-opt code when used with MCTS; keep them for structural parity
    Weight = np.zeros((Virtual_City_Num, Virtual_City_Num), dtype=np.float64)
    Chosen_Times = np.zeros((Virtual_City_Num, Virtual_City_Num), dtype=np.int32)
    Total_Simulation_Times = 0
    Beta = 1.0  # used in Apply_2Opt_Move's backprop-like weight update

    Current_Instance_Best_Distance = float(Inf_Cost)
    start_time = time.time()

    # --------------------------------
    # Helper functions (C names)
    # --------------------------------
    def Calculate_Double_Distance(First_City: int, Second_City: int) -> float:
        dx = float(coords[First_City, 0] - coords[Second_City, 0])
        dy = float(coords[First_City, 1] - coords[Second_City, 1])
        return math.sqrt(dx * dx + dy * dy)

    def Calculate_All_Pair_Distance():
        for i in range(Virtual_City_Num):
            for j in range(Virtual_City_Num):
                if i == j:
                    Distance[i, j] = Inf_Cost
                else:
                    Distance[i, j] = Calculate_Double_Distance(i, j)

    def Get_Distance(First_City: int, Second_City: int) -> float:
        return float(Distance[First_City, Second_City])

    def Convert_Solution_To_All_Node(solution_list: List[int]):
        # Build doubly linked cycle from a linear tour order
        for idx, cur in enumerate(solution_list):
            pre = solution_list[(idx - 1) % Virtual_City_Num]
            nxt = solution_list[(idx + 1) % Virtual_City_Num]
            All_Node[cur].Pre_City = pre
            All_Node[cur].Next_City = nxt
            # All_Node[cur].Salesman = 0  # single TSP tour

    def Check_Solution_Feasible() -> bool:
        cur = Start_City
        visited = 0
        seen = set()
        while True:
            cur = All_Node[cur].Next_City
            if cur == Null:
                return False
            if cur in seen:
                # Should only revisit the start after exactly N steps
                return cur == Start_City and visited == Virtual_City_Num - 1
            seen.add(cur)
            visited += 1
            if visited > Virtual_City_Num:
                return False
            if cur == Start_City and visited == Virtual_City_Num:
                return True

    def Get_Solution_Total_Distance() -> float:
        total = 0.0
        for i in range(Virtual_City_Num):
            nxt = All_Node[i].Next_City
            if nxt == Null:
                return Inf_Cost
            total += Get_Distance(i, nxt)
        return total

    def Check_If_Two_City_Same_Or_Adjacent(First_City: int, Second_City: int) -> bool:
        if First_City == Second_City:
            return True
        if All_Node[First_City].Next_City == Second_City:
            return True
        if All_Node[Second_City].Next_City == First_City:
            return True
        return False

    def Reverse_Sub_Path(First_City: int, Second_City: int):
        # Reverse by swapping Pre/Next along the segment First_City .. Second_City (inclusive)
        cur = First_City
        temp_next = All_Node[cur].Next_City
        while True:
            temp = All_Node[cur].Pre_City
            All_Node[cur].Pre_City = All_Node[cur].Next_City
            All_Node[cur].Next_City = temp
            if cur == Second_City:
                break
            cur = temp_next
            temp_next = All_Node[cur].Next_City

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

    # Candidate set: k-nearest neighbors per city (pruned neighborhood like in DIMES)
    def Identify_Candidate_Set(k: int = None):
        if k is None:
            # Reasonable default pruning
            k = min(32, Virtual_City_Num - 1)
        for u in range(Virtual_City_Num):
            # sort all others by distance
            order = np.argsort(Distance[u]).tolist()
            # remove self (where distance is Inf_Cost)
            order = [v for v in order if v != u]
            # take k nearest
            cand = order[:k]
            Candidate[u] = cand
            Candidate_Num[u] = len(cand)

    # --------------------------------
    # Core 2-opt functions (maintain DIMES names)
    # --------------------------------
    def Get_2Opt_Delta(First_City: int, Second_City: int) -> float:
        nonlocal Total_Simulation_Times
        if Check_If_Two_City_Same_Or_Adjacent(First_City, Second_City):
            return -Inf_Cost

        First_Next_City = All_Node[First_City].Next_City
        Second_Next_City = All_Node[Second_City].Next_City

        Delta = (Get_Distance(First_City, First_Next_City) +
                 Get_Distance(Second_City, Second_Next_City) -
                 Get_Distance(First_City, Second_City) -
                 Get_Distance(First_Next_City, Second_Next_City))

        # MCTS counters (kept for structural parity; harmless for pure 2-opt)
        Chosen_Times[First_City, Second_City] += 1
        Chosen_Times[Second_City, First_City] += 1
        Chosen_Times[First_Next_City, Second_Next_City] += 1
        Chosen_Times[Second_Next_City, First_Next_City] += 1
        Total_Simulation_Times += 1

        return Delta

    def Apply_2Opt_Move(First_City: int, Second_City: int):
        Before_Distance = Get_Solution_Total_Distance()
        Delta = Get_2Opt_Delta(First_City, Second_City)

        First_Next_City = All_Node[First_City].Next_City
        Second_Next_City = All_Node[Second_City].Next_City

        # Perform the 2-opt reversal and reconnections
        Reverse_Sub_Path(First_Next_City, Second_City)
        All_Node[First_City].Next_City = Second_City
        All_Node[Second_City].Pre_City = First_City
        All_Node[First_Next_City].Next_City = Second_Next_City
        All_Node[Second_Next_City].Pre_City = First_Next_City

        # Backprop-like reinforcement on new edges (kept for parity with C)
        if Before_Distance > 0 and Delta > -Inf_Cost:
            Increase_Rate = Beta * (math.exp(float(Delta) / float(Before_Distance)) - 1.0)
            Weight[First_City, Second_City] += Increase_Rate
            Weight[Second_City, First_City] += Increase_Rate
            Weight[First_Next_City, Second_Next_City] += Increase_Rate
            Weight[Second_Next_City, First_Next_City] += Increase_Rate

    def Improve_By_2Opt_Move() -> bool:
        # Try first improving move in candidate neighborhoods
        for i in range(Virtual_City_Num):
            for j_idx in range(Candidate_Num[i]):
                Candidate_City = Candidate[i][j_idx]
                if Get_2Opt_Delta(i, Candidate_City) > 0:
                    Apply_2Opt_Move(i, Candidate_City)
                    return True
            # time stop (outer loop granularity)
            if time.time() - start_time >= time_limit:
                return False
        return False

    def Local_Search_by_2Opt_Move():
        nonlocal Current_Instance_Best_Distance
        # Iterate until no improvement or time limit exceeded
        while time.time() - start_time < time_limit and Improve_By_2Opt_Move() is True:
            pass

        Cur_Solution_Total_Distance = Get_Solution_Total_Distance()
        if Cur_Solution_Total_Distance < Current_Instance_Best_Distance:
            Current_Instance_Best_Distance = Cur_Solution_Total_Distance
            Store_Best_Solution()

    # --------------------------------
    # Initialize and run
    # --------------------------------
    # Build full distance table
    Calculate_All_Pair_Distance()

    # Build candidate neighborhoods (k-NN)
    Identify_Candidate_Set()

    # Initialize tour from initial_solution
    if len(initial_solution) != Virtual_City_Num:
        raise ValueError("initial_solution length must equal number of cities")
    if len(set(initial_solution)) != Virtual_City_Num:
        raise ValueError("initial_solution must be a permutation (no repeats)")
    Convert_Solution_To_All_Node([int(x) for x in initial_solution])

    # Set initial "best"
    Current_Instance_Best_Distance = Get_Solution_Total_Distance()
    Store_Best_Solution()

    # Run 2-opt local search under time budget
    Local_Search_by_2Opt_Move()

    # Build tour from best snapshot (start from Start_City to preserve orientation)
    Restore_Best_Solution()
    tour = [Start_City]
    cur = All_Node[Start_City].Next_City
    while cur != Start_City and cur != Null and len(tour) < Virtual_City_Num:
        tour.append(cur)
        cur = All_Node[cur].Next_City

    # Safety fallback
    if len(tour) != Virtual_City_Num:
        # linearize by walking Next_City from min index
        tour = []
        seen = set()
        cur = Start_City
        while cur not in seen and cur != Null and len(tour) < Virtual_City_Num:
            tour.append(cur)
            seen.add(cur)
            cur = All_Node[cur].Next_City
        if len(tour) != Virtual_City_Num:
            tour = list(range(Virtual_City_Num))

    return tour
