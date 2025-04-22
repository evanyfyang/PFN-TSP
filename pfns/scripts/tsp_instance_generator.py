import random
import json
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import os
import argparse


def create_random_tsp_instance(num_nodes):
    """Generate a random TSP instance"""
    return [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(num_nodes)]


def distance_callback(from_index, to_index, manager, locations):
    """Calculate the Euclidean distance between two nodes"""
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return int(((locations[from_node][0] - locations[to_node][0]) ** 2 + 
                (locations[from_node][1] - locations[to_node][1]) ** 2) ** 0.5)


def solve_tsp(locations):
    """Solve the TSP using OR-Tools"""
    manager = pywrapcp.RoutingIndexManager(len(locations), 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    transit_callback_index = routing.RegisterTransitCallback(
        lambda from_index, to_index: distance_callback(from_index, to_index, manager, locations))
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        return route
    else:
        return None


def save_tsp_instance_and_solution(num_nodes, file_path):
    """Generate a TSP instance and its solution, then save to a file"""
    locations = create_random_tsp_instance(num_nodes)
    solution = solve_tsp(locations)
    data = {
        'locations': locations,
        'solution': solution
    }
    with open(file_path, 'w') as f:
        json.dump(data, f)


def save_multiple_tsp_instances_in_one_file(num_nodes, num_instances, directory):
    """Generate multiple TSP instances and solutions, save to one file, append if file exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'tsp_instances_{num_nodes}_nodes.json')
    # If the file exists, read existing data
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            all_data = json.load(f)
    else:
        all_data = []
    # Generate new instances and append
    for _ in range(num_instances):
        locations = create_random_tsp_instance(num_nodes)
        solution = solve_tsp(locations)
        data = {
            'locations': locations,
            'solution': solution
        }
        all_data.append(data)
    # Save all data
    with open(file_path, 'w') as f:
        json.dump(all_data, f)


def main():
    parser = argparse.ArgumentParser(description='Generate TSP instances and solutions.')
    parser.add_argument('--num_nodes', type=int, required=True, help='Number of nodes in each TSP instance')
    parser.add_argument('--num_instances', type=int, required=True, help='Number of TSP instances to generate')
    parser.add_argument('--directory', type=str, default='../datasets/tsp', help='Directory to save the TSP instances')
    args = parser.parse_args()
    save_multiple_tsp_instances_in_one_file(args.num_nodes, args.num_instances, args.directory)


if __name__ == '__main__':
    main() 