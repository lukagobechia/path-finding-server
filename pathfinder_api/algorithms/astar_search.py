import heapq
import time
from collections import defaultdict


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost from current to goal)
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.name)


def reconstruct_path(node):
    path = []
    current = node
    while current:
        path.append(current.name)
        current = current.parent
    return path[::-1]  # Return reversed path


def calculate_branching_factor(graph):
    total_nodes = len(graph)
    total_edges = sum(len(neighbors) for neighbors in graph.values())
    return total_edges / total_nodes if total_nodes > 0 else 1


def check_heuristic_consistency(graph, heuristic, goal):
    """Check if the heuristic is consistent (monotonic)."""
    for node in graph:
        h_node = heuristic.get(node, 0)
        for neighbor, cost in graph[node].items():
            h_neighbor = heuristic.get(neighbor, 0)
            # For consistency: h(n) ≤ c(n,a,n') + h(n')
            if h_node > cost + h_neighbor and node != goal:
                return False
    return True


def astar_search(graph, start, goal, heuristic):
    start_time = time.time()

    # Calculate average branching factor
    avg_branching_factor = calculate_branching_factor(graph)

    # Check heuristic consistency (admissibility is a subset of consistency)
    is_consistent = check_heuristic_consistency(graph, heuristic, goal)
    if not is_consistent:
        print("Warning: Heuristic may not be consistent, A* optimality not guaranteed")

    # Initialize nodes
    start_node = Node(start)
    start_node.h = heuristic.get(start, 0)
    start_node.f = start_node.h

    # Priority queue for open set
    open_set = []
    # Use counter for tie-breaking when f-values are equal
    counter = 0
    heapq.heappush(open_set, (start_node.f, counter, start_node))
    counter += 1

    # Dictionary to track nodes in the open set for quick lookup and updates
    open_dict = {start: start_node}

    # Closed set to track visited nodes
    closed_set = set()

    # Metrics
    visited_nodes = 0
    max_nodes_in_memory = 1
    expanded_nodes = 0

    while open_set:
        # Get the node with the lowest f-score
        _, _, current_node = heapq.heappop(open_set)

        # Remove from open dictionary
        del open_dict[current_node.name]

        visited_nodes += 1

        # Check if we reached the goal
        if current_node.name == goal:
            path = reconstruct_path(current_node)
            path_length = len(path) - 1

            # Calculate total path distance
            total_distance = 0
            for i in range(path_length):
                total_distance += graph[path[i]][path[i + 1]]

            execution_time = time.time() - start_time

            # Complexity analysis
            worst_case_complexity = avg_branching_factor ** path_length
            best_case_complexity = path_length
            heuristic_effectiveness = expanded_nodes / len(graph) if len(graph) > 0 else 1
            heuristic_adjusted_complexity = avg_branching_factor ** (path_length * heuristic_effectiveness)
            dijkstra_complexity = len(graph) * (len(graph) + sum(len(neighbors) for neighbors in graph.values()))
            bfs_complexity = len(graph) + sum(len(neighbors) for neighbors in graph.values())
            bidirectional_complexity = 2 * (avg_branching_factor ** (path_length / 2))

            time_complexity = {
                "avg_branching_factor": avg_branching_factor,
                "path_length": path_length,
                "expanded_nodes": expanded_nodes,
                "worst_case": f"O(b^d) ≈ {worst_case_complexity:.2e}",
                "best_case": f"O(d) = {best_case_complexity}",
                "heuristic_adjusted": f"O(b^(d*ε)) ≈ {heuristic_adjusted_complexity:.2e}",
                "heuristic_consistency": is_consistent,
                "comparison": {
                    "dijkstra": f"O(|V|log|V| + |E|) ≈ {dijkstra_complexity:.2e}",
                    "bfs": f"O(|V| + |E|) ≈ {bfs_complexity:.2e}",
                    "bidirectional": f"O(2*b^(d/2)) ≈ {bidirectional_complexity:.2e}"
                }
            }

            return {
                "path": path,
                "distance": total_distance,
                "visited_nodes": visited_nodes,
                "space_complexity": max_nodes_in_memory,
                "execution_time": execution_time,
                "time_complexity": time_complexity
            }

        # Add to closed set since we're processing this node
        closed_set.add(current_node.name)
        expanded_nodes += 1

        # Process all neighbors
        for neighbor_name, distance in graph[current_node.name].items():
            # Skip if neighbor is already processed
            if neighbor_name in closed_set:
                continue

            # Calculate tentative g score
            g_score = current_node.g + distance

            # Check if neighbor is in open set
            if neighbor_name in open_dict:
                # If we already found a better path, skip
                if g_score >= open_dict[neighbor_name].g:
                    continue

                # Found a better path to this neighbor
                neighbor_node = open_dict[neighbor_name]
                neighbor_node.g = g_score
                neighbor_node.parent = current_node
                neighbor_node.f = g_score + neighbor_node.h

                # We don't need to update the heap because we'll compare f-values
                # when we pop nodes. The old entry will be skipped when processed
                # since it won't be in the open_dict anymore

                # Add the updated node back to the heap
                heapq.heappush(open_set, (neighbor_node.f, counter, neighbor_node))
                counter += 1
            else:
                # Create a new neighbor node
                neighbor_node = Node(neighbor_name, current_node)
                neighbor_node.g = g_score
                neighbor_node.h = heuristic.get(neighbor_name, 0)
                neighbor_node.f = g_score + neighbor_node.h

                # Add to open set
                heapq.heappush(open_set, (neighbor_node.f, counter, neighbor_node))
                counter += 1
                open_dict[neighbor_name] = neighbor_node

            # Update max nodes in memory
            current_nodes_in_memory = len(open_set) + len(closed_set)
            max_nodes_in_memory = max(max_nodes_in_memory, current_nodes_in_memory)

    # No path found
    execution_time = time.time() - start_time

    # Calculate complexity for no-path case
    worst_case_complexity = avg_branching_factor ** len(graph)
    dijkstra_complexity = len(graph) * (len(graph) + sum(len(neighbors) for neighbors in graph.values()))

    time_complexity = {
        "avg_branching_factor": avg_branching_factor,
        "path_length": None,
        "expanded_nodes": expanded_nodes,
        "worst_case": f"O(b^d) ≈ {worst_case_complexity:.2e}",
        "best_case": "N/A - No path found",
        "heuristic_adjusted": "N/A - No path found",
        "heuristic_consistency": is_consistent,
        "comparison": {
            "dijkstra": f"O(|V|log|V| + |E|) ≈ {dijkstra_complexity:.2e}",
            "bfs": f"O(|V| + |E|) ≈ {len(graph) + sum(len(neighbors) for neighbors in graph.values()):.2e}",
            "bidirectional": f"O(2*b^(|V|/2)) ≈ {2 * (avg_branching_factor ** (len(graph) / 2)):.2e}"
        }
    }

    return {
        "path": [],
        "distance": 0,
        "visited_nodes": visited_nodes,
        "space_complexity": max_nodes_in_memory,
        "execution_time": execution_time,
        "time_complexity": time_complexity
    }
