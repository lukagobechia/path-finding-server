import heapq
import time
import math


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic (estimated cost from current node to goal)
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.name)


def reconstruct_path(node):
    """Reconstruct the path from start to goal."""
    path = []
    current = node
    while current:
        path.append(current.name)
        current = current.parent
    return path[::-1]  # Return reversed path


def calculate_branching_factor(graph):
    """Calculate the average branching factor of the graph."""
    total_nodes = len(graph)
    total_edges = sum(len(neighbors) for neighbors in graph.values())
    return total_edges / total_nodes if total_nodes > 0 else 1


def astar_search(graph, start, goal, heuristic):
    """
    A* pathfinding algorithm for finding the shortest path.

    Args:
        graph: Dictionary of node connections and distances
        start: Starting node name
        goal: Goal node name
        heuristic: Dictionary with estimated distances to goal

    Returns:
        Dictionary containing:
        - path: List of nodes in the path
        - distance: Total distance of the path
        - visited_nodes: Number of nodes visited during search
        - space_complexity: Maximum number of nodes in memory
        - execution_time: Time taken to execute the algorithm
        - time_complexity: Time complexity analysis
    """
    start_time = time.time()

    # Calculate average branching factor
    avg_branching_factor = calculate_branching_factor(graph)

    # Initialize start and goal nodes
    start_node = Node(start)
    goal_node = Node(goal)

    # Set the heuristic for the start node
    start_node.h = heuristic.get(start, 0)
    start_node.f = start_node.h

    # Initialize open and closed sets
    open_set = []
    heapq.heappush(open_set, (start_node.f, id(start_node), start_node))

    # Keep track of nodes in open set for faster lookup
    open_set_hash = {start}

    # Keep track of closed nodes
    closed_set = set()

    # Metrics
    visited_nodes = 0
    max_nodes_in_memory = 1  # Start with 1 for the start node
    expanded_nodes = 0  # Count of nodes where neighbors were explored

    while open_set:
        # Get the node with the lowest f value
        _, _, current_node = heapq.heappop(open_set)
        if current_node.name in open_set_hash:
            open_set_hash.remove(current_node.name)

        # Mark as visited
        visited_nodes += 1

        # Add to closed set
        closed_set.add(current_node.name)

        # If we found the goal
        if current_node.name == goal:
            path = reconstruct_path(current_node)
            path_length = len(path) - 1

            # Calculate total distance
            total_distance = 0
            for i in range(path_length):
                total_distance += graph[path[i]][path[i + 1]]

            execution_time = time.time() - start_time

            # Calculate time complexity
            # For A*, the time complexity is O(b^d) in the worst case
            # where b is the branching factor and d is the depth of the solution
            worst_case_complexity = avg_branching_factor ** path_length

            # With a perfect heuristic, A* is O(d)
            best_case_complexity = path_length

            # With a decent heuristic, A* is closer to O(b^(d*ε)) where ε < 1
            # Let's use the ratio of expanded nodes to estimate the impact of heuristic
            heuristic_effectiveness = expanded_nodes / len(graph) if len(graph) > 0 else 1
            heuristic_adjusted_complexity = avg_branching_factor ** (path_length * heuristic_effectiveness)

            # Comparison with other algorithms:
            # Dijkstra's time complexity is O(|V|^2) without priority queue, O(|E| + |V|log|V|) with priority queue
            dijkstra_complexity = len(graph) ** 2  # Simple approximation

            # BFS/DFS time complexity is O(|V| + |E|)
            bfs_complexity = len(graph) + sum(len(neighbors) for neighbors in graph.values())

            # Bidirectional search time complexity is approximately O(b^(d/2))
            bidirectional_complexity = avg_branching_factor ** (path_length / 2)

            time_complexity = {
                "avg_branching_factor": avg_branching_factor,
                "path_length": path_length,
                "expanded_nodes": expanded_nodes,
                "worst_case": f"O(b^d) ≈ {worst_case_complexity:.2e}",
                "best_case": f"O(d) = {best_case_complexity}",
                "heuristic_adjusted": f"O(b^(d*ε)) ≈ {heuristic_adjusted_complexity:.2e}",
                "comparison": {
                    "dijkstra": f"O(|V|^2) ≈ {dijkstra_complexity:.2e}",
                    "bfs": f"O(|V| + |E|) ≈ {bfs_complexity:.2e}",
                    "bidirectional": f"O(b^(d/2)) ≈ {bidirectional_complexity:.2e}"
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

        # Generate neighbors
        expanded_nodes += 1
        for neighbor_name, distance in graph[current_node.name].items():
            # Skip if neighbor is in closed set
            if neighbor_name in closed_set:
                continue

            # Calculate g value for this neighbor
            g_score = current_node.g + distance

            # Create neighbor node
            neighbor_node = Node(neighbor_name, current_node)
            neighbor_node.g = g_score
            neighbor_node.h = heuristic.get(neighbor_name, 0)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            # If neighbor is in open set with a better g score, skip
            if neighbor_name in open_set_hash:
                # Find the existing node in open_set
                for i, (_, _, node) in enumerate(open_set):
                    if node.name == neighbor_name:
                        if node.g <= g_score:
                            continue
                        else:
                            # Update the node in the open set (remove and add)
                            del open_set[i]
                            heapq.heapify(open_set)
                            break

            # Add to open set
            heapq.heappush(open_set, (neighbor_node.f, id(neighbor_node), neighbor_node))
            open_set_hash.add(neighbor_name)

            # Update max nodes in memory
            current_nodes_in_memory = len(open_set) + len(closed_set)
            max_nodes_in_memory = max(max_nodes_in_memory, current_nodes_in_memory)

    # If we get here, no path was found
    execution_time = time.time() - start_time

    # Calculate time complexity for no path case
    worst_case_complexity = avg_branching_factor ** len(graph)

    time_complexity = {
        "avg_branching_factor": avg_branching_factor,
        "path_length": None,
        "expanded_nodes": expanded_nodes,
        "worst_case": f"O(b^d) ≈ {worst_case_complexity:.2e}",
        "best_case": "N/A - No path found",
        "heuristic_adjusted": "N/A - No path found",
        "comparison": {
            "dijkstra": f"O(|V|^2) ≈ {len(graph) ** 2:.2e}",
            "bfs": f"O(|V| + |E|) ≈ {len(graph) + sum(len(neighbors) for neighbors in graph.values()):.2e}",
            "bidirectional": f"O(b^(d/2)) ≈ {avg_branching_factor ** (len(graph) / 2):.2e}"
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


def print_astar_results(result):
    """Print the results of the A* search in a readable format."""
    if result["path"]:
        print(f"Path found: {' -> '.join(map(str, result['path']))}")
        print(f"Total distance: {result['distance']}")
        print(f"Path length: {result['time_complexity']['path_length']} nodes")
    else:
        print("No path found")

    print(f"\nTime complexity analysis:")
    print(f"  - Average branching factor (b): {result['time_complexity']['avg_branching_factor']:.2f}")
    print(f"  - Worst case complexity: {result['time_complexity']['worst_case']}")
    print(f"  - Best case complexity: {result['time_complexity']['best_case']}")
    print(f"  - Heuristic-adjusted complexity: {result['time_complexity']['heuristic_adjusted']}")

    print(f"\nComparison with other algorithms:")
    print(f"  - Dijkstra's: {result['time_complexity']['comparison']['dijkstra']}")
    print(f"  - BFS: {result['time_complexity']['comparison']['bfs']}")
    print(f"  - Bidirectional search: {result['time_complexity']['comparison']['bidirectional']}")

    print(f"\nPerformance metrics:")
    print(f"  - Nodes visited: {result['visited_nodes']}")
    print(f"  - Nodes expanded: {result['time_complexity']['expanded_nodes']}")
    print(f"  - Max nodes in memory: {result['space_complexity']}")
    print(f"  - Execution time: {result['execution_time']:.6f} seconds")