import heapq
import time


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
    """
    start_time = time.time()

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

    while open_set:
        # Get the node with the lowest f value
        _, _, current_node = heapq.heappop(open_set)
        open_set_hash.remove(current_node.name)

        # Mark as visited
        visited_nodes += 1

        # Add to closed set
        closed_set.add(current_node.name)

        # If we found the goal
        if current_node.name == goal:
            path = reconstruct_path(current_node)
            # Calculate total distance
            total_distance = 0
            for i in range(len(path) - 1):
                total_distance += graph[path[i]][path[i + 1]]

            execution_time = time.time() - start_time

            return {
                "path": path,
                "distance": total_distance,
                "visited_nodes": visited_nodes,
                "space_complexity": max_nodes_in_memory,
                "execution_time": execution_time
            }

        # Generate neighbors
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
                for _, _, node in open_set:
                    if node.name == neighbor_name and node.g <= g_score:
                        continue

            # Add to open set
            heapq.heappush(open_set, (neighbor_node.f, id(neighbor_node), neighbor_node))
            open_set_hash.add(neighbor_name)

            # Update max nodes in memory
            current_nodes_in_memory = len(open_set) + len(closed_set)
            max_nodes_in_memory = max(max_nodes_in_memory, current_nodes_in_memory)

    # If we get here, no path was found
    execution_time = time.time() - start_time

    return {
        "path": [],
        "distance": 0,
        "visited_nodes": visited_nodes,
        "space_complexity": max_nodes_in_memory,
        "execution_time": execution_time
    }