import time
from collections import defaultdict


class BidirectionalIDDFSAlgorithm:
    def __init__(self):
        # Path and other metrics will be stored here
        self.path = []
        self.visited_nodes = 0
        self.max_nodes_in_memory = 0
        self.current_nodes_in_memory = 0
        self.path_found = False
        self.forward_visited = set()  # Track nodes visited from start
        self.backward_visited = set()  # Track nodes visited from goal
        self.meeting_point = None  # Node where forward and backward searches meet
        self.forward_paths = {}  # Maps node -> path from start to node
        self.backward_paths = {}  # Maps node -> path from node to goal
        self.total_distance = 0  # Total distance of the path

    def dls_forward(self, graph, current, depth, path, visited):
        """
        Forward Depth-Limited Search implementation.

        Args:
            graph: Dictionary representing the graph
            current: Current node
            depth: Current depth limit
            path: Current path being explored
            visited: Set of visited nodes in current path

        Returns:
            Boolean: True if a meeting point is found, False otherwise
        """
        # Update metrics
        self.visited_nodes += 1
        self.current_nodes_in_memory = len(visited) + len(self.backward_visited)
        self.max_nodes_in_memory = max(self.max_nodes_in_memory, self.current_nodes_in_memory)

        # Store the path to this node in forward_paths
        self.forward_paths[current] = path.copy()
        self.forward_visited.add(current)

        # Check if we found a node that has been visited by backward search
        if current in self.backward_visited:
            self.meeting_point = current
            return True

        # If depth limit is reached, stop recursion
        if depth <= 0:
            return False

        # Explore neighbors
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                # Add to path and visited
                path.append(neighbor)
                visited.add(neighbor)

                # Recursive call with reduced depth
                if self.dls_forward(graph, neighbor, depth - 1, path, visited):
                    return True

                # Backtrack
                path.pop()
                visited.remove(neighbor)

        return False

    def dls_backward(self, reversed_graph, current, depth, path, visited):
        """
        Backward Depth-Limited Search implementation.

        Args:
            reversed_graph: Dictionary representing the reversed graph
            current: Current node (starting from goal)
            depth: Current depth limit
            path: Current path being explored (from current to goal)
            visited: Set of visited nodes in current path

        Returns:
            Boolean: True if a meeting point is found, False otherwise
        """
        # Update metrics
        self.visited_nodes += 1
        self.current_nodes_in_memory = len(visited) + len(self.forward_visited)
        self.max_nodes_in_memory = max(self.max_nodes_in_memory, self.current_nodes_in_memory)

        # Store the path from this node to goal in backward_paths
        self.backward_paths[current] = path.copy()
        self.backward_visited.add(current)

        # Check if we found a node that has been visited by forward search
        if current in self.forward_visited:
            self.meeting_point = current
            return True

        # If depth limit is reached, stop recursion
        if depth <= 0:
            return False

        # Explore neighbors in reversed graph
        for neighbor in reversed_graph.get(current, []):
            if neighbor not in visited:
                # Add to path and visited
                path.append(neighbor)
                visited.add(neighbor)

                # Recursive call with reduced depth
                if self.dls_backward(reversed_graph, neighbor, depth - 1, path, visited):
                    return True

                # Backtrack
                path.pop()
                visited.remove(neighbor)

        return False

    def calculate_distance(self, graph, path):
        """
        Calculate the total distance of a path based on edge weights.

        Args:
            graph: Dictionary representing the weighted graph
            path: List of nodes representing the path

        Returns:
            Float: Total distance of the path
        """
        total_distance = 0
        for i in range(len(path) - 1):
            # Check if the edge has a weight; if not, use 1 as default (unweighted graph)
            if isinstance(graph[path[i]], dict):
                # If graph uses dict for neighbor weights {neighbor: weight}
                total_distance += graph[path[i]].get(path[i + 1], 1)
            elif isinstance(graph[path[i]], list):
                # If graph uses list for neighbors, assume weight 1 (unweighted)
                total_distance += 1
            else:
                # Fallback for other graph representations
                total_distance += 1

        return total_distance

    def bidirectional_iddfs_search(self, graph, start, goal, max_depth):
        """
        Bidirectional Iterative Deepening Depth-First Search.

        Args:
            graph: Dictionary representing the graph
            start: Starting node
            goal: Target node
            max_depth: Maximum depth to search

        Returns:
            Dictionary containing results and metrics
        """
        # Reset metrics
        self.visited_nodes = 0
        self.max_nodes_in_memory = 0
        self.path = []
        self.path_found = False
        self.forward_visited = set()
        self.backward_visited = set()
        self.meeting_point = None
        self.forward_paths = {}
        self.backward_paths = {}
        self.total_distance = 0

        # Create reversed graph for backward search
        reversed_graph = self.reverse_graph(graph)

        start_time = time.time()

        # Variables to calculate branching factor
        total_nodes = len(graph)
        total_edges = sum(len(neighbors) if isinstance(neighbors, list) else len(neighbors.keys())
                          for neighbors in graph.values())
        avg_branching_factor = total_edges / total_nodes if total_nodes > 0 else 1

        # Iteratively increase depth
        final_depth = 0
        for depth in range(max_depth + 1):
            final_depth = depth
            # Initialize paths and visited sets for this iteration
            forward_path = [start]
            backward_path = [goal]
            forward_visited = {start}
            backward_visited = {goal}

            # Perform one step of forward search
            if self.dls_forward(graph, start, depth, forward_path, forward_visited):
                self.path_found = True
                break

            # Perform one step of backward search
            if self.dls_backward(reversed_graph, goal, depth, backward_path, backward_visited):
                self.path_found = True
                break

        # Construct the complete path if a meeting point was found
        if self.path_found:
            # Get path from start to meeting point
            forward_half = self.forward_paths[self.meeting_point]

            # Get path from meeting point to goal
            backward_half = self.backward_paths[self.meeting_point]
            backward_half.reverse()  # Reverse to get correct order

            # Combine paths (removing duplicate meeting point)
            self.path = forward_half + backward_half[1:]

            # Calculate the total distance
            self.total_distance = self.calculate_distance(graph, self.path)

        execution_time = time.time() - start_time

        # Calculate theoretical time complexity
        # For standard IDDFS: O(b^d)
        std_complexity = avg_branching_factor ** final_depth if final_depth > 0 else 1

        # For bidirectional IDDFS: O(b^(d/2) + b^(d/2)) ≈ O(b^(d/2))
        if self.path_found and len(self.path) > 1:
            # Use the actual path length to calculate the actual depth
            actual_depth = len(self.path) - 1
            bi_complexity = avg_branching_factor ** (actual_depth / 2)
        else:
            # Fallback if no path found
            bi_complexity = avg_branching_factor ** (final_depth / 2)

        # Calculate complexity improvement ratio
        improvement_ratio = std_complexity / bi_complexity if bi_complexity > 0 else float('inf')

        # Prepare results
        result = {
            "path": self.path,
            "path_length": len(self.path) - 1 if self.path_found else 0,
            "total_distance": self.total_distance,
            "visited_nodes": self.visited_nodes,
            "space_complexity": self.max_nodes_in_memory,
            "execution_time": execution_time,
            "path_found": self.path_found,
            "meeting_point": self.meeting_point,
            "time_complexity": {
                "avg_branching_factor": avg_branching_factor,
                "depth": final_depth,
                "actual_path_length": len(self.path) - 1 if self.path_found else None,
                "standard_iddfs_complexity": f"O(b^d) ≈ {std_complexity:.2e}",
                "bidirectional_complexity": f"O(b^(d/2)) ≈ {bi_complexity:.2e}",
                "improvement_ratio": f"{improvement_ratio:.2f}x"
            }
        }

        return result

    def reverse_graph(self, graph):
        """
        Creates a reversed version of the graph where all edges point in the opposite direction.

        Args:
            graph: Dictionary representing the original graph

        Returns:
            Dictionary representing the reversed graph
        """
        reversed_graph = defaultdict(list)

        # Handle different graph representations
        for node, neighbors in graph.items():
            if isinstance(neighbors, dict):
                # If graph uses dict for neighbor weights {neighbor: weight}
                for neighbor, weight in neighbors.items():
                    if neighbor not in reversed_graph:
                        reversed_graph[neighbor] = {}
                    reversed_graph[neighbor][node] = weight
            elif isinstance(neighbors, list):
                # If graph uses list for neighbors
                for neighbor in neighbors:
                    if neighbor not in reversed_graph:
                        reversed_graph[neighbor] = []
                    reversed_graph[neighbor].append(node)

        return reversed_graph


def bidirectional_iddfs(graph, start, goal, max_depth=float('inf')):
    """
    Wrapper function for Bidirectional IDDFS algorithm for graphs.

    Args:
        graph: Dictionary of node connections (adjacency list or weighted graph)
        start: Starting node name
        goal: Goal node name
        max_depth: Maximum depth to search (default: infinity)

    Returns:
        Dictionary containing path, distance, time complexity, and other metrics
    """
    # Create an instance of the algorithm
    algorithm = BidirectionalIDDFSAlgorithm()

    # Set a reasonable max_depth if not specified
    if max_depth == float('inf'):
        max_depth = len(graph)  # One depth level per node should be enough in bidirectional search

    # Run the algorithm
    result = algorithm.bidirectional_iddfs_search(graph, start, goal, max_depth)

    # Print summary of results for convenience
    if result["path_found"]:
        print(f"Path found: {' -> '.join(map(str, result['path']))}")
        print(f"Total distance: {result['total_distance']}")
        print(f"Path length: {result['path_length']} nodes")
    else:
        print("No path found")

    print(f"Time complexity analysis:")
    print(f"  - Average branching factor (b): {result['time_complexity']['avg_branching_factor']:.2f}")
    print(f"  - Standard IDDFS complexity: {result['time_complexity']['standard_iddfs_complexity']}")
    print(f"  - Bidirectional IDDFS complexity: {result['time_complexity']['bidirectional_complexity']}")
    print(f"  - Improvement: {result['time_complexity']['improvement_ratio']}")
    print(f"Performance metrics:")
    print(f"  - Nodes visited: {result['visited_nodes']}")
    print(f"  - Max nodes in memory: {result['space_complexity']}")
    print(f"  - Execution time: {result['execution_time']:.6f} seconds")

    return result