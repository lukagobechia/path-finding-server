import time
from collections import defaultdict


class BidirectionalIDDFSAlgorithm:
    def __init__(self):
        self.path = []
        self.visited_nodes = 0
        self.max_nodes_in_memory = 0
        self.current_nodes_in_memory = 0
        self.path_found = False
        self.forward_visited = set()
        self.backward_visited = set()
        self.meeting_point = None
        self.forward_paths = {}
        self.backward_paths = {}
        self.total_distance = 0

    def dls_forward(self, graph, current, depth, path, visited):
        """
        Depth-limited search in the forward direction (from start)
        """
        self.visited_nodes += 1
        self.current_nodes_in_memory = len(visited) + len(self.backward_visited)
        self.max_nodes_in_memory = max(self.max_nodes_in_memory, self.current_nodes_in_memory)

        # Store the path to this node
        self.forward_paths[current] = path.copy()
        self.forward_visited.add(current)

        # Check if we've met the backward search
        if current in self.backward_visited:
            self.meeting_point = current
            return True

        # If we've reached depth limit, stop exploring
        if depth <= 0:
            return False

        # Explore neighbors
        for neighbor in graph.get(current, {}):
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)

                if self.dls_forward(graph, neighbor, depth - 1, path, visited):
                    return True

                # Backtrack
                path.pop()
                visited.remove(neighbor)

        return False

    def dls_backward(self, reversed_graph, current, depth, path, visited):
        """
        Depth-limited search in the backward direction (from goal)
        """
        self.visited_nodes += 1
        self.current_nodes_in_memory = len(visited) + len(self.forward_visited)
        self.max_nodes_in_memory = max(self.max_nodes_in_memory, self.current_nodes_in_memory)

        # Store the path to this node
        self.backward_paths[current] = path.copy()
        self.backward_visited.add(current)

        # Check if we've met the forward search
        if current in self.forward_visited:
            self.meeting_point = current
            return True

        # If we've reached depth limit, stop exploring
        if depth <= 0:
            return False

        # Explore neighbors
        for neighbor in reversed_graph.get(current, {}):
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)

                if self.dls_backward(reversed_graph, neighbor, depth - 1, path, visited):
                    return True

                # Backtrack
                path.pop()
                visited.remove(neighbor)

        return False

    def calculate_distance(self, graph, path):
        """
        Calculate the total distance of a path
        """
        if not path or len(path) < 2:
            return 0

        total_distance = 0
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]

            # Handle different graph representations
            if isinstance(graph[current], dict):
                # Weighted graph
                if next_node in graph[current]:
                    total_distance += graph[current][next_node]
                else:
                    # This should not happen in a valid path
                    return float('inf')
            elif isinstance(graph[current], list):
                # Unweighted graph
                if next_node in graph[current]:
                    total_distance += 1
                else:
                    # This should not happen in a valid path
                    return float('inf')
            else:
                # Unknown graph format
                total_distance += 1

        return total_distance

    def bidirectional_iddfs_search(self, graph, start, goal, max_depth):
        """
        Main bidirectional IDDFS search function
        """
        # Reset state
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

        # Validation checks
        if start not in graph:
            print(f"Warning: Start node '{start}' not in graph")
            return self._create_result(graph, max_depth, 0, time.time())

        if goal not in graph:
            print(f"Warning: Goal node '{goal}' not in graph")
            return self._create_result(graph, max_depth, 0, time.time())

        if start == goal:
            self.path = [start]
            self.path_found = True
            return self._create_result(graph, 0, 0, time.time())

        # Build reversed graph for backward search
        reversed_graph = self.reverse_graph(graph)

        start_time = time.time()

        # Compute branching factor
        total_nodes = len(graph)
        total_edges = 0
        for node, neighbors in graph.items():
            if isinstance(neighbors, dict):
                total_edges += len(neighbors)
            elif isinstance(neighbors, list):
                total_edges += len(neighbors)

        avg_branching_factor = total_edges / total_nodes if total_nodes > 0 else 1

        # Perform IDDFS
        final_depth = 0
        for depth in range(max_depth + 1):
            final_depth = depth

            # Reset for new depth iteration
            self.forward_visited = set([start])
            self.backward_visited = set([goal])
            self.forward_paths = {start: [start]}
            self.backward_paths = {goal: [goal]}

            # Perform limited DFS from both directions
            forward_result = self.dls_forward(graph, start, depth, [start], {start})
            if forward_result:
                self.path_found = True
                break

            backward_result = self.dls_backward(reversed_graph, goal, depth, [goal], {goal})
            if backward_result:
                self.path_found = True
                break

        # Construct full path if found
        if self.path_found and self.meeting_point:
            forward_half = self.forward_paths[self.meeting_point]
            backward_half = self.backward_paths[self.meeting_point]
            backward_half.reverse()

            # Combine paths, avoiding duplicate meeting point
            self.path = forward_half + backward_half[1:]

            # Calculate path distance
            self.total_distance = self.calculate_distance(graph, self.path)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Create and return result
        return self._create_result(graph, final_depth, execution_time, start_time)

    def _create_result(self, graph, final_depth, execution_time, start_time):
        """Helper method to create consistent result dictionary"""
        total_nodes = len(graph)
        total_edges = 0
        for node, neighbors in graph.items():
            if isinstance(neighbors, dict):
                total_edges += len(neighbors)
            elif isinstance(neighbors, list):
                total_edges += len(neighbors)

        avg_branching_factor = total_edges / total_nodes if total_nodes > 0 else 1

        # Calculate complexity metrics
        std_complexity = avg_branching_factor ** final_depth if final_depth > 0 else 1

        if self.path_found and len(self.path) > 1:
            actual_depth = len(self.path) - 1
            bi_complexity = avg_branching_factor ** (actual_depth / 2)
        else:
            bi_complexity = avg_branching_factor ** (final_depth / 2)

        improvement_ratio = std_complexity / bi_complexity if bi_complexity > 0 else float('inf')

        # Create result dictionary
        result = {
            "path": self.path,
            "path_length": len(self.path) - 1 if self.path_found else 0,
            "distance": self.total_distance,
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
        Create a reversed version of the graph where all edges point in the opposite direction
        """
        reversed_graph = defaultdict(dict)

        for node, neighbors in graph.items():
            # Ensure the node exists in the reversed graph
            if node not in reversed_graph:
                reversed_graph[node] = {}

            if isinstance(neighbors, dict):
                # Weighted graph
                for neighbor, weight in neighbors.items():
                    if neighbor not in reversed_graph:
                        reversed_graph[neighbor] = {}
                    reversed_graph[neighbor][node] = weight
            elif isinstance(neighbors, list):
                # Unweighted graph represented as lists
                for neighbor in neighbors:
                    if neighbor not in reversed_graph:
                        reversed_graph[neighbor] = {}
                    reversed_graph[neighbor][node] = 1  # Default weight of 1 for unweighted

        return dict(reversed_graph)


def bidirectional_iddfs(graph, start, goal, max_depth=float('inf')):
    """
    Run bidirectional IDDFS and print results
    """
    algorithm = BidirectionalIDDFSAlgorithm()

    # Use graph size as default max_depth if not specified
    if max_depth == float('inf'):
        max_depth = len(graph)

        # Run the search
    result = algorithm.bidirectional_iddfs_search(graph, start, goal, max_depth)

    # Print results
    if result["path_found"]:
        print(f"Path found: {' -> '.join(map(str, result['path']))}")
        print(f"Total distance: {result['distance']}")
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
