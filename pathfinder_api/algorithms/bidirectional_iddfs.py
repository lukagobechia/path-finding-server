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

        # Create reversed graph for backward search
        reversed_graph = self.reverse_graph(graph)

        start_time = time.time()

        # Iteratively increase depth
        for depth in range(max_depth + 1):
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

        execution_time = time.time() - start_time

        # Prepare results
        result = {
            "path": self.path,
            "path_length": len(self.path) - 1 if self.path_found else 0,
            "visited_nodes": self.visited_nodes,
            "space_complexity": self.max_nodes_in_memory,
            "execution_time": execution_time,
            "path_found": self.path_found,
            "meeting_point": self.meeting_point
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

        for node, neighbors in graph.items():
            for neighbor in neighbors:
                reversed_graph[neighbor].append(node)

        return reversed_graph


def bidirectional_iddfs(graph, start, goal, max_depth=float('inf')):
    """
    Wrapper function for Bidirectional IDDFS algorithm for unweighted graphs.

    Args:
        graph: Dictionary of node connections (adjacency list)
        start: Starting node name
        goal: Goal node name
        max_depth: Maximum depth to search (default: infinity)

    Returns:
        Dictionary containing path and metrics
    """
    # Create an instance of the algorithm
    algorithm = BidirectionalIDDFSAlgorithm()

    # Set a reasonable max_depth if not specified
    if max_depth == float('inf'):
        max_depth = len(graph)  # One depth level per node should be enough in bidirectional search

    # Run the algorithm
    result = algorithm.bidirectional_iddfs_search(graph, start, goal, max_depth)

    return result