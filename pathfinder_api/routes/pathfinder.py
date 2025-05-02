from flask import Blueprint, request, jsonify
from flask import jsonify
from bson import ObjectId
from pathfinder_api.db import collection
import time
import sys
from pathfinder_api.algorithms.astar_search import astar_search
from pathfinder_api.algorithms.bidirectional_iddfs import bidirectional_iddfs
from pymongo import MongoClient, DESCENDING, ASCENDING
from bson.objectid import ObjectId
from bson.errors import InvalidId


pathfinder_bp = Blueprint('pathfinder', __name__)

@pathfinder_bp.route('/find_path', methods=['POST'])
def find_path():
    """API endpoint that finds the shortest path using specified algorithm."""
    data = request.get_json()

    # Extract necessary information
    graph = data.get('graph', {})
    start = data.get('start', '')
    goal = data.get('goal', '')
    method = data.get('method', 'astar').lower()
    heuristic = data.get('heuristic', {})
    max_depth = data.get('max_depth', len(graph) * 2)  # Default: twice the number of nodes

    # Error handling
    if not graph:
        return jsonify({"error": "Graph is empty"}), 400
    if not start or not goal:
        return jsonify({"error": "Start or goal node not specified"}), 400
    if start not in graph:
        return jsonify({"error": f"Start node '{start}' not found in graph"}), 400
    if goal not in graph and goal not in heuristic:
        return jsonify({"error": f"Goal node '{goal}' not found in graph or heuristic"}), 400

    # Check if method is supported
    if method not in ['astar', 'iddfs']:
        return jsonify({"error": f"Method '{method}' not supported. Available methods: 'astar', 'iddfs'"}), 400

    result = {}

    # Execute the selected algorithm
    if method == 'astar':
        result = astar_search(graph, start, goal, heuristic)
        time_complexity = "O(E log V) where E is the number of edges and V is the number of vertices"
        space_complexity = f"O(V) where V is the number of vertices. Max nodes in memory: {result['space_complexity']}"
    else:  # iddfs
        result = bidirectional_iddfs(graph, start, goal, max_depth)
        time_complexity = "O(b^d) where b is the branching factor and d is the depth of the solution"
        space_complexity = f"O(d) where d is the depth of the solution. Max nodes in memory: {result['space_complexity']}"

    # Prepare response
    response = {
        "path": result["path"],
        "distance": result["path_length"],
        "time_complexity": time_complexity,
        "space_complexity": space_complexity,
        "visited_nodes": result["visited_nodes"],
        "execution_time_seconds": result["execution_time"]
    }

    if method == 'iddfs':
        response["path_found"] = result["path_found"]
        response["max_depth_used"] = max_depth

    return jsonify(response)


@pathfinder_bp.route('/save_graph', methods=['POST'])
def save_graph():
    data = request.get_json()
    if not data or "graph" not in data:
        return jsonify({"error": "Graph data missing"}), 400

    # Get graph name from request body, or fallback to a default name if not provided
    graph_id = data.get("name")  # Allow the client to specify a name

    if not graph_id:
        return jsonify({"error": "Graph name is required"}), 400  # Ensure a name is provided

    # Check if the graph already exists
    existing_graph = collection.find_one({"name": graph_id})
    if existing_graph:
        return jsonify({"error": f"Graph '{graph_id}' already exists. Use a different name or update the existing one."}), 400

    # If graph does not exist, insert new graph
    collection.insert_one({
        "name": graph_id,
        "graph": data["graph"],
        "heuristic": data.get("heuristic"),
        "start": data.get("start"),
        "goal": data.get("goal"),
        "method": data.get("method"),
        "max_depth": data.get("max_depth")
    })

    return jsonify({"message": f"Graph '{graph_id}' saved successfully"})




@pathfinder_bp.route('/graph/<graph_id>', methods=['GET'])
def get_graph(graph_id):
    """Retrieve a specific graph by ObjectId."""
    try:
        object_id = ObjectId(graph_id)
    except InvalidId:
        return jsonify({"error": "Invalid graph ID format"}), 400

    graph = collection.find_one({"_id": object_id})
    if not graph:
        return jsonify({"error": f"Graph with ID '{graph_id}' not found"}), 404

    # Convert ObjectId to string
    graph['id'] = str(graph.pop('_id'))

    return jsonify(graph)


@pathfinder_bp.route('/graph', methods=['GET'])
def get_all_graphs():
    """Retrieve all graphs with optional filtering and pagination."""
    try:
        # Extract query parameters
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        sort_by = request.args.get('sort_by', 'updated_at')
        sort_order = request.args.get('sort_order', 'desc')
        filter_name = request.args.get('name')

        # Build MongoDB query
        query = {}
        if filter_name:
            query["name"] = {"$regex": filter_name, "$options": "i"}  # Case-insensitive filter

        skip = (page - 1) * limit
        sort_direction = DESCENDING if sort_order.lower() == 'desc' else ASCENDING

        cursor = collection.find(query) \
                           .sort(sort_by, sort_direction) \
                           .skip(skip) \
                           .limit(limit)

        graphs = []
        for doc in cursor:
            doc['id'] = str(doc.pop('_id'))  # Rename and convert ObjectId to string
            graphs.append(doc)

        total_documents = collection.count_documents(query)
        total_pages = (total_documents + limit - 1) // limit

        return jsonify({
            "graphs": graphs,
            "page": page,
            "limit": limit,
            "total_documents": total_documents,
            "total_pages": total_pages
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@pathfinder_bp.route('/graph/<graph_id>', methods=['PATCH'])
def update_graph(graph_id):
    """Update an existing graph by MongoDB _id."""
    if not graph_id:
        return jsonify({"error": "Graph ID is required"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided for update"}), 400

    # Check if graph exists by _id
    try:
        graph_object_id = ObjectId(graph_id)
    except Exception as e:
        return jsonify({"error": "Invalid Graph ID format"}), 400

    existing_graph = collection.find_one({"_id": graph_object_id})
    if not existing_graph:
        return jsonify({"error": f"Graph with ID '{graph_id}' not found"}), 404

    # Prepare update document
    update_data = {"updated_at": time.time()}

    # Add fields to update
    allowed_fields = ["name","graph", "heuristic", "start", "goal", "method", "max_depth"]
    for field in allowed_fields:
        if field in data:
            update_data[field] = data[field]

    # Execute update
    result = collection.update_one(
        {"_id": graph_object_id},
        {"$set": update_data}
    )

    if result.modified_count:
        return jsonify({
            "message": f"Graph with ID '{graph_id}' updated successfully",
            "updated_fields": list(update_data.keys())
        })
    else:
        return jsonify({
            "message": f"No changes made to graph with ID '{graph_id}'",
        })


@pathfinder_bp.route('/graph/<graph_id>', methods=['DELETE'])
def delete_graph(graph_id):
    """Delete a graph by ID."""

    try:
        object_id = ObjectId(graph_id)
    except InvalidId:
        return jsonify({"error": "Invalid graph ID format"}), 400
    # Check if graph exists
    existing_graph = collection.find_one({"_id": object_id})
    if not existing_graph:
        return jsonify({"error": f"Graph '{object_id}' not found"}), 404

    # Execute deletion
    result = collection.delete_one({"_id": object_id})

    if result.deleted_count:
        return jsonify({
            "message": f"Graph '{object_id}' deleted successfully"
        })
    else:
        return jsonify({
            "error": f"Failed to delete graph '{graph_id}'"
        }), 500
