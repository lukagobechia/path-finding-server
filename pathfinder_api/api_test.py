import requests
import json
import time


def test_api():
    # Test data - Georgian cities graph
    test_data ={
  "graph": {
    "Telavi": {"Sagarejo": 37, "Tbilisi": 60},
    "Sagarejo": {"Gardabani": 45},
    "Gardabani": {"Tbilisi": 40},
    "Tbilisi": {"Mtskheta": 25},
    "Mtskheta": {"Kaspi": 30},
    "Kaspi": {"Gori": 25},
    "Gori": {"Khashuri": 45},
    "Khashuri": {"Kharagauli": 45},
    "Kharagauli": {"Zestafoni": 35},
    "Zestafoni": {"Kutaisi": 30},
    "Kutaisi": {"Samtredia": 27},
    "Samtredia": {"Senaki": 70},
    "Senaki": {"Zugdidi": 40},
    "Zugdidi": {"Gali": 40},
    "Gali": {"Otchamtchire": 35},
    "Otchamtchire": {"Sokhumi": 40},
    "Sokhumi": {"Gudauta": 40},
    "Gudauta": {"Gagra": 35},
    "Gagra": {}
  },
  "start": "Telavi",
  "goal": "Sokhumi",
  "method": "astar",
  "max_depth": 20,
  "heuristic": {
    "Telavi": 340,
    "Sagarejo": 330,
    "Gardabani": 310,
    "Tbilisi": 300,
    "Mtskheta": 295,
    "Kaspi": 280,
    "Gori": 260,
    "Khashuri": 240,
    "Kharagauli": 220,
    "Zestafoni": 200,
    "Kutaisi": 180,
    "Samtredia": 160,
    "Senaki": 120,
    "Zugdidi": 90,
    "Gali": 70,
    "Otchamtchire": 45,
    "Sokhumi": 0,
    "Gudauta": 30,
    "Gagra": 20
  }
}

    try:
        url = "http://localhost:5000/find_path"

        # Test A* algorithm
        astar_data = test_data.copy()
        astar_data["method"] = "astar"
        print("\n===== Testing A* Algorithm =====")
        print(f"Finding path from {astar_data['start']} to {astar_data['goal']}...")
        response = requests.post(url, json=astar_data)

        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\nA* Results:")
            print(f"Path: {result['path']}")
            print(f"Distance: {result['distance']} km")
            print(f"Time Complexity: {result['time_complexity']}")
            print(f"Space Complexity: {result['space_complexity']}")
            print(f"Visited Nodes: {result['visited_nodes']}")
            print(f"Execution Time: {result['execution_time_seconds']:.6f} seconds")
        else:
            print(f"Error: {response.text}")

        # Test IDDFS algorithm
        iddfs_data = test_data.copy()
        iddfs_data["method"] = "iddfs"
        iddfs_data["max_depth"] = 20  # Setting a reasonable max depth

        print("\n===== Testing IDDFS Algorithm =====")
        print(f"Finding path from {iddfs_data['start']} to {iddfs_data['goal']}...")
        response = requests.post(url, json=iddfs_data)

        print(f"Status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\nIDDFS Results:")
            print(f"Path: {result['path']}")
            print(f"Distance: {result['distance']} km")
            print(f"Time Complexity: {result['time_complexity']}")
            print(f"Space Complexity: {result['space_complexity']}")
            print(f"Visited Nodes: {result['visited_nodes']}")
            print(f"Path Found: {result.get('path_found', 'Not Available')}")
            print(f"Max Depth Used: {result.get('max_depth_used', 'Not Available')}")
            print(f"Execution Time: {result['execution_time_seconds']:.6f} seconds")
        else:
            print(f"Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {str(e)}")


def test_simple_graph():
    """Test with a simpler graph to verify both algorithms."""

    simple_graph = {
        "A": {"B": 5, "C": 3},
        "B": {"D": 2, "E": 4},
        "C": {"F": 6, "G": 7},
        "D": {"H": 1},
        "E": {"H": 3},
        "F": {"H": 5},
        "G": {"H": 4},
        "H": {}
    }

    heuristic = {
        "A": 8,
        "B": 6,
        "C": 7,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 3,
        "H": 0
    }

    test_data = {
        "graph": simple_graph,
        "start": "A",
        "goal": "H",
        "heuristic": heuristic
    }

    try:
        url = "http://localhost:5000/find_path"

        # Test A* on simple graph
        print("\n===== Testing A* on Simple Graph =====")
        test_data["method"] = "astar"
        response = requests.post(url, json=test_data)

        if response.status_code == 200:
            result = response.json()
            print("A* Results:")
            print(f"Path: {result['path']}")
            print(f"Distance: {result['distance']}")
            print(f"Visited Nodes: {result['visited_nodes']}")
            print(f"Execution Time: {result['execution_time_seconds']:.6f} seconds")

        # Test IDDFS on simple graph
        print("\n===== Testing IDDFS on Simple Graph =====")
        test_data["method"] = "iddfs"
        test_data["max_depth"] = 5
        response = requests.post(url, json=test_data)

        if response.status_code == 200:
            result = response.json()
            print("IDDFS Results:")
            print(f"Path: {result['path']}")
            print(f"Distance: {result['distance']}")
            print(f"Visited Nodes: {result['visited_nodes']}")
            print(f"Path Found: {result.get('path_found', 'Not Available')}")
            print(f"Execution Time: {result['execution_time_seconds']:.6f} seconds")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("===== Pathfinding Algorithms API Test =====")
    print("Note: Please ensure the API server is running before running this test.")

    test_api()
    test_simple_graph()