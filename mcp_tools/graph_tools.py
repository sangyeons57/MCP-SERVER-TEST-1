import numpy as np
from typing import List, Callable

# Note: The @mcp.tool() decorator will be applied in main_mcp_server.py

def node_degree(adjacency_matrix: List[List[int]], node_index: int) -> int:
    """
    Calculates the degree of a specific node in an unweighted, undirected graph 
    represented by an adjacency matrix.
    The degree is the sum of connections (1s) for that node's row (or column).

    Args:
        adjacency_matrix: A list of lists representing the square adjacency matrix.
                          Matrix[i][j] = 1 if node i and j are connected, 0 otherwise.
                          Assumes symmetric matrix for an undirected graph.
        node_index: The 0-based index of the node for which to calculate the degree.
    Returns:
        The degree of the specified node.
    Raises:
        ValueError: If the matrix is not square, not a list of lists,
                    if node_index is out of bounds, or if matrix elements are not 0 or 1.
    """
    if not isinstance(adjacency_matrix, list) or not all(isinstance(row, list) for row in adjacency_matrix):
        raise ValueError("Adjacency matrix must be a list of lists.")
    
    num_nodes = len(adjacency_matrix)
    if num_nodes == 0:
        raise ValueError("Adjacency matrix cannot be empty.")
    if not all(len(row) == num_nodes for row in adjacency_matrix):
        raise ValueError("Adjacency matrix must be square.")
    
    if not (0 <= node_index < num_nodes):
        raise ValueError(f"Node index {node_index} is out of bounds for a graph with {num_nodes} nodes.")

    # Validate matrix elements (0 or 1 for unweighted graph)
    # Also check for symmetry for undirected graph assumption, though degree calc only uses one row.
    degree = 0
    for j in range(num_nodes):
        val = adjacency_matrix[node_index][j]
        if val not in (0, 1):
            raise ValueError(f"Adjacency matrix elements must be 0 or 1. Found {val} at ({node_index},{j}).")
        # For undirected, matrix[i][j] should be equal to matrix[j][i]. 
        # Not strictly needed for degree calculation of one node but good for matrix validation.
        # if adjacency_matrix[j][node_index] != val: 
        #     print(f"Warning: Adjacency matrix not symmetric at ({node_index},{j}) and ({j},{node_index}). Assuming undirected based on row.")
        degree += val
        
    # If self-loops (diagonal elements) are counted in degree, this is correct.
    # If they are not, one might subtract adjacency_matrix[node_index][node_index] if it was 1.
    # Standard definition usually includes self-loops in degree count from adjacency matrix sum.
    return degree

def graph_density(adjacency_matrix: List[List[int]], is_directed: bool = False) -> float:
    """
    Calculates the density of a graph represented by an adjacency matrix.
    Density = (Number of actual edges) / (Number of maximum possible edges).
    For an undirected graph, max edges = N*(N-1)/2.
    For a directed graph, max edges = N*(N-1).
    Assumes unweighted graph (elements are 0 or 1).

    Args:
        adjacency_matrix: A list of lists representing the square adjacency matrix.
        is_directed: False (default) if the graph is undirected, True if directed.
                     This affects how edges are counted and the max possible edges.
    Returns:
        The density of the graph (a float between 0 and 1).
    Raises:
        ValueError: If the matrix is not square, empty, or contains invalid elements.
    """
    if not isinstance(adjacency_matrix, list) or not all(isinstance(row, list) for row in adjacency_matrix):
        raise ValueError("Adjacency matrix must be a list of lists.")

    num_nodes = len(adjacency_matrix)
    if num_nodes == 0:
        return 0.0 # Or raise ValueError("Adjacency matrix cannot be empty for density calculation.")
    
    if not all(len(row) == num_nodes for row in adjacency_matrix):
        raise ValueError("Adjacency matrix must be square.")

    actual_edges = 0
    has_self_loops = False
    for i in range(num_nodes):
        for j in range(num_nodes):
            val = adjacency_matrix[i][j]
            if val not in (0, 1):
                raise ValueError(f"Adjacency matrix elements must be 0 or 1. Found {val} at ({i},{j}).")
            if i == j and val == 1:
                has_self_loops = True
            
            if is_directed:
                if val == 1:
                    actual_edges += 1
            else: # Undirected: count each edge once (e.g., from upper or lower triangle)
                if i < j and val == 1: # Only count upper triangle for undirected to avoid double counting
                    actual_edges += 1
                elif i == j and val == 1: # Count self-loops once
                    actual_edges += 1 # If self-loops contribute to edge count
    
    if num_nodes <= 1: # For a single node or no nodes, density can be considered 0 or undefined.
        return 0.0 # Avoid division by zero for N*(N-1)

    if is_directed:
        # Max edges in a directed graph (allowing self-loops if they were counted or not, N*N if allowed, N*(N-1) if not)
        # Standard N*(N-1) for directed simple graph (no self-loops for this formula part)
        # If we counted self-loops in actual_edges, should max_possible also consider them? Or define density for simple graphs.
        # For simple directed graph (no self-loops): N*(N-1)
        max_possible_edges = num_nodes * (num_nodes - 1)
        if max_possible_edges == 0 and num_nodes == 1: max_possible_edges = 1 # Avoid div by 0 for single node graph (density could be 0 or 1)

    else: # Undirected
        # Max edges in an undirected simple graph (no self-loops): N*(N-1)/2
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        if max_possible_edges == 0 and num_nodes == 1: max_possible_edges = 1 # Avoid div by 0

    if max_possible_edges == 0: # Should only happen if num_nodes is 0 or 1 and wasn't handled
        return 0.0 if actual_edges == 0 else 1.0 # Or handle as error, density is ill-defined

    # Refined edge counting for undirected graphs to ensure accuracy with self-loops
    if not is_directed:
        actual_edges = 0
        for r in range(num_nodes):
            for c in range(r, num_nodes): # Iterate r to N, c from r to N
                if adjacency_matrix[r][c] == 1:
                    if r == c: # Self-loop
                        actual_edges += 1
                    else: # Edge between distinct nodes
                        actual_edges += 1
    
    density = actual_edges / max_possible_edges
    return density

def get_graph_tools() -> List[Callable]:
    """Returns a list of all graph tool functions."""
    return [
        node_degree,
        graph_density
    ] 