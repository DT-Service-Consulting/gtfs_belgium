import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]  # points to gtfs_to_networks folder
# print("BASE_DIR:", BASE_DIR)

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from utils import *

path_to_sqlite = f"{BASE_DIR}/sqlite/belgium.sqlite"
g = load_gtfs(path_to_sqlite)

import pandas as pd
import numpy as np
import networkx as nx
from collections import deque
from functools import wraps
from IPython.display import display
import matplotlib.pyplot as plt
import copy
import os
import copy
import random
import time

def extract_directed_subgraph(G, target_size, min_edges=3, seed=None):
    if seed is not None:
        random.seed(seed)

    nodes = list(G.nodes())
    random.shuffle(nodes)
    seen_node_sets = set()

    for seed_node in nodes:
        visited = set([seed_node])
        queue = deque([seed_node])

        while queue and len(visited) < target_size:
            current = queue.popleft()
            neighbors = list(G.successors(current))
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                if len(visited) == target_size:
                    break

        if len(visited) == target_size:
            node_tuple = tuple(sorted(visited))
            if node_tuple in seen_node_sets:
                continue

            subG = G.subgraph(visited).copy()
            if subG.number_of_edges() >= min_edges:
                seen_node_sets.add(node_tuple)
                yield subG

def generate_subgraph_batches(G, sizes=(5, 10, 15), num_per_size=10, seed=42, min_edges=3):
    all_subgraphs = {size: [] for size in sizes}
    rng = random.Random(seed)

    for size in sizes:
        count = 0
        attempt = 0
        while count < num_per_size and attempt < 1000:
            sub_seed = rng.randint(0, 100000)
            for subG in extract_directed_subgraph(G, size, min_edges, seed=sub_seed):
                all_subgraphs[size].append(subG)
                count += 1
                break
            attempt += 1

        if count < num_per_size:
            print(f"Warning: Only found {count} subgraphs of size {size} after {attempt} attempts.")
    
    return all_subgraphs

def compute_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Function '{func.__name__}' completed.")
        print(f"Execution time: {end_time - start_time:.2f} seconds\n")
        return result
    return wrapper


def get_all_GTC_refactored_(L_space, P_space, k, wait_pen, transfer_pen):
    import networkx as nx

    # Precompute all attributes
    P_veh = nx.get_edge_attributes(P_space, "veh")
    P_wait = nx.get_edge_attributes(P_space, "avg_wait")
    L_dur = nx.get_edge_attributes(L_space, "duration_avg")
    L_dist = nx.get_edge_attributes(L_space, "d")

    # Precompute route directions as sets to avoid redundant set conversions
    routes_dirs = {}
    for e in P_veh:
        routes_dirs[e] = set()
        for ro in P_veh[e]:
            for dr in P_veh[e][ro]:
                routes_dirs[e].add(str(ro) + str(dr))

    # Compute all shortest paths using Dijkstra's algorithm
    paths = dict(nx.all_pairs_dijkstra_path(L_space, weight="duration_avg"))
    shortest_paths = {}

    for n1 in L_space.nodes:
        for target in L_space.nodes:
            if n1 == target:
                continue

            if n1 not in shortest_paths:
                shortest_paths[n1] = {}

            tt_paths = []
            only_tts = []

            # We consider just one path
            if target in paths[n1]:
                k_paths = [paths[n1][target]]
            else:
                k_paths = []

            for p in k_paths:
                possible_routes = routes_dirs.get((p[0], p[1]), set()).copy()

                dist = 0
                tt = 0
                wait = 0
                tf = 0
                t_stations = [n1]

                for l1, l2 in zip(p, p[1:]):
                    tt += L_dur[(l1, l2)]
                    dist += L_dist[(l1, l2)]

                    routes = routes_dirs.get((l1, l2), set())
                    possible_routes.intersection_update(routes)

                    if not possible_routes:
                        possible_routes = routes.copy()
                        tf += 1
                        t_stations.append(l1)

                t_stations.append(target)
                tt = round(tt / 60)

                for t1, t2 in zip(t_stations, t_stations[1:]):
                    wait += P_wait[(t1, t2)]

                wait = round(wait)
                transfer_cost = sum([transfer_pen[i] if i < len(transfer_pen) else transfer_pen[-1] for i in range(tf)])
                total_tt = tt + wait * wait_pen + transfer_cost

                only_tts.append(total_tt)
                tt_paths.append({
                    'path': p,
                    'GTC': total_tt,
                    'in_vehicle': tt,
                    'waiting_time': wait,
                    'n_transfers': tf,
                    'traveled_distance': dist
                })

            if k_paths:
                min_path_tt = min(only_tts)
                min_path = tt_paths[only_tts.index(min_path_tt)]
                shortest_paths[n1][target] = min_path
            else:
                shortest_paths[n1][target] = []

    return shortest_paths

def P_space_(L, mode, start_hour=5, end_hour=24, dir_indicator=None):
    '''
    Create P-space graph given
    L: L-space
    Optional:
        start_hour: start hour considered when building L-space. Defaults to 5 am
        end_hour: end hour considered when building L-space. Defaults to midnight.
        dir_indicator: override which indicator direction_id, headsign, or shape_id should be used.
    '''

    # Validate inputs
    if not (0 <= start_hour < end_hour <= 24):
        raise AssertionError("Start/end hour must be in [0, 24] and start < end")
    if not (isinstance(start_hour, int) and isinstance(end_hour, int)):
        raise AssertionError("Start/end hours must be integers")

    time = end_hour - start_hour

    backup_colors = [
        '0000FF', '008000', 'FF0000', '00FFFF', 'FF00FF', 'FFFF00', '800080', 'FFC0CB', 'A52A2A',
        'FFA500', 'FF7F50', 'ADD8E6', '00FF00', 'E6E6FA', '40E0D0', 
        '006400', 'D2B48C', 'FA8072', 'FFD700'
    ]

    # Prepare graph and data
    P_G = nx.DiGraph()
    P_G.add_nodes_from(L.nodes(data=True))

    location = g.get_location_name()
    mode_val = mode_from_string(mode)
    routes = get_routes_for_mode(g, mode)

    colors = get_color_per_route(g, routes)
    L_edges = list(L.edges(data=True))

    # Precompute final route-to-color mapping
    route_colors = {}
    for i, r in enumerate(routes):
        c = colors.get(r)
        if not c or len(c) != 6:
            c = backup_colors[i % len(backup_colors)]
        route_colors[r] = '#' + c

    # Determine dir_indicator
    if not dir_indicator:
        dir_indicator = 'empty'
        if L_edges:
            sample_edge = L_edges[0][2]
            if sample_edge.get('direction_id'):
                dir_indicator = 'direction_id'
            elif sample_edge.get('headsign'):
                dir_indicator = 'headsign'
            elif sample_edge.get('shape_id'):
                dir_indicator = 'shape_id'

    # Main loop over routes
    for r_idx, r in enumerate(routes):
        color = route_colors[r]

        # Get all direction indicators for this route
        dirs = set()
        for _, _, edge_data in L_edges:
            if r in edge_data.get('route_I_counts', {}):
                for d in edge_data.get(dir_indicator, {}).keys():
                    dirs.add(d)

        # For each direction, build subgraph and add edges
        for d in dirs:
            sub = nx.DiGraph()
            sub_edges = []

            for a, b, edge_data in L_edges:
                if r in edge_data.get('route_I_counts', {}) and d in edge_data.get(dir_indicator, {}):
                    sub_edges.append((a, b, edge_data))

            if not sub_edges:
                continue

            sub.add_edges_from(sub_edges)

            for n1 in sub:
                try:
                    paths = nx.single_source_shortest_path(sub, n1)
                except nx.NetworkXError:
                    continue

                for n2, path in paths.items():
                    if n1 == n2 or len(path) < 2:
                        continue

                    path_set = set(path)

                    out_e = next(((a, b, c) for a, b, c in sub.out_edges(n1, data=True)
                                  if a in path_set and b in path_set), None)
                    in_e = next(((a, b, c) for a, b, c in sub.in_edges(n2, data=True)
                                 if a in path_set and b in path_set), None)

                    if not out_e or not in_e:
                        continue

                    veh_out = out_e[2]['route_I_counts'][r]
                    veh_in = in_e[2]['route_I_counts'][r]
                    veh = min(veh_out, veh_in)

                    veh_per_hour = veh / time
                    avg_wait = 60 / veh_per_hour / 2

                    if P_G.has_edge(n1, n2):
                        P_G[n1][n2]['edge_color'] = '#000000'
                        if r not in P_G[n1][n2]['veh']:
                            P_G[n1][n2]['veh'][r] = {d: veh_per_hour}
                        else:
                            P_G[n1][n2]['veh'][r][d] = veh_per_hour

                        tot_veh = sum(
                            v for route_data in P_G[n1][n2]['veh'].values()
                            for v in route_data.values()
                        )
                        P_G[n1][n2]['avg_wait'] = 60 / tot_veh / 2
                    else:
                        P_G.add_edge(n1, n2, veh={r: {d: veh_per_hour}},
                                     avg_wait=avg_wait, edge_color=color)

    return P_G

def eg_(L):
    P = P_space_(L,
                start_hour=5,
                end_hour=24,
                mode="Rail")

    sp = get_all_GTC_refactored_(L, P, 3, 2, [5])
    
    eg = 0
    for n1 in sorted(L.nodes()):
        for n2 in sorted(L.nodes()):
            if n1 != n2:
                if sp[n1][n2]:
                    tt = sp[n1][n2]["GTC"]
                    eg += 1 / tt

    return eg / (L.number_of_nodes() * (L.number_of_nodes() - 1))
    

def random_node_removal(G, num_to_remove, seed=None, verbose=False):
    """
    Removes nodes from the graph in a random order and tracks the impact on global efficiency.

    Parameters:
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes to remove from the graph.
        seed (int, optional): Seed for reproducible random node selection.
        verbose (bool): Whether to print detailed logs during execution.

    Returns:
        original_efficiency (float): The initial global efficiency before any removals.
        efficiencies (list of float): Normalized global efficiencies after each removal.
        num_removed (list of int): Step count corresponding to each node removal.
        removed_nodes (list of node): List of nodes removed in the order of removal.
        removal_times (list of float): Time taken (in seconds) for each removal step.
    """
    if seed is not None:
        random.seed(seed)
    removal_nodes = random.sample(list(G.nodes()), num_to_remove)

    if verbose:
        print(f"Random removal order: {removal_nodes}")

    original_efficiency = eg_(G)
    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    for i, node in enumerate(removal_nodes):
        start_time = time.perf_counter()

        G.remove_node(node)
        removed_nodes.append(node)

        try:
            eff = eg_(G)
        except Exception as e:
            if verbose:
                print(f"Error after {i + 1} removals: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(i + 1)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Removed {node} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def targeted_node_removal(G, num_to_remove, verbose=False):
    """
    Removes nodes from the graph using a greedy strategy that selects the node whose removal
    results in the largest drop in global efficiency at each step.

    Parameters:
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes to remove from the graph.
        verbose (bool): Whether to print detailed logs including time per step.

    Returns:
        original_efficiency (float): The initial global efficiency before any removals.
        efficiencies (list of float): Normalized global efficiencies after each removal.
        num_removed (list of int): Step count corresponding to each node removal.
        removed_nodes (list of node): List of nodes removed in the order of removal.
        removal_times (list of float): Time taken (in seconds) for each full removal step,
                                       including node evaluation and removal.
    """
    original_efficiency = eg_(G)
    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    for step in range(num_to_remove):
        start_time = time.perf_counter()

        current_eff = eg_(G)
        max_drop = -1
        best_node = None

        for node in G.nodes():
            temp_G = G.copy()
            temp_G.remove_node(node)

            try:
                eff = eg_(temp_G)
            except:
                continue

            drop = current_eff - eff
            if drop > max_drop:
                max_drop = drop
                best_node = node

        if best_node is None:
            if verbose:
                print("No valid node to remove.")
            break

        G.remove_node(best_node)
        removed_nodes.append(best_node)

        try:
            eff = eg_(G)
        except Exception as e:
            if verbose:
                print(f"Error after {step + 1} removals: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(step + 1)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Step {step + 1}: Removed {best_node} → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def betweenness_node_removal(G, num_to_remove, verbose=False):
    """
    Removes nodes from the graph in descending order of weighted betweenness centrality
    and tracks the impact on global efficiency.

    Parameters:
        G (networkx.Graph): The input graph to modify (passed by reference).
        num_to_remove (int): Number of nodes to remove from the graph.
        verbose (bool): Whether to print detailed logs during execution.

    Returns:
        original_efficiency (float): The initial global efficiency before any removals.
        efficiencies (list of float): Normalized global efficiencies after each removal.
        num_removed (list of int): Step count corresponding to each node removal.
        removed_nodes (list of node): List of nodes removed in the order of removal.
        removal_times (list of float): Time taken (in seconds) for each removal step.
    """
    original_efficiency = eg_(G)
    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    for step in range(num_to_remove):
        start_time = time.perf_counter()

        # Compute weighted betweenness centrality
        try:
            centrality = nx.betweenness_centrality(G, weight='duration_avg')
        except Exception as e:
            if verbose:
                print(f"Failed to compute betweenness at step {step + 1}: {e}")
            break

        if not centrality:
            if verbose:
                print("No centrality values computed. Possibly empty graph.")
            break

        # Select node with highest centrality
        node_to_remove = max(centrality, key=centrality.get)

        G.remove_node(node_to_remove)
        removed_nodes.append(node_to_remove)

        try:
            eff = eg_(G)
        except Exception as e:
            if verbose:
                print(f"Error after removing {node_to_remove}: {e}")
            break

        elapsed = time.perf_counter() - start_time
        normalized_eff = eff / original_efficiency

        efficiencies.append(normalized_eff)
        num_removed.append(step + 1)
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Step {step + 1}: Removed {node_to_remove} (Centrality: {centrality[node_to_remove]:.4f})")
            print(f"Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time taken: {elapsed:.4f} seconds\n")

    return original_efficiency, efficiencies, num_removed, removed_nodes, removal_times


def simulate_fixed_node_removal_efficiency(
    L_graph,
    num_to_remove=None,
    pct_to_remove=None,  # priority over num_to_remove
    method='random', # random or targeted or betweenness
    seed=None,
    verbose=False
):
    """
    Simulates the impact of fixed sequential node removals on the global efficiency of a graph.
    
    Parameters:
        L_graph (networkx.Graph): The subgraph from which nodes will be removed.
        num_to_remove (int, optional): Number of nodes to remove. Ignored if percentage is given.
        pct_to_remove (int, optional): Percentage of nodes to remove (between 1 and 100).
        seed (int, optional): Random seed for node selection.
        verbose (bool): Whether to print progress and debug information.
    """
    G = copy.deepcopy(L_graph)
    total_nodes = G.number_of_nodes()

    if pct_to_remove is not None:
        if not (1 <= pct_to_remove <= 100):
            raise ValueError("Percentage must be an integer between 1 and 100.")
        num_to_remove = int(total_nodes * (pct_to_remove / 100))
    elif num_to_remove is None:
        raise ValueError("You must specify either num_to_remove or percentage.")

    if method == "random":
        return random_node_removal(G, num_to_remove, seed, verbose)
    elif method == "targeted":
        return targeted_node_removal(G, num_to_remove, verbose)
    elif method == "betweenness":
        return betweenness_node_removal(G, num_to_remove, verbose)
    else:
        raise ValueError("Invalid method. Choose 'random' or 'targeted'.")


def plot_efficiency_results(num_removed, efficiencies, title="Impact of Node Removal on Network Efficiency (Normalized)"):
    """
    Plots the change in normalized efficiency as nodes are removed.

    Parameters:
    - num_removed: List of number of nodes removed
    - efficiencies: Corresponding list of normalized efficiencies
    - title: Plot title
    """
    plt.figure(figsize=(6, 4))
    plt.plot(num_removed, efficiencies, marker='o')
    plt.xlabel("Number of Nodes Removed")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def run_removal_simulations(subgraphs_by_size, num_to_remove=None, pct_to_remove=None, method='random', seed=42, verbose=False):
    """
    Run node removal simulations across all subgraphs grouped by size and collect efficiency and timing metrics.

    Parameters:
        g (networkx.Graph): The original graph used to compute baseline efficiency.
        subgraphs_by_size (dict): A dictionary where each key is a subgraph size and each value is a list of subgraphs (networkx.Graph).
        num_to_remove (int): Number of nodes to remove from each subgraph. Default is 5.
        seed (int): Random seed for reproducibility. Default is 42.
        verbose (bool): Whether to print detailed output during simulation. Default is False.

    Returns:
        pd.DataFrame: A DataFrame where each row corresponds to one subgraph simulation and contains:
            - graph_index: Index of the subgraph within its group
            - num_nodes: Number of nodes in the subgraph
            - num_edges: Number of edges in the subgraph
            - runtime_seconds: Total time taken for the simulation
            - original_efficiency: Efficiency before any node removal
            - final_efficiency: Efficiency after all removals
            - efficiency_after_each_removal: List of normalized efficiencies after each removal (excluding original)
            - removed_nodes: List of removed node IDs
            - removal_times: List of cumulative times after each removal
            - eff_after_{i}: Normalized efficiency after i-th removal, where i=0 is the original
    """
    results = []

    for size, graphs in subgraphs_by_size.items():
        for idx, L in enumerate(graphs):
            start = time.perf_counter()
            try:
                original_efficiency, efficiencies, num_removed, removed_nodes, removal_times = simulate_fixed_node_removal_efficiency(
                    L_graph=L,
                    num_to_remove=num_to_remove,
                    pct_to_remove=pct_to_remove, # priority over num_to_remove
                    method=method, # random or targeted
                    seed=seed,
                    verbose=verbose
                )
            except Exception as e:
                print(f"Error on graph size {size}, index {idx}: {e}")
                continue
            end = time.perf_counter()
            elapsed = end - start

            result = {
                "graph_index": idx,
                "num_nodes": L.number_of_nodes(),
                "num_edges": L.number_of_edges(),
                "runtime_seconds": round(elapsed, 3),
                "original_efficiency": original_efficiency,
                "final_efficiency": efficiencies[-1] if efficiencies else None,
                "efficiency_after_each_removal": efficiencies[0:] if len(efficiencies) > 1 else [],
                "removed_nodes": removed_nodes,
                "removal_times": removal_times
            }

            for i, eff in enumerate(efficiencies):
                result[f"eff_after_{i}"] = eff

            results.append(result)

    return pd.DataFrame(results)



def plot_efficiency_results_from_batch(row):
    """
    Plot the efficiency drop across node removals for a single subgraph.

    Parameters:
    row (pd.Series): A row from the DataFrame containing the following keys:
        - 'original_efficiency': efficiency before any node removal
        - 'efficiency_after_each_removal': list of efficiency values after each node is removed
        - 'num_nodes': number of nodes in the subgraph
        - 'graph_index': index of the subgraph within its group

    The function combines the original efficiency with the efficiency after each removal,
    and plots them as a line chart with points for visual tracking of efficiency drop.
    """
    # Full efficiency list: original + after each removal
    all_efficiencies = row['efficiency_after_each_removal']
    num_removed = list(range(1, len(all_efficiencies) + 1))
    plot_efficiency_results(num_removed, all_efficiencies)


def compute_avg_runtime_by_num_nodes(df_results):
    """
    Compute the average runtime and average number of nodes removed for subgraphs grouped by number of nodes.

    Parameters:
        df_results (pd.DataFrame): DataFrame with columns:
            - 'num_nodes': int, number of nodes in the subgraph
            - 'runtime_seconds': float, total runtime for removals on the subgraph
            - 'removed_nodes': list, nodes removed from the subgraph

    Returns:
        pd.DataFrame: DataFrame with columns:
            - 'num_nodes': number of nodes in each subgraph
            - 'num_nodes_removed': average number of nodes removed
            - 'avg_runtime_seconds': average runtime (in seconds) for graphs of that size
    """
    # Add a column for number of removed nodes
    df_results["num_nodes_removed"] = df_results["removed_nodes"].apply(len)

    # Group by 'num_nodes' and compute average runtime and average number of nodes removed
    grouped = df_results.groupby("num_nodes").agg(
        num_nodes_removed=("num_nodes_removed", "mean"),
        avg_runtime_seconds=("runtime_seconds", "mean")
    ).reset_index()

    grouped = grouped[["num_nodes", "num_nodes_removed", "avg_runtime_seconds"]]

    return grouped



def plot_removal_time_vs_steps(row):
    """
    Plot cumulative runtime and individual removal times against number of node removals for a single subgraph,
    with two side-by-side subplots. Also displays a table of removed nodes and corresponding removal times.
    
    Parameters:
    row (pd.Series): Row from df_results containing 'removal_times' and 'removed_nodes'.
    """
    if "removal_times" not in row or not row["removal_times"]:
        print("No timing data available for this row.")
        return

    individual_times = row["removal_times"]
    cumulative_times = np.cumsum(individual_times)
    steps = list(range(1, len(individual_times) + 1))
    
    # Display tabular data
    df = pd.DataFrame({
        "Node Removed": row["removed_nodes"],
        "Time Elapsed (s)": individual_times
    })
    display(df)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: time line plot
    ax1.plot(steps, individual_times, marker='o', color='b')
    ax1.set_title(f"Removal Time\nGraph Size {row['num_nodes']} Index {row['graph_index']}")
    ax1.set_xlabel("Node Removal Step")
    ax1.set_ylabel("Time per Removal (seconds)")
    ax1.grid(True)

    # Right: individual removal time bar plot
    ax2.bar(steps, individual_times, color='orange', alpha=0.7)
    ax2.set_title("Individual Removal Time per Node")
    ax2.set_xlabel("Node Removal Step")
    ax2.set_ylabel("Time per Removal (seconds)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()