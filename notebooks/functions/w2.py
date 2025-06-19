import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[2]  # points to gtfs_to_networks folder
# print("BASE_DIR:", BASE_DIR)

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))


from utils import *

import networkx as nx
import random
from collections import deque
import time
from functools import wraps
import copy
import os
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        print(f"Function '{func.__name__}'")
        print(f"Execuion time: {end_time - start_time:.2f} seconds\n")
        return result
    return wrapper

def get_random_removal_nodes(graph, num_to_remove, seed=None):
    """
    Returns a list of nodes randomly selected from G for removal.

    Parameters:
    - G: NetworkX graph
    - num_to_remove: Number of nodes to remove (int)
    - seed: Optional random seed for reproducibility (int or None)

    Returns:
    - List of node IDs selected for removal
    """
    if num_to_remove > graph.number_of_nodes() - 2:
        raise ValueError("Cannot remove all or almost all nodes. Reduce 'num_to_remove'.")

    if seed is not None:
        random.seed(seed)

    return random.sample(list(graph.nodes()), num_to_remove)

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

def P_space_(g, L, mode, start_hour=5, end_hour=24, dir_indicator=None):
    '''
    Create P-space graph given:
    g: gtfs feed
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

def eg_(g, L):
    P = P_space_(g, L,
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

import copy
import time

def simulate_fixed_node_removal_efficiency(
    g,
    L_graph,
    num_to_remove,
    seed=None,
    verbose=True
):
    """
    Simulates the impact of fixed sequential node removals on the global efficiency of a graph.
    
    Parameters:
        g (networkx.Graph): The original graph used as a reference.
        L_graph (networkx.Graph): The subgraph from which nodes will be removed.
        num_to_remove (int): Number of nodes to remove.
        seed (int, optional): Random seed for node selection.
        verbose (bool): Whether to print progress and debug information.
    
    Returns:
        efficiencies (list): Normalized global efficiency after each removal.
        num_removed (list): Step count corresponding to each removal.
        removed_nodes (list): The exact nodes removed in order.
        removal_times (list): Cumulative time (in seconds) taken after each removal.
    """
    G = copy.deepcopy(L_graph)
    original_efficiency = eg_(g, G)
    
    if verbose:
        print(f"Original efficiency: {original_efficiency:.4f}")

    removal_nodes = get_random_removal_nodes(G, num_to_remove, seed)
    
    if verbose:
        print(f"Random node removal order (seed={seed}): {removal_nodes}")

    efficiencies = []
    num_removed = []
    removed_nodes = []
    removal_times = []

    for i, node_to_remove in enumerate(removal_nodes):
        if node_to_remove not in G:
            if verbose:
                print(f"Node {node_to_remove} not in graph, skipping.")
            continue

        if verbose:
            print(f"\nIteration {i+1}: Removing node → {node_to_remove}")

        start_time = time.perf_counter()

        G.remove_node(node_to_remove)
        removed_nodes.append(node_to_remove)

        try:
            eff = eg_(g, G)
        except Exception as e:
            if verbose:
                print(f"Error after removing {i+1} nodes: {e}")
            break

        normalized_eff = eff / original_efficiency
        efficiencies.append(normalized_eff)
        num_removed.append(i + 1)
        elapsed = time.perf_counter() - start_time
        removal_times.append(round(elapsed, 4))

        if verbose:
            print(f"Removed {i+1} node(s) → Normalized Efficiency: {normalized_eff:.4f}")
            print(f"Time elapsed: {elapsed:.4f} seconds\n")

    return efficiencies, num_removed, removed_nodes, removal_times



def plot_efficiency_results(num_removed, efficiencies, title="Impact of Node Removal on Network Efficiency (Normalized)"):
    """
    Plots the change in normalized efficiency as nodes are removed.

    Parameters:
    - num_removed: List of number of nodes removed
    - efficiencies: Corresponding list of normalized efficiencies
    - title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(num_removed, efficiencies, marker='o')
    plt.xlabel("Number of Nodes Removed")
    plt.ylabel("Normalized Efficiency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import time
import pandas as pd

def run_removal_simulations(g, subgraphs_by_size, num_to_remove=5, seed=42, verbose=False):
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
                efficiencies, num_removed, removed_nodes, removal_times = simulate_fixed_node_removal_efficiency(
                    g,
                    L_graph=L,
                    num_to_remove=num_to_remove,
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
                "original_efficiency": efficiencies[0] if efficiencies else None,
                "final_efficiency": efficiencies[-1] if efficiencies else None,
                "efficiency_after_each_removal": efficiencies[1:] if len(efficiencies) > 1 else [],
                "removed_nodes": removed_nodes,
                "removal_times": removal_times
            }

            for i, eff in enumerate(efficiencies):
                result[f"eff_after_{i}"] = eff

            results.append(result)

    return pd.DataFrame(results)



def plot_single_efficiency(row):
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
    all_efficiencies = [row['original_efficiency']] + row['efficiency_after_each_removal']
    
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(all_efficiencies)), all_efficiencies, marker='o')
    plt.title(f"Efficiency Drop – Graph Size {row['num_nodes']} Index {row['graph_index']}")
    plt.xlabel("Nodes Removed")
    plt.ylabel("Efficiency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_avg_runtime_by_num_nodes(df_results):
    """
    Compute the average runtime for subgraphs grouped by number of nodes.

    Parameters:
    df_results (pd.DataFrame): DataFrame containing at least 'num_nodes' and 'runtime_seconds' columns.

    Returns:
    pd.DataFrame: A DataFrame with two columns:
        - 'num_nodes': number of nodes in each subgraph
        - 'runtime_seconds': average runtime (in seconds) to process graphs of that size
    """
    return (
        df_results.groupby("num_nodes")["runtime_seconds"]
        .mean()
        .reset_index()
        .sort_values("num_nodes")
    )


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

    cumulative_times = row["removal_times"]
    steps = list(range(1, len(cumulative_times) + 1))
    individual_times = np.diff([0] + cumulative_times)

    # Display tabular data
    df = pd.DataFrame({
        "Node Removed": row["removed_nodes"],
        "Time Elapsed (s)": individual_times
    })
    print("\nNode Removal Details:\n", df)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: cumulative time line plot
    ax1.plot(steps, cumulative_times, marker='o', color='b')
    ax1.set_title(f"Cumulative Removal Time\nGraph Size {row['num_nodes']} Index {row['graph_index']}")
    ax1.set_xlabel("Number of Nodes Removed")
    ax1.set_ylabel("Cumulative Time (seconds)")
    ax1.grid(True)

    # Right: individual removal time bar plot
    ax2.bar(steps, individual_times, color='orange', alpha=0.7)
    ax2.set_title("Individual Removal Time per Node")
    ax2.set_xlabel("Node Removal Step")
    ax2.set_ylabel("Time per Removal (seconds)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()