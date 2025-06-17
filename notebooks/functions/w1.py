import networkx as nx
import random
from collections import deque
import time
from functools import wraps
import copy
import sys
import os

# Add the path to the project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from utils import mode_from_string  # or specific functions

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


@compute_time
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


@compute_time
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

@compute_time
def eg_(g, G):
    P = P_space_(g, G,
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


