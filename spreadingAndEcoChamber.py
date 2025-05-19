import networkx as nx
import matplotlib.pyplot as plt
import random
import copy

# Constants
P_Q_SETTINGS = {
    "strong_homophily": (1.0, 0.0),
    "p_homophily": (0.7, 0.3)
}

REGULATION_TYPES = ["no_regulation", "rlr", "blue_only"]
RHO_VALUES = [0.25, 0.5]

# Colors
RED, BLUE = "red", "blue"

# Set random seed for reproducibility
random.seed(42)

# Placeholder: assumes you already have a function that returns the base graph
def get_base_graph():
    G = nx.erdos_renyi_graph(100, 0.05, directed=True)
    while not nx.is_weakly_connected(G):
        G = nx.erdos_renyi_graph(100, 0.05, directed=True)
    for node in G.nodes:
        G.nodes[node]['color'] = RED if random.random() < 0.5 else BLUE
    return G

# Apply homophily filter to neighbors
def get_neighbors_with_homophily(G, node, p, q):
    neighbors = list(G.successors(node))
    color = G.nodes[node]['color']
    filtered = []
    for neighbor in neighbors:
        neighbor_color = G.nodes[neighbor]['color']
        prob = p if color == neighbor_color else q
        if random.random() < prob:
            filtered.append(neighbor)
    return filtered

# Apply RLR regulation by rewiring a portion of edges randomly
def apply_rlr(G, rho):
    G_new = copy.deepcopy(G)
    edges = list(G_new.edges())
    num_to_rewire = int(rho * len(edges))
    to_rewire = random.sample(edges, num_to_rewire)
    nodes = list(G_new.nodes())
    for u, v in to_rewire:
        G_new.remove_edge(u, v)
        new_v = random.choice(nodes)
        while new_v == u or G_new.has_edge(u, new_v):
            new_v = random.choice(nodes)
        G_new.add_edge(u, new_v)
    return G_new

# Apply BLUE-only regulation
def apply_blue_only(G, rho):
    G_new = copy.deepcopy(G)
    red_nodes = [n for n in G_new.nodes if G_new.nodes[n]['color'] == RED]
    for node in red_nodes:
        if random.random() < rho:
            G_new.remove_edges_from(list(G_new.out_edges(node)))
    return G_new

# Updated spreading simulation to accept multiple sources
def simulate_spreading(G, source_nodes, p, q):
    visited = set(source_nodes)
    queue = list(source_nodes)
    for node in source_nodes:
        print(f"Source node {node} color: {G.nodes[node]['color']}, neighbors: {[G.nodes[n]['color'] for n in G.successors(node)]}")
    while queue:
        current = queue.pop(0)
        for neighbor in get_neighbors_with_homophily(G, current, p, q):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    red_count = sum(1 for node in visited if G.nodes[node]['color'] == RED)
    blue_count = sum(1 for node in visited if G.nodes[node]['color'] == BLUE)
    return red_count, blue_count

# Wrapper for a full simulation scenario
def run_simulation(G, regulation, homophily_type, target_size, rho=None):
    p, q = P_Q_SETTINGS[homophily_type]

    if regulation == "no_regulation":
        G_sim = G
    elif regulation == "rlr":
        G_sim = apply_rlr(G, rho)
    elif regulation == "blue_only":
        G_sim = apply_blue_only(G, rho)
    else:
        raise ValueError("Invalid regulation type")

    source_nodes = random.sample(list(G_sim.nodes()), target_size)
    return simulate_spreading(G_sim, source_nodes, p, q)

# Run all experiments and collect results
def run_all_experiments():
    G = get_base_graph()
    results = {}
    target_sizes = [1, 2, 5, 10, 20]

    for homophily_type in P_Q_SETTINGS:
        for regulation in REGULATION_TYPES:
            if regulation == "no_regulation":
                for t_size in target_sizes:
                    label = f"{homophily_type} {regulation}"
                    red, blue = run_simulation(G, regulation, homophily_type, t_size)
                    results.setdefault(label, []).append((t_size, red, blue))
            else:
                for rho in RHO_VALUES:
                    for t_size in target_sizes:
                        label = f"{homophily_type} {regulation}({rho})"
                        red, blue = run_simulation(G, regulation, homophily_type, t_size, rho)
                        results.setdefault(label, []).append((t_size, red, blue))
    return results

# Plotting all curves in a single line plot
def plot_results(results):
    plt.figure(figsize=(16, 10))
    for label, values in results.items():
        x = [v[0] for v in values]  # target set size
        red_y = [v[1] for v in values]
        blue_y = [v[2] for v in values]

        plt.plot(x, red_y, marker='o', label=f'{label} - Red')
        plt.plot(x, blue_y, marker='x', label=f'{label} - Blue')

    plt.xlabel("Target Set Size")
    plt.ylabel("Active Set Size")
    plt.title("Spreading Influence: Active Set vs. Target Set Size")
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Run everything
if __name__ == "__main__":
    results = run_all_experiments()
    plot_results(results)