import os
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import math
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter
import random
from scipy.stats import linregress
import powerlaw
import matplotlib.pyplot as plt
import networkx as nx


def estimate_r_and_rho(G):
    colors = nx.get_node_attributes(G, "color")
    reds = [n for n, c in colors.items() if c == 'red']
    blues = [n for n, c in colors.items() if c == 'blue']
    r = len(reds) / G.number_of_nodes()

    same_red_edges = 0
    total_red_edges = 0
    same_blue_edges = 0
    total_blue_edges = 0

    for u, v in G.edges():
        if colors[u] == 'red':
            total_red_edges += 1
            if colors[v] == 'red':
                same_red_edges += 1
        if colors[u] == 'blue':
            total_blue_edges += 1
            if colors[v] == 'blue':
                same_blue_edges += 1

    rho_R = same_red_edges / total_red_edges if total_red_edges > 0 else 0.5
    rho_B = same_blue_edges / total_blue_edges if total_blue_edges > 0 else 0.5

    return r, rho_R, rho_B

def mixed_preferential_attachment(G_orig, m=3):
    n = G_orig.number_of_nodes()
    r, rho_R, rho_B = estimate_r_and_rho(G_orig)

    G = nx.DiGraph()
    G.add_node(0, color='red')
    G.add_node(1, color='blue')
    G.add_edge(0, 1)

    n0 = 2

    for new_node in range(n0, n):
        new_color = 'red' if random.random() < r else 'blue'
        G.add_node(new_node, color=new_color)

        targets = set()
        degrees = dict(G.degree())
        total_degree = sum(degrees.values())

        if total_degree == 0:
            break

        attempts = 0
        max_attempts = 1000

        while len(targets) < m and attempts < max_attempts:
            attempts += 1
            probs = [degrees[node] / total_degree for node in G.nodes()]
            candidate = random.choices(list(G.nodes()), weights=probs, k=1)[0]

            if candidate == new_node or candidate in targets:
                continue

            candidate_color = G.nodes[candidate]['color']
            if new_color == 'red':
                accept_prob = rho_R if candidate_color == 'red' else 1 - rho_R
            else:
                accept_prob = rho_B if candidate_color == 'blue' else 1 - rho_B

            if random.random() < accept_prob:
                targets.add(candidate)

        if len(targets) < m:
            print(f"Warning: Node {new_node} connected to only {len(targets)} targets instead of {m} after {attempts} attempts.")

        for target in targets:
            G.add_edge(new_node, target)

    return G

def build_original_graph():

    # path to the folder containing the files
    folder_path = "soc-sign-bitcoinotc.csv"  # or full path if needed

    G = nx.DiGraph()

    # Each file name represents a record (as you described)
    for filename in os.listdir(folder_path):
        try:
            parts = filename.strip().split(',')
            if len(parts) != 4:
                continue  # skip malformed names
            source = int(parts[0])
            target = int(parts[1])
            rating = int(parts[2])
            time = float(parts[3])
            G.add_edge(source, target, weight=rating, time=time)

        except Exception as e:
            print(f"Skipping file {filename} due to error: {e}")

    return G

def color_by_degree_threshold(G, K=2):
    for node in G.nodes():
        degree = G.degree(node)
        if degree > K:
            G.nodes[node]['color'] = 'red'
        else:
            G.nodes[node]['color'] = 'blue'
    return G

def draw_Graph(G, title="Bit Coin"):
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G, seed=42)

    # בונים רשימת צבעים לפי הצומת
    colors = [G.nodes[n].get('color', 'red') for n in G.nodes()]

    # סופרים את מספר הקודקודים מכל צבע
    color_counts = Counter(colors)
    num_red = color_counts.get('red', 0)
    num_blue = color_counts.get('blue', 0)

    # מציירים את הגרף
    ax = plt.gca()
    nx.draw(G, pos, with_labels=False, node_color=colors, edge_color='gray',
            node_size=20, width=1, ax=ax)

    # מוסיפים מקרא
    ax.scatter([], [], c='blue', label='>= 2')
    ax.scatter([], [], c='red', label='< 2')
    ax.legend(frameon=False, labelcolor='black', loc='upper right')

    # מוסיפים טקסט עם מספר הקודקודים מכל צבע בפינה השמאלית העליונה
    text_str = f"Red: {num_red}\nBlue: {num_blue}"
    plt.text(0.02, 0.95, text_str, transform=ax.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def build_max_connected_component_graph(G):
    strongly_components_G = list(nx.strongly_connected_components(G))
    print(f"Number of strongly connected components in the graph: {len(strongly_components_G)}")

    max_strong_component = max(strongly_components_G, key=len)
    max_strong_component_subgraph = G.subgraph(max_strong_component).copy()

    num_nodes = max_strong_component_subgraph.number_of_nodes()
    num_edges = max_strong_component_subgraph.number_of_edges()

    print("====== Max Strongly Connected Component Info ======")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print("===================================================")

    return max_strong_component_subgraph

def plot_degree_distribution_by_color(G, color, degree_type='in', title=' '):
    """
    Plot degree distribution (log-X) of nodes with a specific color in a directed graph.

    Parameters:
    - G: A directed NetworkX graph (nx.DiGraph)
    - color: Node color to filter by (must match value in node attribute 'color')
    - degree_type: 'in' or 'out'
    - title: Title to include in the plot
    """

    if degree_type not in {'in', 'out'}:
        raise ValueError("degree_type must be 'in' or 'out'")

    # סינון צמתים לפי צבע
    filtered_nodes = [n for n, data in G.nodes(data=True) if data.get('color') == color]

    if not filtered_nodes:
        print(f"No nodes with color '{color}' found.")
        return

    # חישוב דרגות לפי סוג הדרגה
    if degree_type == 'in':
        degrees = [G.in_degree(n) for n in filtered_nodes]
        label = 'In-Degree'
    else:  # 'out'
        degrees = [G.out_degree(n) for n in filtered_nodes]
        label = 'Out-Degree'

    # הכנה לבניית ההיסטוגרמה
    max_deg = max(degrees)
    bins = np.arange(1, max_deg + 2) - 0.5  # integer bins

    # שרטוט ההיסטוגרמה בלוגריתם של X
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=bins, color='salmon', edgecolor='black')
    plt.xscale('log')
    plt.xlabel(f'{label} (log scale)')
    plt.ylabel('Frequency')
    plt.title(f'{label} Distribution (Log X-axis) - Color: {color} {title}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def color_by_average_incoming_rating(G, threshold=2):
    for node in G.nodes():
        in_edges = G.in_edges(node, data=True)
        ratings = [data['weight'] for _, _, data in in_edges if 'weight' in data]

        if ratings:
            avg_rating = sum(ratings) / len(ratings)
        else:
            avg_rating = 0  # אם אין קשתות נכנסות, נניח ממוצע 0

        color = 'blue' if avg_rating >= threshold else 'red'
        G.nodes[node]['color'] = color

    return G

def draw_graph_by_fixed_colors(G, node_colors):
    plt.style.use('default')
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors,
        node_size=40, ax=ax
    )

    nx.draw_networkx_edges(
        G, pos, edge_color='darkblue',
        alpha=0.3, arrows=False, ax=ax
    )

    ax.set_title("Bitcoin OTC Trust Graph – Colored by Rating Ranges",
                 fontsize=14, color='black')
    ax.axis("off")

    # מקרא חדש
    legend_labels = {
        '#0000FF': '< -2',  # כחול כהה
        '#FF0000': '= 2',   # אדום
        '#FFFF00': '> 2'    # צהוב
    }

    for color, label in legend_labels.items():
        ax.scatter([], [], c=color, label=label, s=40)

    ax.legend(frameon=False, labelcolor='black')

    plt.tight_layout()
    plt.show()


G = build_original_graph()
max_connected_component_graph = build_max_connected_component_graph(G)

max_connected_component_graph = color_by_average_incoming_rating(max_connected_component_graph)
G_mpa = mixed_preferential_attachment(max_connected_component_graph, m=3)

draw_Graph( max_connected_component_graph, "Original Graph")
draw_Graph( G_mpa, "Mixed Preferential Attachment")

plot_degree_distribution_by_color(max_connected_component_graph, color='red', degree_type='in', title='Original Graph')
plot_degree_distribution_by_color(max_connected_component_graph, color='blue', degree_type='in', title='Original Graph')
plot_degree_distribution_by_color(max_connected_component_graph, color='red', degree_type='out', title='Original Graph')
plot_degree_distribution_by_color(max_connected_component_graph, color='blue', degree_type='out', title='Original Graph')

plot_degree_distribution_by_color(G_mpa, color='red', degree_type='in', title='Mixed Preferential Attachment')
plot_degree_distribution_by_color(G_mpa, color='blue', degree_type='in', title='Mixed Preferential Attachment')
plot_degree_distribution_by_color(G_mpa, color='red', degree_type='out', title='Mixed Preferential Attachment')
plot_degree_distribution_by_color(G_mpa, color='blue', degree_type='out', title='Mixed Preferential Attachment')


plot_degree_distribution_by_color(G_mpa, color='red', degree_type='in', title='G_mpa Graph')
plot_degree_distribution_by_color(G_mpa, color='blue', degree_type='out', title='G_mpa Graph')