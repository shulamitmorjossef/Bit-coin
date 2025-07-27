import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import math
import numpy as np
import random
import powerlaw
from scipy.stats import linregress
from collections import defaultdict
from networkx.algorithms.community import greedy_modularity_communities, modularity
import warnings

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

def draw_graph(G):
    def compute_average_rating(G):

        # Compute average incoming rating per node
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = 0
        return node_avg_rating

    def compute_fixed_colors_by_ranges(G, node_avg_rating):
        colors = []
        for node in G.nodes():
            avg = node_avg_rating.get(node, 0)
            if avg <= -2:
                colors.append('#FF0000')  # red
            elif avg > -2 and avg < 2:
                colors.append('#FFFF00')  # yellow
            else:
                colors.append('#0000FF')  # blue
        return colors

    node_colors = compute_fixed_colors_by_ranges(G, compute_average_rating(G))
    plt.style.use('default')
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    nx.draw_networkx_nodes(
        G, pos, node_color = node_colors,
        node_size=40, ax=ax
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color='black',
        alpha=0.3,
        arrows=True,
        arrowsize=10,
        width=0.8,
        ax=ax
    )

    ax.set_title("Bitcoin OTC Trust Graph – Colored by Rating Ranges",
                 fontsize=14, color='black')
    ax.axis("off")

    # מקרא חדש
    legend_labels = {
        '#FF0000': '<= -2',  # red
        '#FFFF00': '-2 < and < 2 and ',   # yellow
        '#0000FF': '>= 2'    # blue
    }

    for color, label in legend_labels.items():
        ax.scatter([], [], c=color, label=label, s=40)

    ax.legend(frameon=False, labelcolor='black')

    plt.tight_layout()
    plt.show()

def min_max_rating(G):

    def compute_average_rating(G):

        # Compute average incoming rating per node
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = 0
        return node_avg_rating

    node_avg_rating = compute_average_rating(G)

    # Normalize ratings for color mapping
    min_rating = min(node_avg_rating.values())
    max_rating = max(node_avg_rating.values())

    return min_rating, max_rating

def small_world(G):
    # Assuming G is already a directed graph (DiGraph)
    print("\n--- Small-World Property Check (Directed Graph Only) ---")

    # 1. Average Shortest Path Length (among reachable pairs)
    print("\nCalculating all shortest path lengths...")
    path_lengths = []

    for source in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, source)
        for target, dist in lengths.items():
            if source != target:
                path_lengths.append(dist)

    if path_lengths:
        diameter = max(path_lengths)
        avg_path_length = sum(path_lengths) / len(path_lengths)
        print(f" Reachable pairs: {len(path_lengths)}")
        print(f" Diameter (max shortest path): {diameter}")
        print(f" Average shortest path length: {avg_path_length:.4f}")
    else:
        print(" No reachable pairs found. Cannot calculate path metrics.")

    # 2. Reciprocity
    print("\nCalculating reciprocity...")
    reciprocity = nx.reciprocity(G)
    if reciprocity is not None:
        print(f" Reciprocity (rate of mutual edges): {reciprocity:.4f}")
    else:
        print(" Could not calculate reciprocity (graph might be empty or trivial).")

    # 3. Compare to log(N)
    log_n = math.log(G.number_of_nodes())
    print(f"\nlog(N) = {log_n:.4f}, where N = {G.number_of_nodes()}")

    # 4. Final Interpretation
    print("\n--- Interpretation ---")
    print("Small-world networks typically have:")
    print("1. Short average path length (≈ log(N))")
    print("2. Significant reciprocity or local clustering")

    if path_lengths:
        short_path_check = avg_path_length <= log_n * 1.5
        reciprocity_check = reciprocity is not None and reciprocity > 0.1

        if short_path_check and reciprocity_check:
            print(" The graph likely exhibits small-world properties.")
        elif short_path_check:
            print(" Short paths detected, but low reciprocity – might still be small-world.")
        else:
            print(" The graph likely does NOT exhibit small-world properties.")
    else:
        print(" Could not assess small-world properties due to lack of path data.")

def density(G):

    density = nx.density(G)
    print("Density:", density)
    return density

def degree_distributions(G, degree_type='in', title=' ', x_min=None, x_max=None, color=None):
    """
    Plot log-X histogram of degree distribution for nodes of a specific color.

    Parameters:
    - G: NetworkX DiGraph
    - degree_type: 'in', 'out', or 'total'
    - title: Plot title
    - x_min, x_max: Exponents for logspace binning
    - color: One of 'red', 'yellow', 'blue', or None (for all nodes)
    """

    def compute_average_rating(G):
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = 0
        return node_avg_rating

    avg_rating = compute_average_rating(G)

    if color == 'red':
        nodes = [n for n, avg in avg_rating.items() if avg <= -2]
        title += ' (Red: ≤ -2)'
    elif color == 'yellow':
        nodes = [n for n, avg in avg_rating.items() if -2 < avg < 2]
        title += ' (Yellow: -2 < avg < 2)'
    elif color == 'blue':
        nodes = [n for n, avg in avg_rating.items() if avg >= 2]
        title += ' (Blue: ≥ 2)'
    elif color is None:
        nodes = list(G.nodes())
    else:
        raise ValueError("color must be 'red', 'yellow', 'blue', or None")

    if degree_type == 'in':
        degrees = [G.in_degree(n) for n in nodes]
        label = 'In-Degree'
    elif degree_type == 'out':
        degrees = [G.out_degree(n) for n in nodes]
        label = 'Out-Degree'
    elif degree_type == 'total':
        degrees = [G.in_degree(n) + G.out_degree(n) for n in nodes]
        label = 'Total Degree'
    else:
        raise ValueError("degree_type must be 'in', 'out', or 'total'")

    degrees = [d for d in degrees if d > 0]  # להסיר אפסים כדי לא לשבור את log

    if not degrees:
        print(f"No degrees to plot for color: {color}")
        return

    if x_min is None:
        x_min = math.log10(max(min(degrees), 1))
    if x_max is None:
        x_max = math.log10(max(degrees))

    bins = np.logspace(x_min, x_max, num=20)

    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=bins, color=color if color else 'salmon', edgecolor='black')
    plt.xscale('log')
    plt.xlabel(f'{label} (log scale)')
    plt.ylabel('Frequency')
    plt.title(f'{label} Distribution (Log X-axis) {title}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def all_degree_distributions(G):
    degree_distributions(G, degree_type='in', title='max_connected_component_graph',x_min=2, x_max=3, color='blue')
    degree_distributions(G, degree_type='out', title='max_connected_component_graph', x_min=2, x_max=3, color='blue')
    degree_distributions(G, degree_type='total', title='max_connected_component_graph', x_min=2, x_max=3, color='blue')

    degree_distributions(G, degree_type='in', title='max_connected_component_graph',x_min=1, x_max=3, color='red')
    degree_distributions(G, degree_type='out', title='max_connected_component_graph', x_min=1, x_max=3, color='red')
    degree_distributions(G, degree_type='total', title='max_connected_component_graph', x_min=1, x_max=3, color='red')

    degree_distributions(G, degree_type='in', title='max_connected_component_graph',x_min=2, x_max=3, color='yellow')
    degree_distributions(G, degree_type='out', title='max_connected_component_graph', x_min=2, x_max=3, color='yellow')
    degree_distributions(G, degree_type='total', title='max_connected_component_graph', x_min=2, x_max=3, color='yellow')

    degree_distributions(G, degree_type='in', title='max_connected_component_graph',x_min=2, x_max=3)
    degree_distributions(G, degree_type='out', title='max_connected_component_graph', x_min=2, x_max=3)
    degree_distributions(G, degree_type='total', title='max_connected_component_graph', x_min=2, x_max=3)

def centrality(G):
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    def draw_centrality(centrality, title, min_exp, max_exp):

        values = np.array(list(centrality.values()))
        values = values[values > 0]

        bins = np.logspace(min_exp, max_exp, num=20)

        plt.figure(figsize=(12, 6), facecolor='white')
        plt.hist(values, bins=bins, color='salmon', edgecolor='cyan')
        plt.xscale('log')
        plt.xlabel(f"{title} Centrality (log scale)", color='black')
        plt.ylabel("Number of Nodes", color='black')
        plt.title(f"Distribution of {title} Centrality", color='black')
        plt.tick_params(colors='black')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    draw_centrality(closeness_centrality, "Closeness", -1, 0)
    draw_centrality(betweenness_centrality, "Betweenness", -5, -1)

def compare_centrality(G, weight='weight', alpha=0.85, max_iter=100):
    def compute_top_pagerank_nodes(G, weight, alpha, max_iter, top_k=10):
        # שימוש במשקלים מוחלטים
        G_abs = G.copy()
        for u, v, data in G_abs.edges(data=True):
            if weight in data:
                data[weight] = abs(data[weight])

        try:
            pagerank_scores = nx.pagerank(G_abs, weight=weight, alpha=alpha, max_iter=max_iter)
        except nx.PowerIterationFailedConvergence:
            print("⚠️ Power iteration did not converge with weights. Retrying without weights...")
            pagerank_scores = nx.pagerank(G_abs, weight=None, alpha=alpha, max_iter=max_iter)

        sorted_scores = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]

    # חישובי מרכזיות אחרים
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    top_pagerank = compute_top_pagerank_nodes(G, weight, alpha, max_iter)
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]


    print("\nTop 10 PageRank Nodes:")
    # for node, score in top_pagerank:
    #     print(f"Node {node}: PageRank = {score:.5f}")
    print([node for node, _ in top_pagerank])


    print("\nTop 10 Betweenness Centrality Nodes:")
    print([node for node, _ in top_betweenness])

    print("\nTop 10 Closeness Centrality Nodes:")
    print([node for node, _ in top_closeness])

    # Venn Diagram
    plt.figure(figsize=(5, 3))
    venn = venn3([set(node for node, _ in top_pagerank),
                  set(node for node, _ in top_betweenness),
                  set(node for node, _ in top_closeness)],
                 set_labels=("PageRank", "Betweenness", "Closeness"))

    colors = {
        '100': '#1f77b4', '010': '#ff7f0e', '001': '#2ca02c',
        '110': '#9467bd', '101': '#8c564b', '011': '#e377c2', '111': '#7f7f7f'
    }

    for subset_id, color in colors.items():
        patch = venn.get_patch_by_id(subset_id)
        if patch:
            patch.set_facecolor(color)

    for label in venn.subset_labels:
        if label:
            label.set_text('')

    plt.title("Top 10 Centrality Nodes Overlap")
    plt.tight_layout()
    plt.show()

def power_law_no_binning(G, show_fit=True, color=None):
    import warnings
    warnings.filterwarnings("ignore")

    def compute_average_rating(G):
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = 0
        return node_avg_rating

    avg_rating = compute_average_rating(G)

    if color == 'red':
        nodes = [n for n, avg in avg_rating.items() if avg <= -2]
        title = "Power-law (Red ≤ -2)"
    elif color == 'yellow':
        nodes = [n for n, avg in avg_rating.items() if -2 < avg < 2]
        title = "Power-law (Yellow -2 < avg < 2)"
    elif color == 'blue':
        nodes = [n for n, avg in avg_rating.items() if avg >= 2]
        title = "Power-law (Blue ≥ 2)"
    else:
        nodes = list(G.nodes())
        title = "Power-law (All Nodes)"

    degrees = [G.degree(n) for n in nodes if G.degree(n) > 0]

    if not degrees:
        print(f"No degrees to plot for color: {color}")
        return

    hist, bin_edges = np.histogram(degrees, bins=range(1, max(degrees) + 2), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist, width=0.8, color='orange', edgecolor='black', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Vertex Degree")
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)

    if show_fit:
        try:
            fit = powerlaw.Fit(degrees, discrete=True)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            R, p = fit.distribution_compare('power_law', 'lognormal')

            print(f"⚙️ Raw Fit Results for {color or 'all'}:")
            print(f"  α (power-law exponent): {alpha:.3f}")
            print(f"  xmin: {xmin}")
            print(f"  Distribution is power-law? {'Yes' if p > 0.05 else 'No'} (p={p:.4f})")

            x_fit = np.linspace(xmin, max(degrees), 100)
            y_fit = (x_fit / xmin) ** (-alpha)
            y_fit *= hist[bin_centers >= xmin][0] / y_fit[0]
            plt.xlim(left=xmin)
            plt.plot(x_fit, y_fit, 'r--', label=f'Power-law fit (γ={alpha:.2f})')
            plt.legend()
        except ImportError:
            print("⚠️ כדי להשתמש באופציית fit, יש להתקין את הספרייה 'powerlaw'")

    plt.tight_layout()
    plt.show()

def power_law_binning_logarithm(G, bins=20, show_fit=True, color=None):
    import warnings
    warnings.filterwarnings("ignore")

    def compute_average_rating(G):
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = 0
        return node_avg_rating

    avg_rating = compute_average_rating(G)

    if color == 'red':
        nodes = [n for n, avg in avg_rating.items() if avg <= -2]
        title = "Log-Binned Power-law (Red ≤ -2)"
    elif color == 'yellow':
        nodes = [n for n, avg in avg_rating.items() if -2 < avg < 2]
        title = "Log-Binned Power-law (Yellow -2 < avg < 2)"
    elif color == 'blue':
        nodes = [n for n, avg in avg_rating.items() if avg >= 2]
        title = "Log-Binned Power-law (Blue ≥ 2)"
    else:
        nodes = list(G.nodes())
        title = "Log-Binned Power-law (All Nodes)"

    degrees = [G.degree(n) for n in nodes if G.degree(n) > 0]

    if not degrees:
        print(f"No degrees to plot for color: {color}")
        return

    min_deg = max(min(degrees), 1)
    max_deg = max(degrees)
    log_bins = np.logspace(np.log10(min_deg), np.log10(max_deg), bins)

    hist, bin_edges = np.histogram(degrees, bins=log_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist, width=np.diff(bin_edges), align='center', alpha=0.7, color='orange', edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Vertex Degree")
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)

    if show_fit:
        try:
            fit = powerlaw.Fit(degrees, discrete=True)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            R, p = fit.distribution_compare('power_law', 'lognormal')

            print(f"⚙️ Log-Binned Fit Results for {color or 'all'}:")
            print(f"  α (power-law exponent): {alpha:.3f}")
            print(f"  xmin: {xmin}")
            print(f"  Distribution is power-law? {'Yes' if p > 0.05 else 'No'} (p={p:.4f})")

            x_fit = np.linspace(xmin, max_deg, 100)
            y_fit = (x_fit / xmin) ** (-alpha)
            y_fit *= hist[bin_centers >= xmin][0] / y_fit[0]
            plt.xlim(left=xmin)
            plt.plot(x_fit, y_fit, 'r--', label=f'Power-law fit (γ={alpha:.2f})')
            plt.legend()
        except ImportError:
            print("⚠️ כדי להשתמש באופציית fit, יש להתקין את הספרייה 'powerlaw'")

    plt.tight_layout()
    plt.show()

def all_power_law(G):

    power_law_no_binning(G, show_fit=True, color='blue')
    power_law_binning_logarithm(G, bins=20, show_fit=True, color='blue')

    power_law_no_binning(G, show_fit=True, color='red')
    power_law_binning_logarithm(G, bins=20, show_fit=True, color='red')

    power_law_no_binning(G, show_fit=True, color='yellow')
    power_law_binning_logarithm(G, bins=20, show_fit=True, color='yellow')

    power_law_no_binning(G, show_fit=True)
    power_law_binning_logarithm(G, bins=20, show_fit=True)

def draw_rating_histogram(G):

    def compute_average_rating(G):

        # Compute average incoming rating per node
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = 0
        return node_avg_rating

    node_avg_rating = compute_average_rating(G)
    # המרת הדירוגים למערך של ערכים
    ratings = list(node_avg_rating.values())

    # יצירת היסטוגרמה עם טווח ערכים ב-10-10
    plt.figure(figsize=(8, 6))

    # הגדרת טווח ציר X (מ-10 עד 10, כל 1 יחידה)
    bins = np.arange(-10, 11, 1)

    # יצירת היסטוגרמה
    plt.hist(ratings, bins=bins, edgecolor='black', color='deepskyblue', alpha=0.7)

    # כותרת ותיוגים
    plt.title('Distribution of Node Average Ratings', fontsize=16)
    plt.xlabel('Average Rating', fontsize=12)
    plt.ylabel('Number of Nodes', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # הצגת ההיסטוגרמה
    plt.tight_layout()
    plt.show()

def overlap(G, title, filename):

    from collections import defaultdict

    # Function to calculate the neighborhood overlap for each edge in the graph
    def calculate_neighborhood_overlap(G):
        overlaps = []  # List to store overlap coefficients
        weights = []  # List to store edge weights

        for u, v, data in G.edges(data=True):  # Iterate through all edges in the graph
            neighbors_u = set(G.neighbors(u))  # Get the neighbors of node u
            neighbors_v = set(G.neighbors(v))  # Get the neighbors of node v

            common = neighbors_u & neighbors_v  # Find the common neighbors between u and v
            union = neighbors_u | neighbors_v  # Find the union of neighbors between u and v

            # Calculate the overlap coefficient (common neighbors / total unique neighbors)
            overlap = len(common) / len(union) if len(union) > 0 else 0

            overlaps.append(overlap)  # Append the overlap coefficient to the list
            weights.append(data['weight'])  # Append the weight of the edge to the list

        return overlaps, weights  # Return the overlap coefficients and weights

    def plot_neighborhood_overlapNEW(G, title, filename):
        overlaps, weights = calculate_neighborhood_overlap(G)

        # צבירת overlaps לכל ערך weight ייחודי
        weight_to_overlaps = defaultdict(list)
        for w, o in zip(weights, overlaps):
            weight_to_overlaps[w].append(o)

        # חישוב ממוצע לכל משקל
        unique_weights = sorted(weight_to_overlaps.keys())
        avg_overlaps = [np.mean(weight_to_overlaps[w]) for w in unique_weights]

        # חישוב קו מגמה
        slope, intercept, _, _, _ = linregress(unique_weights, avg_overlaps)
        trend_y = [slope * w + intercept for w in unique_weights]

        # ציור
        plt.style.use('default')
        plt.figure(figsize=(10, 6), facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')

        # ציור נקודה אחת לכל ערך weight
        plt.scatter(unique_weights, avg_overlaps, alpha=0.8, color='blue',
                    edgecolors='green', linewidths=0.3, label='Avg Overlap per Weight')

        # קו מגמה
        plt.plot(unique_weights, trend_y, linestyle='--', color='red', linewidth=2, label='Trend Line')

        plt.title(title, fontsize=14, weight='bold', color='black')
        plt.xlabel("Weight", fontsize=12, color='black')
        plt.ylabel("Average Overlap", fontsize=12, color='black')
        plt.xticks(color='black')
        plt.yticks(color='black')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()

    plot_neighborhood_overlapNEW(G, title, filename)

def random_graph(G):
    # --- Gilbert model Random graphes  G(n, m) ---
    def build_gnm_graph(G):
        n = G.number_of_nodes()
        m = G.number_of_edges()
        gnm = nx.gnm_random_graph(n, m, directed=True)
        return gnm

    def build_preferential_attachment_model(original_graph, seed=None):
        if seed is not None:
            random.seed(seed)

        n = original_graph.number_of_nodes()
        m_total = original_graph.number_of_edges()

        # Estimate m: average out-degree (can be rounded down to avoid over-connectivity)
        # m = max(1, int(m_total / n))
        m = 5
        G = nx.DiGraph()

        # Start with m isolated nodes
        for i in range(m):
            G.add_node(i)

        for new_node in range(m, n):
            G.add_node(new_node)

            # Get current in-degrees with smoothing to avoid zero probability
            targets = list(G.nodes)
            in_degrees = [G.in_degree(node) + 1 for node in targets]
            total_weight = sum(in_degrees)

            chosen = set()
            while len(chosen) < min(m, len(targets)):
                r = random.uniform(0, total_weight)
                acc = 0
                for node, weight in zip(targets, in_degrees):
                    acc += weight
                    if acc >= r:
                        if node != new_node and node not in chosen:
                            chosen.add(node)
                        break

            # Add directed edges from the new node to the chosen targets
            for target in chosen:
                G.add_edge(new_node, target)

        # Print summary
        print("Original graph:")
        print(f"- {n} nodes")
        print(f"- {m_total} edges")

        print("\nGenerated directed graph (Preferential Attachment):")
        print(f"- {G.number_of_nodes()} nodes")
        print(f"- {G.number_of_edges()} edges")
        print(f"Chosen m value: {m}")

        return G

    def mixed_preferential_attachment(G_orig, m=3):
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
                print(
                    f"Warning: Node {new_node} connected to only {len(targets)} targets instead of {m} after {attempts} attempts.")

            for target in targets:
                G.add_edge(new_node, target)

        return G

def average_distance_directed(G):
    """
    מחשבת את מרחק המסלול הקצר הממוצע בגרף מכוון.
    תומכת רק אם הוא strongly connected.

    Parameters:
    - G: nx.DiGraph
    """
    if nx.is_strongly_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        print("Warning: Graph is not strongly connected.")
        return float('inf')

def count_directed_cycles(G):
    if not G.is_directed():
        raise ValueError("graph must be directed")

    cycles = list(nx.simple_cycles(G))
    print(f"Num of cycles in the graph: {len(cycles)}")
    return cycles

def calculate_directed_triangle_percentage(G):
    if not G.is_directed():
        raise ValueError("This function is for directed graphs only.")

    # ממיר לגרף לא מכוון בשביל לספור משולשים (סופרת את כל המשולשים)
    undirected_G = G.to_undirected()
    triangle_counts = nx.triangles(undirected_G)

    total_triangles = sum(triangle_counts.values()) // 3  # כל משולש נספר שלוש פעמים
    possible_triplets = sum(1 for _ in nx.triads_by_type(G) if _ != '003')

    percentage = (total_triangles / G.number_of_nodes()) * 100 if G.number_of_nodes() > 0 else 0

    print(f"מספר המשולשים (משולשים סגורים): {total_triangles}")
    print(f"אחוז המשולשים מתוך מספר הקודקודים: {percentage:.2f}%")

    return total_triangles, percentage

def create_orders_and_draw(G):

    def giant_component_sizes(G, edge_order):
        sizes = []
        for edge in edge_order:
            if G.has_edge(*edge):
                G.remove_edge(*edge)
            if len(G.edges) > 0:
                largest_cc = max(nx.strongly_connected_components(G), key=len)
                sizes.append(len(largest_cc))
            else:
                sizes.append(0)
        return sizes

    G_random = build_max_connected_component_graph(G)
    G_heavy_first = build_max_connected_component_graph(G)
    G_light_first = build_max_connected_component_graph(G)
    G_betweenness = build_max_connected_component_graph(G)

    # 1. רנדומלי
    random_edges = list(G_random.edges())
    random.shuffle(random_edges)
    sizes_random = giant_component_sizes(G_random, random_edges)

    # 2. כבדות -> קלות
    heavy_edges = sorted(G_heavy_first.edges(data=True), key=lambda x: -x[2]['weight'])
    heavy_edges_list = [(u, v) for u, v, _ in heavy_edges]
    sizes_heavy = giant_component_sizes(G_heavy_first, heavy_edges_list)

    # 3. קלות -> כבדות
    light_edges = sorted(G_light_first.edges(data=True), key=lambda x: x[2]['weight'])
    light_edges_list = [(u, v) for u, v, _ in light_edges]
    sizes_light = giant_component_sizes(G_light_first, light_edges_list)

    # 4. לפי Betweenness
    betweenness = nx.edge_betweenness_centrality(G_betweenness)
    betweenness_edges = sorted(betweenness.items(), key=lambda x: -x[1])
    betweenness_edges_list = [edge for edge, _ in betweenness_edges]
    sizes_betweenness = giant_component_sizes(G_betweenness, betweenness_edges_list)

    # ציור גרף אחד עם ארבע עקומות
    plt.figure(figsize=(12, 8))
    plt.plot(sizes_random, label="Random Removal", color='blue')
    plt.plot(sizes_heavy, label="Heavy → Light", color='red')
    plt.plot(sizes_light, label="Light → Heavy", color='green')
    plt.plot(sizes_betweenness, label="High Betweenness", color='purple')

    plt.xlabel("Edges Removed")
    plt.ylabel("Size of Giant Component")
    plt.title("Edge Removal and Giant Component Size")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    axs[0].plot(sizes_random, color='blue')
    axs[0].set_title("Random Edge Removal")

    axs[1].plot(sizes_heavy, color='red')
    axs[1].set_title("Heavy to Light Edge Removal")

    axs[2].plot(sizes_light, color='green')
    axs[2].set_title("Light to Heavy Edge Removal")

    axs[3].plot(sizes_betweenness, color='purple')
    axs[3].set_title("High Betweenness Edge Removal")

    for ax in axs:
        ax.set_xlabel("Edges Removed")
        ax.set_ylabel("Size of Giant Component")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def calculate_symmetric_edge_percentage(G):
    if not G.is_directed():
        raise ValueError("Graph must be directed.")

    total_edges = G.number_of_edges()
    symmetric_edges = 0

    visited = set()

    for u, v, data in G.edges(data=True):
        if (u, v) in visited or (v, u) in visited:
            continue  # Avoid counting the same pair twice
        if G.has_edge(v, u):
            weight_uv = data.get('weight')
            weight_vu = G[v][u].get('weight')
            if weight_uv == weight_vu:
                symmetric_edges += 2  # Count both (u,v) and (v,u)
        visited.add((u, v))
        visited.add((v, u))

    percentage = (symmetric_edges / total_edges) * 100 if total_edges > 0 else 0
    print(f"Percentage of symmetric edges with identical weight: {percentage:.2f}% ({symmetric_edges} out of {total_edges})")
    return percentage

def compute_average_rating(G):
    node_avg_rating = {}
    for node in G.nodes():
        in_edges = G.in_edges(node, data=True)
        if in_edges:
            avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
            node_avg_rating[node] = avg
        else:
            node_avg_rating[node] = 0
    return node_avg_rating

def get_color(avg):
    if avg <= -2:
        return 'red'
    elif avg < 2:
        return 'yellow'
    else:
        return 'blue'

def compute_symmetric_edge_percentages_by_color(G):
    node_avg_rating = compute_average_rating(G)
    color_map = {node: get_color(avg) for node, avg in node_avg_rating.items()}

    color_groups = {'red': set(), 'yellow': set(), 'blue': set()}
    for node, color in color_map.items():
        color_groups[color].add(node)

    percentages = {}
    for color, nodes in color_groups.items():
        sub_edges = [(u, v) for u, v in G.edges() if u in nodes and v in nodes]
        edge_set = set(sub_edges)
        total = len(sub_edges)
        symmetric = 0

        for u, v in sub_edges:
            if (v, u) in edge_set:
                if G[u][v]['weight'] == G[v][u]['weight']:
                    symmetric += 1

        percent = (symmetric / total * 100) if total > 0 else 0
        percentages[color] = {
            'total_edges': total,
            'symmetric_edges': symmetric,
            'percentage': percent
        }

    return percentages

def check_symmetric_edge_percentages_all_colors(G):
    result = compute_symmetric_edge_percentages_by_color(G)
    for color, data in result.items():
        print(f"{color.upper()} — Symmetric edges: {data['symmetric_edges']}/{data['total_edges']} "
              f"({data['percentage']:.2f}%)")

def compute_equal_in_out_degree_percentage(G):
    count_equal = 0
    total_nodes = G.number_of_nodes()

    for node in G.nodes():
        indeg = G.in_degree(node)
        outdeg = G.out_degree(node)
        if indeg == outdeg:
            count_equal += 1

    percentage = (count_equal / total_nodes * 100) if total_nodes > 0 else 0
    return percentage, count_equal, total_nodes

def directed_graph_modularity(G: nx.DiGraph, use_weights=True):
        """
        מקבלת גרף מכוון, מזהה קהילות ומחזירה את המודולריות שלהן.

        :param G: גרף מכוון (nx.DiGraph)
        :param use_weights: האם להשתמש במשקלים
        :return: ערך המודולריות (float)
        """
        # יוצרים גרף לא מכוון לצורך גילוי קהילות (אלגוריתם greedy לא עובד על גרפים מכוונים)
        G_undirected = G.to_undirected()

        # זיהוי קהילות באמצעות greedy modularity
        communities = list(greedy_modularity_communities(G_undirected, weight='weight' if use_weights else None))

        # חישוב מודולריות לפי הגרף המקורי (המכוון)
        mod = modularity(G, communities, weight='weight' if use_weights else None)

        return mod


# -----------------------------------------------------------------
# draw 3 graphes - 3 colores
def split_graph_by_color(G):
    def compute_average_rating(G):
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = 0
        return node_avg_rating

    def get_color(avg):
        if avg <= -2:
            return 'red'
        elif avg < 2:
            return 'yellow'
        else:
            return 'blue'

    node_avg_rating = compute_average_rating(G)
    color_map = {node: get_color(avg) for node, avg in node_avg_rating.items()}

    # קיבוץ צמתים לפי צבע
    color_groups = {'red': set(), 'yellow': set(), 'blue': set()}
    for node, color in color_map.items():
        color_groups[color].add(node)

    subgraphs = {}
    for color, nodes in color_groups.items():
        # סינון הקשתות: רק קשתות בין צמתים באותו צבע
        edges = [(u, v, data) for u, v, data in G.edges(data=True)
                 if u in nodes and v in nodes]
        subG = nx.DiGraph()
        subG.add_nodes_from(nodes)
        subG.add_edges_from(edges)
        subgraphs[color] = subG

    # הדפסת מידע
    for color, subG in subgraphs.items():
        print(f"\n=== Subgraph for {color.upper()} nodes ===")
        print(f"Number of nodes: {subG.number_of_nodes()}")
        print(f"Number of edges: {subG.number_of_edges()}")

        # אופציונלית: ציור כל תת-גרף
        pos = nx.spring_layout(subG, seed=42)
        plt.figure(figsize=(6, 5))
        nx.draw(subG, pos, with_labels=False, node_color=color,
                edge_color='gray', node_size=40, arrowsize=10)
        plt.title(f"{color.upper()} Subgraph")
        plt.tight_layout()
        plt.show()

def build_cross_color_edges_graph(G):
    def compute_average_rating(G):
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = None
        return node_avg_rating

    def get_color(avg):
        if avg is None:
            return None
        if avg <= -2:
            return 'red'
        elif avg < 2:
            return 'yellow'
        else:
            return 'blue'

    # שלב 1: חישוב צבע לכל קודקוד
    node_avg_rating = compute_average_rating(G)
    color_map = {node: get_color(avg) for node, avg in node_avg_rating.items() if get_color(avg)}

    # שלב 2: מציאת כל הקשתות בין צבעים שונים
    cross_color_edges = []
    color_counts = defaultdict(int)
    weights = []

    for u, v, data in G.edges(data=True):
        if u in color_map and v in color_map:
            color_u = color_map[u]
            color_v = color_map[v]
            if color_u != color_v:
                cross_color_edges.append((u, v, data))
                weights.append(data['weight'])
                color_counts[(color_u, color_v)] += 1

    # שלב 3: בניית גרף חדש
    cross_color_G = nx.DiGraph()
    cross_color_G.add_edges_from(cross_color_edges)

    print(f"\n🎯 קשתות צבעוניות (מקודקודים בצבעים שונים): {len(cross_color_edges)}")
    print(f"🧮 מספר קודקודים בגרף החדש: {cross_color_G.number_of_nodes()}")
    print(f"📊 טווח משקלים: {min(weights)} עד {max(weights)}")
    print(f"📈 ממוצע משקלים: {sum(weights) / len(weights):.2f}")

    print("\n📚 פירוט לפי סוגי צבעים:")
    for (src_color, dst_color), count in color_counts.items():
        print(f"  {src_color.upper()} → {dst_color.upper()}: {count} קשתות")

    # שלב 4: ציור
    pos = nx.spring_layout(cross_color_G, seed=42)
    node_colors = [color_map[n] for n in cross_color_G.nodes()]
    nx.draw(cross_color_G, pos,
            node_color=node_colors,
            edge_color='gray',
            node_size=40,
            arrowsize=10)
    plt.title("גרף של קשתות צבעוניות (בין צבעים שונים)")
    plt.tight_layout()
    plt.show()

    return cross_color_G

def compute_symmetric_edge_percentage_by_sign(G):
    if not G.is_directed():
        raise ValueError("Graph must be directed.")

    pos_total = 0
    pos_symmetric_edges = 0  # ספירת קשתות סימטריות עם משקל חיובי

    neg_total = 0
    neg_symmetric_edges = 0  # ספירת קשתות סימטריות עם משקל שלילי

    checked_pairs = set()

    for u, v, data in G.edges(data=True):
        weight = data.get('weight')
        if weight is None:
            continue

        # סופרים את הקשת ככולה
        if weight > 0:
            pos_total += 1
        elif weight < 0:
            neg_total += 1
        else:
            continue  # משקל 0 - מתעלמים

        # בודקים אם כבר סקרנו את הזוג ההפוך, כדי למנוע ספירה כפולה
        if (v, u) in checked_pairs:
            continue

        # בודקים אם קיימת קשת הפוכה עם אותו משקל
        if G.has_edge(v, u):
            rev_weight = G[v][u].get('weight')
            if rev_weight == weight:
                # סימטריה - מוסיפים פעמיים כי זו זוג קשתות סימטריות
                if weight > 0:
                    pos_symmetric_edges += 2
                else:
                    neg_symmetric_edges += 2

        checked_pairs.add((u, v))
        checked_pairs.add((v, u))

    pos_percent = (pos_symmetric_edges / pos_total * 100) if pos_total > 0 else 0
    neg_percent = (neg_symmetric_edges / neg_total * 100) if neg_total > 0 else 0

    print(f"POSITIVE — Symmetric edges: {pos_symmetric_edges}/{pos_total} ({pos_percent:.2f}%)")
    print(f"NEGATIVE — Symmetric edges: {neg_symmetric_edges}/{neg_total} ({neg_percent:.2f}%)")

    return {
        'positive': {
            'total_edges': pos_total,
            'symmetric_edges': pos_symmetric_edges,
            'percentage': pos_percent
        },
        'negative': {
            'total_edges': neg_total,
            'symmetric_edges': neg_symmetric_edges,
            'percentage': neg_percent
        }
    }

def directed_graph_modularity_with_communities(G: nx.DiGraph, use_weights=True):
    """
    Detects communities in a directed graph (by temporarily converting it to undirected),
    computes the modularity score, and returns both the score and the community structure.

    :param G: A directed NetworkX graph (DiGraph)
    :param use_weights: Whether to use edge weights
    :return: tuple:
        - modularity_score (float): the modularity value
        - community_dict (dict): mapping of node -> community index
        - communities (list of sets): list of communities as sets of nodes
    """
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()

    # Detect communities using the greedy modularity algorithm
    communities = list(
        greedy_modularity_communities(
            G_undirected,
            weight='weight' if use_weights else None
        )
    )

    # Compute modularity using the original directed graph
    modularity_score = modularity(
        G,
        communities,
        weight='weight' if use_weights else None
    )

    # Build node -> community mapping
    community_dict = {}
    for i, comm in enumerate(communities):
        for node in comm:
            community_dict[node] = i

    print(f"Modularity: {modularity_score:.4f}")
    print(f"Number of communities: {len(communities)}")
    # for i, comm in enumerate(communities):
        # print(f"Community {i + 1}: {sorted(comm)}")

    return modularity_score, community_dict, communities

def remove_node_from_graph(G: nx.Graph, node_id):
    """
    Returns a copy of the graph without the specified node.

    :param G: NetworkX graph (can be directed or undirected)
    :param node_id: the node to remove
    :return: a new graph without the given node
    """
    if node_id not in G:
        raise ValueError(f"Node {node_id} does not exist in the graph.")

    G_copy = G.copy()
    G_copy.remove_node(node_id)
    return G_copy

def build_graph_weight_eq_to_target_only(G: nx.DiGraph, target_weight=-10):
    filtered_G = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data.get('weight') == target_weight:
            filtered_G.add_edge(u, v, weight=target_weight)  # שמירת המשקל

    filtered_G = filtered_G.subgraph(set(filtered_G.nodes)).copy()

    print(f"\nFiltered Graph: {filtered_G.number_of_nodes()} nodes, {filtered_G.number_of_edges()} edges")

    # ציור גרף
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(filtered_G, seed=42)
    nx.draw_networkx_nodes(filtered_G, pos, node_color='red', node_size=80)
    nx.draw_networkx_edges(filtered_G, pos, edge_color='gray', arrows=True, arrowstyle='->', width=1)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return filtered_G


# def spreading_mode(G):
#     import pandas as pd
#     import networkx as nx
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import random
#
#     # =====================
#     # Color Assignment by Rating
#     # =====================
#     def compute_average_rating(G):
#         node_avg_rating = {}
#         for node in G.nodes():
#             in_edges = G.in_edges(node, data=True)
#             if in_edges:
#                 # avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
#                 avg = sum(data.get('weight', 1.0) for _, _, data in in_edges) / len(in_edges)
#
#                 node_avg_rating[node] = avg
#             else:
#                 node_avg_rating[node] = 0
#         return node_avg_rating
#
#     def assign_colors_by_rating(G):
#         avg_rating = compute_average_rating(G)
#         for node in G.nodes():
#             avg = avg_rating.get(node, 0)
#             if avg <= -2:
#                 color = 'red'
#             elif -2 < avg < 2:
#                 color = 'yellow'
#             else:
#                 color = 'blue'
#             G.nodes[node]['color'] = color
#
#     assign_colors_by_rating(G)
#
#     # =====================
#     # Spreading function (with regulation set injection)
#     # =====================
#     def spread_message(G, source_nodes, p, steps=10, regulation_set=None):
#         informed = set(source_nodes)
#         history = []
#
#         for _ in range(steps):
#             new_informed = set(informed)
#             for node in informed:
#                 node_color = G.nodes[node]['color']
#                 for neighbor in G.neighbors(node):
#                     if neighbor in informed:
#                         continue
#                     neighbor_color = G.nodes[neighbor]['color']
#                     prob = p if node_color == neighbor_color else (1 - p)
#                     if random.random() < prob:
#                         new_informed.add(neighbor)
#
#             if regulation_set:
#                 new_informed |= regulation_set
#
#             informed = new_informed
#             reds = sum(1 for n in informed if G.nodes[n]['color'] == 'red')
#             yellows = sum(1 for n in informed if G.nodes[n]['color'] == 'yellow')
#             history.append((reds, yellows))
#
#         return history
#
#     # =====================
#     # Regulation functions
#     # =====================
#     def rlr_set(rho):
#         return set(random.sample(list(G.nodes()), int(len(G) * rho)))
#
#     def blue_only_set(rho):
#         yellow_nodes = [n for n in G.nodes if G.nodes[n]['color'] == 'yellow']
#         return set(random.sample(yellow_nodes, int(len(yellow_nodes) * rho)))
#
#     # =====================
#     # Averaging Function
#     # =====================
#     def average_spread(G, source_selector, p, steps=10, regulation_set_generator=None, runs=100):
#         reds_all = np.zeros(steps)
#         yellows_all = np.zeros(steps)
#
#         for _ in range(runs):
#             source_nodes = source_selector()
#             reg_set = regulation_set_generator() if regulation_set_generator else None
#             history = spread_message(G, source_nodes, p, steps, regulation_set=reg_set)
#             reds = np.array([r for r, y in history])
#             yellows = np.array([y for r, y in history])
#             reds_all += reds
#             yellows_all += yellows
#
#         return reds_all / runs, yellows_all / runs
#
#     def select_random_reds(k=1):
#         red_nodes = [n for n in G.nodes if G.nodes[n]['color'] == 'red']
#         return random.sample(red_nodes, k)
#
#     # =====================
#     # Run Scenarios
#     # =====================
#     scenarios = {
#         "Strong No-Reg": (1.0, None),
#         "p=0.7 No-Reg": (0.7, None),
#         "Strong RLR(0.25)": (1.0, lambda: rlr_set(0.25)),
#         "p=0.7 RLR(0.25)": (0.7, lambda: rlr_set(0.25)),
#         "Strong BlueOnly(0.25)": (1.0, lambda: blue_only_set(0.25)),
#         "p=0.7 BlueOnly(0.25)": (0.7, lambda: blue_only_set(0.25)),
#     }
#
#     average_results = {}
#     for label, (p_val, reg_fn) in scenarios.items():
#         reds_avg, yellows_avg = average_spread(G, select_random_reds, p=p_val,
#                                                regulation_set_generator=reg_fn, steps=10, runs=100)
#         average_results[label] = (reds_avg, yellows_avg)
#
#     # =====================
#     # Plot Results
#     # =====================
#     plt.figure(figsize=(10, 6))
#     colors = {
#         "p=0.7 No-Reg": "orange",
#         "p=0.7 RLR(0.25)": "purple",
#         "Strong No-Reg": "blue",
#         "Strong RLR(0.25)": "green",
#         "Strong BlueOnly(0.25)": "brown",
#         "p=0.7 BlueOnly(0.25)": "red"
#     }
#
#     for label, (reds, yellows) in average_results.items():
#         plt.plot(reds, label=f"{label} – Red", linestyle='--', color=colors[label])
#         plt.plot(yellows, label=f"{label} – Yellow", linestyle='-', color=colors[label])
#
#     plt.xlabel("Time Step")
#     plt.ylabel("Average Informed Users")
#     plt.title("Average Spread Over 100 Runs (with Regulation Support)")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()
#
#     # =====================
#     # Compute Echo Chamber Metrics
#     # =====================
#     print("\n=== Echo Chamber Metrics ===")
#     for label, (reds_avg, yellows_avg) in average_results.items():
#         alpha = reds_avg[-1] + yellows_avg[-1]
#         phi_red = reds_avg[-1] / alpha if alpha > 0 else 0
#         phi_yellow = yellows_avg[-1] / alpha if alpha > 0 else 0
#         print(f"{label}:")
#         print(f"  α (size) = {alpha:.2f}, ϕ_red = {phi_red:.2f}, ϕ_yellow = {phi_yellow:.2f}")

def spreading_mode(G):
    import pandas as pd
    import networkx as nx
    import matplotlib.pyplot as plt
    import numpy as np
    import random

    # =====================
    # Color Assignment by Rating
    # =====================
    def compute_average_rating(G):
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data.get('weight', 1.0) for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = 0
        return node_avg_rating

    def assign_colors_by_rating(G):
        avg_rating = compute_average_rating(G)
        for node in G.nodes():
            avg = avg_rating.get(node, 0)
            if avg <= -2:
                color = 'red'
            elif -2 < avg < 2:
                color = 'yellow'
            else:
                color = 'blue'
            G.nodes[node]['color'] = color

    assign_colors_by_rating(G)

    # =====================
    # Spreading function (with regulation set injection)
    # =====================
    def spread_message(G, source_nodes, p, steps=10, regulation_set=None):
        informed = set(source_nodes)
        history = []

        for _ in range(steps):
            new_informed = set(informed)
            for node in informed:
                node_color = G.nodes[node]['color']
                for neighbor in G.neighbors(node):
                    if neighbor in informed:
                        continue
                    neighbor_color = G.nodes[neighbor]['color']
                    prob = p if node_color == neighbor_color else (1 - p)
                    if random.random() < prob:
                        new_informed.add(neighbor)

            if regulation_set:
                new_informed |= regulation_set

            informed = new_informed
            reds = sum(1 for n in informed if G.nodes[n]['color'] == 'red')
            yellows = sum(1 for n in informed if G.nodes[n]['color'] == 'yellow')
            history.append((reds, yellows))

        return history

    # =====================
    # Regulation functions
    # =====================
    def rlr_set(rho):
        return set(random.sample(list(G.nodes()), int(len(G) * rho)))

    def blue_only_set(rho):
        yellow_nodes = [n for n in G.nodes if G.nodes[n]['color'] == 'yellow']
        return set(random.sample(yellow_nodes, int(len(yellow_nodes) * rho))) if yellow_nodes else set()

    # =====================
    # Averaging Function
    # =====================
    def average_spread(G, source_selector, p, steps=10, regulation_set_generator=None, runs=100):
        reds_all = np.zeros(steps)
        yellows_all = np.zeros(steps)
        valid_runs = 0

        for _ in range(runs):
            source_nodes = source_selector()
            if not source_nodes:
                continue  # skip if no red nodes
            reg_set = regulation_set_generator() if regulation_set_generator else None
            history = spread_message(G, source_nodes, p, steps, regulation_set=reg_set)
            reds = np.array([r for r, y in history])
            yellows = np.array([y for r, y in history])
            reds_all += reds
            yellows_all += yellows
            valid_runs += 1

        if valid_runs == 0:
            return np.zeros(steps), np.zeros(steps)

        return reds_all / valid_runs, yellows_all / valid_runs

    def select_random_reds(k=1):
        red_nodes = [n for n in G.nodes if G.nodes[n]['color'] == 'red']
        if len(red_nodes) < k:
            return []
        return random.sample(red_nodes, k)

    # =====================
    # Run Scenarios
    # =====================
    scenarios = {
        "Strong No-Reg": (1.0, None),
        "p=0.7 No-Reg": (0.7, None),
        "Strong RLR(0.25)": (1.0, lambda: rlr_set(0.25)),
        "p=0.7 RLR(0.25)": (0.7, lambda: rlr_set(0.25)),
        "Strong BlueOnly(0.25)": (1.0, lambda: blue_only_set(0.25)),
        "p=0.7 BlueOnly(0.25)": (0.7, lambda: blue_only_set(0.25)),
    }

    average_results = {}
    for label, (p_val, reg_fn) in scenarios.items():
        reds_avg, yellows_avg = average_spread(G, select_random_reds, p=p_val,
                                               regulation_set_generator=reg_fn, steps=10, runs=100)
        average_results[label] = (reds_avg, yellows_avg)

    # =====================
    # Plot Results
    # =====================
    plt.figure(figsize=(10, 6))
    colors = {
        "p=0.7 No-Reg": "orange",
        "p=0.7 RLR(0.25)": "purple",
        "Strong No-Reg": "blue",
        "Strong RLR(0.25)": "green",
        "Strong BlueOnly(0.25)": "brown",
        "p=0.7 BlueOnly(0.25)": "red"
    }

    for label, (reds, yellows) in average_results.items():
        plt.plot(reds, label=f"{label} – Red", linestyle='--', color=colors[label])
        plt.plot(yellows, label=f"{label} – Yellow", linestyle='-', color=colors[label])

    plt.xlabel("Time Step")
    plt.ylabel("Average Informed Users")
    plt.title("Average Spread Over 100 Runs (with Regulation Support)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # =====================
    # Compute Echo Chamber Metrics
    # =====================
    print("\n=== Echo Chamber Metrics ===")
    for label, (reds_avg, yellows_avg) in average_results.items():
        alpha = reds_avg[-1] + yellows_avg[-1]
        phi_red = reds_avg[-1] / alpha if alpha > 0 else 0
        phi_yellow = yellows_avg[-1] / alpha if alpha > 0 else 0
        print(f"{label}:")
        print(f"  α (size) = {alpha:.2f}, ϕ_red = {phi_red:.2f}, ϕ_yellow = {phi_yellow:.2f}")

def count_zero_weight_edges(G):
    count = sum(1 for _, _, data in G.edges(data=True) if data.get('weight') == 0)
    print(f"Number of edges with weight 0: {count}")
    return count

def analyze_top10_pagerank_reciprocal(G, weight='weight', alpha=0.85, max_iter=500):

    # נירמול המשקלים לערך מוחלט
    for _, _, data in G.edges(data=True):
        if weight in data:
            data[weight] = abs(data[weight])

    try:
        pagerank_scores = nx.pagerank(G, weight=weight, alpha=alpha, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        print("⚠️ Power iteration did not converge with weights. Retrying without weights...")
        pagerank_scores = nx.pagerank(G, weight=None, alpha=alpha, max_iter=max_iter)

    # מיון הקודקודים לפי PageRank מהגבוה לנמוך
    sorted_scores = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

    top10 = sorted_scores[:10]

    print("Top 10 nodes by PageRank:")
    for node, score in top10:
        print(f"Node {node}: PageRank = {score:.5f}")

    print("\nCalculating reciprocal edge percentages...\n")

    for node, _ in top10:
        out_edges = list(G.out_edges(node, data=True))
        if not out_edges:
            pct = 0.0
        else:
            reciprocal_count = 0
            total_out = len(out_edges)
            for u, v, data in out_edges:
                w = data.get(weight, None)
                if w is None:
                    continue
                if G.has_edge(v, u):
                    w_back = G[v][u].get(weight, None)
                    if w_back == w:
                        reciprocal_count += 1
            pct = (reciprocal_count / total_out) * 100

        print(f"Node {node}: {pct:.2f}% of out-edges are reciprocated with the same weight")

def analyze_top10_pagerank_reciprocal_by_sign(G, weight='weight', alpha=0.85, max_iter=500):
    # שמירה על משקלים מקוריים לשם ניתוח החיוביים והשליליים
    G_abs = G.copy()
    for u, v, data in G_abs.edges(data=True):
        if weight in data:
            data[weight] = abs(data[weight])

    try:
        pagerank_scores = nx.pagerank(G_abs, weight=weight, alpha=alpha, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        print("⚠️ Power iteration did not converge with weights. Retrying without weights...")
        pagerank_scores = nx.pagerank(G_abs, weight=None, alpha=alpha, max_iter=max_iter)

    sorted_scores = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    top10 = sorted_scores[:10]

    print("Top 10 nodes by PageRank:")
    for node, score in top10:
        print(f"Node {node}: PageRank = {score:.5f}")

    print("\nCalculating reciprocal edge sign percentages...\n")

    for node, _ in top10:
        out_edges = list(G.out_edges(node, data=True))
        total_out = len(out_edges)
        pos_reciprocal = 0
        neg_reciprocal = 0

        for u, v, data in out_edges:
            w = data.get(weight, None)
            if w is None:
                continue
            if G.has_edge(v, u):
                w_back = G[v][u].get(weight, None)
                if w_back == w and w > 0:
                    pos_reciprocal += 1
                elif w_back == w and w < 0:
                    neg_reciprocal += 1

        if total_out > 0:
            pos_pct = (pos_reciprocal / total_out) * 100
            neg_pct = (neg_reciprocal / total_out) * 100
        else:
            pos_pct = neg_pct = 0.0

        print(f"Node {node}: {pos_pct:.2f}% positive reciprocal, {neg_pct:.2f}% negative reciprocal")

def degree_distribution_negative_graph(G):

    n_g = build_graph_weight_eq_to_target_only(G)

    degree_distributions(n_g, degree_type='in', title='max_connected_component_graph', x_min=1, x_max=2)
    degree_distributions(n_g, degree_type='out', title='max_connected_component_graph', x_min=1, x_max=2)
    degree_distributions(n_g, degree_type='total', title='max_connected_component_graph', x_min=1, x_max=3)

def mixed_preferential_attachment_three_colors(G_orig, m=3):
    def get_node_color(avg):
        if avg <= -2:
            return 'red'
        elif avg < 2:
            return 'yellow'
        else:
            return 'blue'

    def compute_node_colors(G):
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
            else:
                avg = 0
            node_avg_rating[node] = avg

        color_map = {node: get_node_color(avg) for node, avg in node_avg_rating.items()}
        return color_map

    def estimate_rho_matrix_three_colors(G, color_map):
        colors = ['red', 'blue', 'yellow']
        edge_counts = {src: {dst: 0 for dst in colors} for src in colors}
        total_counts = {src: 0 for src in colors}

        for u, v in G.edges():
            src_color = color_map.get(u)
            dst_color = color_map.get(v)
            if src_color and dst_color:
                edge_counts[src_color][dst_color] += 1
                total_counts[src_color] += 1

        rho = {}
        for src in colors:
            rho[src] = {}
            for dst in colors:
                total = total_counts[src]
                rho[src][dst] = edge_counts[src][dst] / total if total > 0 else 0
        return rho


    n = G_orig.number_of_nodes()

    # Step 1: מחשבים צבעים לכל הצמתים בגרף המקורי
    color_map_orig = compute_node_colors(G_orig)

    # Step 2: מחשבים את מטריצת ההסתברויות
    rho = estimate_rho_matrix_three_colors(G_orig, color_map_orig)

    # Step 3: מתחילים את הגרף החדש עם 3 צמתים - אחד מכל צבע
    G = nx.DiGraph()
    G.add_node(0, color='red')
    G.add_node(1, color='blue')
    G.add_node(2, color='yellow')
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)

    colors = ['red', 'blue', 'yellow']
    color_counts = {'red': 0, 'blue': 0, 'yellow': 0}
    n0 = 3

    for new_node in range(n0, n):
        # בוחרים צבע חדש לפי יחס הצבעים בגרף המקורי
        color_distribution = {
            'red': sum(1 for c in color_map_orig.values() if c == 'red'),
            'blue': sum(1 for c in color_map_orig.values() if c == 'blue'),
            'yellow': sum(1 for c in color_map_orig.values() if c == 'yellow'),
        }
        total = sum(color_distribution.values())
        color_probs = [color_distribution[c] / total for c in colors]

        new_color = random.choices(colors, weights=color_probs, k=1)[0]
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
            accept_prob = rho[new_color][candidate_color]

            if random.random() < accept_prob:
                targets.add(candidate)

        if len(targets) < m:
            print(f"⚠️ Node {new_node} connected to only {len(targets)} targets instead of {m} after {attempts} attempts.")

        for target in targets:
            G.add_edge(new_node, target)

    return G

def draw_graph_by_node_color(G, title="Mixed Preferential Attachment (3 Colors)"):
    # הגדרת צבעים מתאימים לכל צבע לוגי
    color_map = {
        'red': '#FF0000',
        'yellow': '#FFFF00',
        'blue': '#0000FF'
    }

    node_colors = [
        color_map.get(G.nodes[node].get('color', 'gray'), 'gray')
        for node in G.nodes()
    ]

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=40
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color='black',
        alpha=0.3,
        arrows=True,
        arrowsize=10,
        width=0.8
    )

    plt.title(title, fontsize=14)
    plt.axis('off')

    # מקרא
    for name, hex_color in color_map.items():
        plt.scatter([], [], c=hex_color, label=name, s=40)

    plt.legend(frameon=False, title="Node Color")
    plt.tight_layout()
    plt.show()

def plot_avg_rating_distribution(G, color=None, bins=30):
    """
    Plots a histogram of average incoming ratings for nodes.

    Parameters:
    - G: NetworkX DiGraph
    - color: One of 'red', 'yellow', 'blue', or None (all nodes)
    - bins: Number of histogram bins
    """

    def compute_average_rating(G):
        node_avg_rating = {}
        for node in G.nodes():
            in_edges = G.in_edges(node, data=True)
            if in_edges:
                avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
                node_avg_rating[node] = avg
            else:
                node_avg_rating[node] = 0
        return node_avg_rating

    avg_rating = compute_average_rating(G)

    # סינון לפי צבע אם ביקשו
    if color == 'red':
        values = [avg for avg in avg_rating.values() if avg <= -2]
        title = 'Red (avg ≤ -2)'
        plot_color = 'red'
    elif color == 'yellow':
        values = [avg for avg in avg_rating.values() if -2 < avg < 2]
        title = 'Yellow (-2 < avg < 2)'
        plot_color = 'gold'
    elif color == 'blue':
        values = [avg for avg in avg_rating.values() if avg >= 2]
        title = 'Blue (avg ≥ 2)'
        plot_color = 'blue'
    else:
        values = list(avg_rating.values())
        title = 'All Nodes'
        plot_color = 'gray'

    if not values:
        print(f"No nodes to plot for color: {color}")
        return

    # ציור ההיסטוגרמה
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color=plot_color, edgecolor='black')
    plt.xlabel('Average Incoming Rating')
    plt.ylabel('Number of Nodes')
    plt.title(f'Average Incoming Rating Distribution – {title}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# -----------NEW COLORED----------------------------------------------------------------------------

def compute_bayesian_trust_scores(G):
    node_scores = {}
    avg_rating_all = []

    for node in G.nodes():
        in_edges = G.in_edges(node, data=True)
        ratings = [data['weight'] for _, _, data in in_edges]
        if ratings:
            avg = sum(ratings) / len(ratings)
            avg_rating_all.extend(ratings)
            node_scores[node] = {'avg': avg, 'count': len(ratings)}
        else:
            node_scores[node] = {'avg': 0, 'count': 0}

    global_avg = sum(avg_rating_all) / len(avg_rating_all) if avg_rating_all else 0
    C = sum(G.in_degree(n) for n in G.nodes()) / G.number_of_nodes()

    bayesian_scores = {}
    for node, stats in node_scores.items():
        avg = stats['avg']
        count = stats['count']
        bayesian = (C * global_avg + count * avg) / (C + count) if (C + count) > 0 else global_avg
        bayesian_scores[node] = bayesian

    return bayesian_scores

def draw_graph_with_bayesian(G):

    def assign_colors(bayesian_scores):
        colors = []
        for node in G.nodes():
            score = bayesian_scores.get(node, 0)
            if score <= 0:
                colors.append('#FF0000')  # red
            elif 0 < score < 1.5:
                colors.append('#FFFF00')  # yellow
            else:
                colors.append('#0000FF')  # blue
        return colors

    # חישוב ציונים וצביעה
    bayesian_scores = compute_bayesian_trust_scores(G)
    node_colors = assign_colors(bayesian_scores)

    # ציור הגרף
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=40, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='black', alpha=0.3, arrows=True, arrowsize=10, width=0.8, ax=ax)

    ax.set_title("Bitcoin OTC Trust Graph – Colored by Bayesian Trust", fontsize=14, color='black')
    ax.axis("off")

    legend_labels = {
        '#FF0000': 'Bayesian ≤ 0',
        '#FFFF00': '0 < Bayesian < 1.5',
        '#0000FF': 'Bayesian ≥ 1.5'
    }

    for color, label in legend_labels.items():
        ax.scatter([], [], c=color, label=label, s=40)

    ax.legend(frameon=False, labelcolor='black')
    plt.tight_layout()
    plt.show()

def plot_bayesian_trust_distribution(G, color=None, bins=30):
    """
    Plots a histogram of Bayesian trust scores for nodes.

    Parameters:
    - G: NetworkX DiGraph
    - color: One of 'red', 'yellow', 'blue', or None (all nodes)
    - bins: Number of histogram bins
    """

    bayesian_scores = compute_bayesian_trust_scores(G)

    if color == 'red':
        values = [s for s in bayesian_scores.values() if s <= 0]
        title = 'Red (Bayesian ≤ 0)'
        plot_color = 'red'
    elif color == 'yellow':
        values = [s for s in bayesian_scores.values() if 0 < s < 1.5]
        title = 'Yellow (0 < Bayesian < 1.5)'
        plot_color = 'gold'
    elif color == 'blue':
        values = [s for s in bayesian_scores.values() if s >= 1.5]
        title = 'Blue (Bayesian ≥ 1.5)'
        plot_color = 'blue'
    else:
        values = list(bayesian_scores.values())
        title = 'All Nodes'
        plot_color = 'salmon'

    if not values:
        print(f"No nodes to plot for color: {color}")
        return

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color=plot_color, edgecolor='black')
    plt.xlabel('Bayesian Trust Score')
    plt.ylabel('Number of Nodes')
    plt.title(f'Bayesian Trust Score Distribution – {title}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def degree_distributions_bayesian(G, degree_type='in', title=' ', x_min=None, x_max=None, color=None):
    """
    Plot log-X histogram of degree distribution for nodes filtered by Bayesian Trust color.

    Parameters:
    - G: NetworkX DiGraph
    - degree_type: 'in', 'out', or 'total'
    - title: Plot title
    - x_min, x_max: Exponents for logspace binning
    - color: One of 'red', 'yellow', 'blue', or None (for all nodes)
    """


    bayesian_scores = compute_bayesian_trust_scores(G)

    if color == 'red':
        nodes = [n for n, score in bayesian_scores.items() if score <= 0]
        title += ' (Red: Bayesian ≤ 0)'
        plot_color = 'red'
    elif color == 'yellow':
        nodes = [n for n, score in bayesian_scores.items() if 0 < score < 1.5]
        title += ' (Yellow: 0 < Bayesian < 1.5)'
        plot_color = 'gold'
    elif color == 'blue':
        nodes = [n for n, score in bayesian_scores.items() if score >= 1.5]
        title += ' (Blue: Bayesian ≥ 1.5)'
        plot_color = 'blue'
    elif color is None:
        nodes = list(G.nodes())
        plot_color = 'gray'
    else:
        raise ValueError("color must be 'red', 'yellow', 'blue', or None")

    if degree_type == 'in':
        degrees = [G.in_degree(n) for n in nodes]
        label = 'In-Degree'
    elif degree_type == 'out':
        degrees = [G.out_degree(n) for n in nodes]
        label = 'Out-Degree'
    elif degree_type == 'total':
        degrees = [G.in_degree(n) + G.out_degree(n) for n in nodes]
        label = 'Total Degree'
    else:
        raise ValueError("degree_type must be 'in', 'out', or 'total'")

    degrees = [d for d in degrees if d > 0]

    if not degrees:
        print(f"No degrees to plot for color: {color}")
        return

    if x_min is None:
        x_min = math.log10(max(min(degrees), 1))
    if x_max is None:
        x_max = math.log10(max(degrees))

    bins = np.logspace(x_min, x_max, num=20)

    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=bins, color=plot_color, edgecolor='black')
    plt.xscale('log')
    plt.xlabel(f'{label} (log scale)')
    plt.ylabel('Frequency')
    plt.title(f'{label} Distribution (Log X-axis) {title}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def all_degree_distributions_bayesian(G):
    degree_distributions_bayesian(G, degree_type='in', title='max_connected_component_graph',x_min=0, x_max=4, color='blue')
    degree_distributions_bayesian(G, degree_type='out', title='max_connected_component_graph', x_min=0, x_max=4, color='blue')
    degree_distributions_bayesian(G, degree_type='total', title='max_connected_component_graph', x_min=0, x_max=4, color='blue')

    degree_distributions_bayesian(G, degree_type='in', title='max_connected_component_graph',x_min=0, x_max=4, color='red')
    degree_distributions_bayesian(G, degree_type='out', title='max_connected_component_graph', x_min=0, x_max=4, color='red')
    degree_distributions_bayesian(G, degree_type='total', title='max_connected_component_graph', x_min=0, x_max=4, color='red')

    degree_distributions_bayesian(G, degree_type='in', title='max_connected_component_graph',x_min=0, x_max=4, color='yellow')
    degree_distributions_bayesian(G, degree_type='out', title='max_connected_component_graph', x_min=0, x_max=4, color='yellow')
    degree_distributions_bayesian(G, degree_type='total', title='max_connected_component_graph', x_min=0, x_max=4, color='yellow')

    degree_distributions_bayesian(G, degree_type='in', title='max_connected_component_graph',x_min=0, x_max=4)
    degree_distributions_bayesian(G, degree_type='out', title='max_connected_component_graph', x_min=0, x_max=4)
    degree_distributions_bayesian(G, degree_type='total', title='max_connected_component_graph', x_min=0, x_max=4)

def power_law_no_binning_bayesian(G, show_fit=True, color=None):
    import warnings
    warnings.filterwarnings("ignore")

    bayesian_scores = compute_bayesian_trust_scores(G)

    if color == 'red':
        nodes = [n for n, s in bayesian_scores.items() if s <= 0]
        title = "Power-law (Red ≤ 0)"
    elif color == 'yellow':
        nodes = [n for n, s in bayesian_scores.items() if 0 < s < 1.5]
        title = "Power-law (Yellow 0 < s < 1.5)"
    elif color == 'blue':
        nodes = [n for n, s in bayesian_scores.items() if s >= 1.5]
        title = "Power-law (Blue ≥ 1.5)"
    else:
        nodes = list(G.nodes())
        title = "Power-law (All Nodes)"

    degrees = [G.degree(n) for n in nodes if G.degree(n) > 0]

    if not degrees:
        print(f"No degrees to plot for color: {color}")
        return

    hist, bin_edges = np.histogram(degrees, bins=range(1, max(degrees) + 2), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist, width=0.8, color='orange', edgecolor='black', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Vertex Degree")
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)

    if show_fit:
        try:
            fit = powerlaw.Fit(degrees, discrete=True)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            R, p = fit.distribution_compare('power_law', 'lognormal')

            print(f"⚙️ Raw Fit Results for {color or 'all'}:")
            print(f"  α (power-law exponent): {alpha:.3f}")
            print(f"  xmin: {xmin}")
            print(f"  Distribution is power-law? {'Yes' if p > 0.05 else 'No'} (p={p:.4f})")

            x_fit = np.linspace(xmin, max(degrees), 100)
            y_fit = (x_fit / xmin) ** (-alpha)
            y_fit *= hist[bin_centers >= xmin][0] / y_fit[0]
            plt.xlim(left=xmin)
            plt.plot(x_fit, y_fit, 'r--', label=f'Power-law fit (γ={alpha:.2f})')
            plt.legend()
        except ImportError:
            print("⚠️ כדי להשתמש באופציית fit, יש להתקין את הספרייה 'powerlaw'")

    plt.tight_layout()
    plt.show()

def power_law_binning_logarithm_bayesian(G, bins=20, show_fit=True, color=None):
    warnings.filterwarnings("ignore")

    bayesian_scores = compute_bayesian_trust_scores(G)

    if color == 'red':
        nodes = [n for n, s in bayesian_scores.items() if s <= 0]
        title = "Log-Binned Power-law (Red ≤ 0)"
    elif color == 'yellow':
        nodes = [n for n, s in bayesian_scores.items() if 0 < s < 1.5]
        title = "Log-Binned Power-law (Yellow 0 < s < 1.5)"
    elif color == 'blue':
        nodes = [n for n, s in bayesian_scores.items() if s >= 1.5]
        title = "Log-Binned Power-law (Blue ≥ 1.5)"
    else:
        nodes = list(G.nodes())
        title = "Log-Binned Power-law (All Nodes)"

    degrees = [G.degree(n) for n in nodes if G.degree(n) > 0]

    if not degrees:
        print(f"No degrees to plot for color: {color}")
        return

    min_deg = max(min(degrees), 1)
    max_deg = max(degrees)
    log_bins = np.logspace(np.log10(min_deg), np.log10(max_deg), bins)

    hist, bin_edges = np.histogram(degrees, bins=log_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, hist, width=np.diff(bin_edges), align='center',
            alpha=0.7, color='orange', edgecolor='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Vertex Degree")
    plt.ylabel("Probability")
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.4)

    if show_fit:
        try:
            fit = powerlaw.Fit(degrees, discrete=True)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            R, p = fit.distribution_compare('power_law', 'lognormal')

            print(f"⚙️ Log-Binned Fit Results for {color or 'all'}:")
            print(f"  α (power-law exponent): {alpha:.3f}")
            print(f"  xmin: {xmin}")
            print(f"  Distribution is power-law? {'Yes' if p > 0.05 else 'No'} (p={p:.4f})")

            x_fit = np.linspace(xmin, max_deg, 100)
            y_fit = (x_fit / xmin) ** (-alpha)
            y_fit *= hist[bin_centers >= xmin][0] / y_fit[0]
            plt.xlim(left=xmin)
            plt.plot(x_fit, y_fit, 'r--', label=f'Power-law fit (γ={alpha:.2f})')
            plt.legend()
        except ImportError:
            print("⚠️ כדי להשתמש באופציית fit, יש להתקין את הספרייה 'powerlaw'")

    plt.tight_layout()
    plt.show()

def split_graph_by_bayesian_color(G):

    def get_color(score):
        if score <= 0:
            return 'red'
        elif score < 1.5:
            return 'yellow'
        else:
            return 'blue'

    # חישוב ציוני Bayesian
    bayesian_scores = compute_bayesian_trust_scores(G)
    color_map = {node: get_color(score) for node, score in bayesian_scores.items()}

    # קיבוץ צמתים לפי צבע
    color_groups = {'red': set(), 'yellow': set(), 'blue': set()}
    for node, color in color_map.items():
        color_groups[color].add(node)

    # יצירת תתי גרפים
    subgraphs = {}
    for color, nodes in color_groups.items():
        edges = [(u, v, data) for u, v, data in G.edges(data=True)
                 if u in nodes and v in nodes]
        subG = nx.DiGraph()
        subG.add_nodes_from(nodes)
        subG.add_edges_from(edges)
        subgraphs[color] = subG

    # הדפסת מידע + ציור
    for color, subG in subgraphs.items():
        print(f"\n=== Subgraph for {color.upper()} nodes ===")
        print(f"Number of nodes: {subG.number_of_nodes()}")
        print(f"Number of edges: {subG.number_of_edges()}")

        pos = nx.spring_layout(subG, seed=42)
        plt.figure(figsize=(6, 5))
        nx.draw(subG, pos, with_labels=False, node_color=color,
                edge_color='gray', node_size=40, arrowsize=10)
        plt.title(f"{color.upper()} Subgraph")
        plt.tight_layout()
        plt.show()

def build_cross_color_edges_graph_bayesian(G):

    def get_color(score):
        if score <= 0:
            return 'red'
        elif score < 1.5:
            return 'yellow'
        else:
            return 'blue'

    # שלב 1: חישוב ציוני Bayesian וקביעת צבע לכל קודקוד
    bayesian_scores = compute_bayesian_trust_scores(G)
    color_map = {node: get_color(score) for node, score in bayesian_scores.items()}

    # שלב 2: מציאת כל הקשתות בין צבעים שונים
    cross_color_edges = []
    color_counts = defaultdict(int)
    weights = []

    for u, v, data in G.edges(data=True):
        if u in color_map and v in color_map:
            color_u = color_map[u]
            color_v = color_map[v]
            if color_u != color_v:
                cross_color_edges.append((u, v, data))
                weights.append(data['weight'])
                color_counts[(color_u, color_v)] += 1

    # שלב 3: בניית גרף חדש
    cross_color_G = nx.DiGraph()
    cross_color_G.add_edges_from(cross_color_edges)

    print(f"\n🎯 קשתות צבעוניות (מקודקודים בצבעים שונים): {len(cross_color_edges)}")
    print(f"🧮 מספר קודקודים בגרף החדש: {cross_color_G.number_of_nodes()}")
    if weights:
        print(f"📊 טווח משקלים: {min(weights)} עד {max(weights)}")
        print(f"📈 ממוצע משקלים: {sum(weights) / len(weights):.2f}")

    print("\n📚 פירוט לפי סוגי צבעים:")
    for (src_color, dst_color), count in color_counts.items():
        print(f"  {src_color.upper()} → {dst_color.upper()}: {count} קשתות")

    # שלב 4: ציור
    pos = nx.spring_layout(cross_color_G, seed=42)
    node_colors = [color_map[n] for n in cross_color_G.nodes()]
    nx.draw(cross_color_G, pos,
            node_color=node_colors,
            edge_color='gray',
            node_size=40,
            arrowsize=10)
    plt.title("גרף של קשתות בין צבעים שונים (לפי Bayesian Trust)")
    plt.tight_layout()
    plt.show()

    return cross_color_G

def all_power_law_bayesian(G):

    power_law_no_binning_bayesian(G, show_fit=True, color='blue')
    power_law_binning_logarithm_bayesian(G, bins=20, show_fit=True, color='blue')

    power_law_no_binning_bayesian(G, show_fit=True, color='red')
    power_law_binning_logarithm_bayesian(G, bins=20, show_fit=True, color='red')

    power_law_no_binning_bayesian(G, show_fit=True, color='yellow')
    power_law_binning_logarithm_bayesian(G, bins=20, show_fit=True, color='yellow')

    power_law_no_binning_bayesian(G, show_fit=True)
    power_law_binning_logarithm_bayesian(G, bins=20, show_fit=True)

def mixed_preferential_attachment_three_colors_bayesian(G_orig, m=3):
    def get_node_color(bayesian_score):
        if bayesian_score <= 0:
            return 'red'
        elif bayesian_score < 1.5:
            return 'yellow'
        else:
            return 'blue'

    def compute_node_colors(G):
        bayesian_scores = compute_bayesian_trust_scores(G)
        color_map = {node: get_node_color(score) for node, score in bayesian_scores.items()}
        return color_map

    def estimate_rho_matrix_three_colors(G, color_map):
        colors = ['red', 'blue', 'yellow']
        edge_counts = {src: {dst: 0 for dst in colors} for src in colors}
        total_counts = {src: 0 for src in colors}

        for u, v in G.edges():
            src_color = color_map.get(u)
            dst_color = color_map.get(v)
            if src_color and dst_color:
                edge_counts[src_color][dst_color] += 1
                total_counts[src_color] += 1

        rho = {}
        for src in colors:
            rho[src] = {}
            for dst in colors:
                total = total_counts[src]
                rho[src][dst] = edge_counts[src][dst] / total if total > 0 else 0
        return rho

    n = G_orig.number_of_nodes()

    # Step 1: Compute colors based on Bayesian score
    color_map_orig = compute_node_colors(G_orig)

    # Print color distribution
    color_counts = {
        'red': sum(1 for c in color_map_orig.values() if c == 'red'),
        'blue': sum(1 for c in color_map_orig.values() if c == 'blue'),
        'yellow': sum(1 for c in color_map_orig.values() if c == 'yellow'),
    }
    total_nodes = sum(color_counts.values())
    print("🎨 Color Distribution in Original Graph:")
    for color in ['red', 'blue', 'yellow']:
        prob = color_counts[color] / total_nodes
        print(f"  {color.capitalize()}: {color_counts[color]} nodes ({prob:.2%})")

    # Step 2: Compute rho matrix
    rho = estimate_rho_matrix_three_colors(G_orig, color_map_orig)

    print("\n🔢 Estimated Rho Matrix (Connection Probabilities):")
    print(f"{'':>10} {'Red':>10} {'Blue':>10} {'Yellow':>10}")
    for src in ['red', 'blue', 'yellow']:
        row = f"{src.capitalize():>10}"
        for dst in ['red', 'blue', 'yellow']:
            row += f" {rho[src][dst]:>10.3f}"
        print(row)

    # Step 3: Initialize the new graph
    G = nx.DiGraph()
    G.add_node(0, color='red')
    G.add_node(1, color='blue')
    G.add_node(2, color='yellow')
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 0)

    colors = ['red', 'blue', 'yellow']
    n0 = 3
    incomplete_nodes = 0

    for new_node in range(n0, n):
        # Sample color for new node
        total = sum(color_counts.values())
        color_probs = [color_counts[c] / total for c in colors]
        new_color = random.choices(colors, weights=color_probs, k=1)[0]
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
            accept_prob = rho[new_color][candidate_color]

            if random.random() < accept_prob:
                targets.add(candidate)

        if len(targets) < m:
            print(f"⚠️ Node {new_node} connected to only {len(targets)} targets instead of {m} after {attempts} attempts.")
            incomplete_nodes += 1

        for target in targets:
            G.add_edge(new_node, target)

    # Final summary
    print("\n📊 Final Graph Summary:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Nodes with < {m} connections: {incomplete_nodes}")

    return G

def draw_graph_by_node_color_bayesian(G, title="Graph by Bayesian Node Colors"):

    def get_node_color(bayesian_score):
        if bayesian_score <= 0:
            return 'red'
        elif bayesian_score < 1.5:
            return 'yellow'
        else:
            return 'blue'

    # צבעים גרפיים לתצוגה
    display_colors = {
        'red': '#FF0000',
        'yellow': '#FFFF00',
        'blue': '#0000FF'
    }

    # מחשבים צביעה לפי Bayesian אם לא קיימת
    if 'color' not in next(iter(G.nodes(data=True)))[1]:
        bayesian_scores = compute_bayesian_trust_scores(G)
        color_map = {node: get_node_color(score) for node, score in bayesian_scores.items()}
    else:
        color_map = {node: G.nodes[node]['color'] for node in G.nodes()}

    node_colors = [
        display_colors.get(color_map.get(node, 'gray'), 'gray') for node in G.nodes()
    ]

    # ציור
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=40
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color='black',
        alpha=0.3,
        arrows=True,
        arrowsize=10,
        width=0.8
    )

    plt.title(title, fontsize=14)
    plt.axis('off')

    # מקרא
    for name, hex_color in display_colors.items():
        plt.scatter([], [], c=hex_color, label=name, s=40)

    plt.legend(frameon=False, title="Node Color")
    plt.tight_layout()
    plt.show()

# def get_bayesian_color(score):
#     if score <= 0:
#         return 'red'
#     elif score < 1.5:
#         return 'yellow'
#     else:
#         return 'blue'
#
# def filter_nodes_by_bayesian_color(G, bayesian_scores, color):
#     if color == 'red':
#         return [n for n, s in bayesian_scores.items() if s <= 0]
#     elif color == 'yellow':
#         return [n for n, s in bayesian_scores.items() if 0 < s < 1.5]
#     elif color == 'blue':
#         return [n for n, s in bayesian_scores.items() if s >= 1.5]
#     else:
#         return list(G.nodes())
#
# def get_color_properties(color):
#     if color == 'red':
#         return 'red', 'Bayesian ≤ 0'
#     elif color == 'yellow':
#         return 'gold', '0 < Bayesian < 1.5'
#     elif color == 'blue':
#         return 'blue', 'Bayesian ≥ 1.5'
#     else:
#         return 'gray', 'All Nodes'

if __name__ == '__main__':


    G = build_original_graph()
    max_connected_component_graph = build_max_connected_component_graph(G)


    # draw_graph(G)
    # draw_graph(max_connected_component_graph)

    # mod, comm_dict, comms = directed_graph_modularity_with_communities(max_connected_component_graph)
    # print(f"Total number of communities: {len(comms)}")
    #
    # for i, comm in enumerate(comms, start=1):
    #     print(f"Community {i} has {len(comm)} nodes")

    # compute_symmetric_edge_percentage_by_sign(max_connected_component_graph)


    # new_G = remove_node_from_graph(max_connected_component_graph, 3744)



    # print(directed_graph_modularity(max_connected_component_graph))


    # coloredG = build_cross_color_edges_graph(max_connected_component_graph)
    # degree_distributions(coloredG, degree_type='in', title='max_connected_component_graph',x_min=2, x_max=3)
    # degree_distributions(coloredG, degree_type='out', title='max_connected_component_graph', x_min=2, x_max=3)
    # degree_distributions(coloredG, degree_type='total', title='max_connected_component_graph', x_min=2, x_max=3)

    # min_rating, max_rating = min_max_rating(max_connected_component_graph)
    # print("min rating: ", min_rating, "\nmax rating: ", min_rating)

    # centrality(max_connected_component_graph)
    # draw_graph(max_connected_component_graph)

    # draw_rating_histogram(max_connected_component_graph)

    # all_power_law(max_connected_component_graph)
    # all_degree_distributions(max_connected_component_graph)


    # compare_centrality(max_connected_component_graph)
    # density(max_connected_component_graph)
    # small_world(max_connected_component_graph)
    # overlap(max_connected_component_graph, "Overlap and Weight", "neighborhood_overlap.png")

    # avg_dist = average_distance_directed(max_connected_component_graph)

    # run for ever
    # create_orders_and_draw(G)
    # calculate_directed_triangle_percentage(max_connected_component_graph)


    # ----------------------------------------------------------


    # calculate_symmetric_edge_percentage(max_connected_component_graph)
    # split_graph_by_color(max_connected_component_graph)

    # check_symmetric_edge_percentages_all_colors(max_connected_component_graph)
    # percent, equal_count, total = compute_equal_in_out_degree_percentage(max_connected_component_graph)
    # print(f"Percentage of nodes with equal in-degree and out-degree: {percent:.2f}% "
    #       f"({equal_count} out of {total})")

    # spreading_mode(max_connected_component_graph)
    # count_zero_weight_edges(max_connected_component_graph)

    # analyze_top10_pagerank_reciprocal(max_connected_component_graph)
    # analyze_top10_pagerank_reciprocal_by_sign(max_connected_component_graph)
    # directed_graph_modularity(max_connected_component_graph)
    # analyze_top10_pagerank_reciprocal_by_sign(max_connected_component_graph)

    # pre = build_preferential_attachment_model(max_connected_component_graph)
    #
    # spreading_mode(max_connected_component_graph)
    #
    # spreading_mode(pre)

    # degree_distribution_negative_graph(max_connected_component_graph)


    # centrality(max_connected_component_graph)




    # mpa = mixed_preferential_attachment_three_colors(max_connected_component_graph)
    # draw_graph_by_node_color(mpa)
    # mod, comm_dict, comms = directed_graph_modularity_with_communities(new_G)
    # plot_avg_rating_distribution(max_connected_component_graph)
    # plot_avg_rating_distribution(max_connected_component_graph, color='red')
    # plot_avg_rating_distribution(max_connected_component_graph, color='yellow')
    # plot_avg_rating_distribution(max_connected_component_graph, color='blue')


#     -----------------NEW COLORED---------------------------

    # draw_graph_with_bayesian(max_connected_component_graph)

    # plot_bayesian_trust_distribution(max_connected_component_graph)
    # plot_bayesian_trust_distribution(max_connected_component_graph, color='red')
    # plot_bayesian_trust_distribution(max_connected_component_graph, color='yellow')
    # plot_bayesian_trust_distribution(max_connected_component_graph, color='blue')

    # all_degree_distributions_bayesian(max_connected_component_graph)

    # split_graph_by_bayesian_color(max_connected_component_graph)

    # all_power_law_bayesian(max_connected_component_graph)
    # new_G = mixed_preferential_attachment_three_colors_bayesian(max_connected_component_graph, m=3)
    # draw_graph_by_node_color(new_G, title="Bayesian Mixed Preferential Attachment")

    # build_cross_color_edges_graph_bayesian(max_connected_component_graph)

    n = build_graph_weight_eq_to_target_only(max_connected_component_graph)
    calculate_symmetric_edge_percentage(n)


    n1 = build_graph_weight_eq_to_target_only(max_connected_component_graph, 10)
    calculate_symmetric_edge_percentage(n1)