import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import math
import numpy as np
import random
from scipy.stats import linregress
from collections import defaultdict


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

    # ax.legend(frameon=False, labelcolor='black')

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
    degree_distributions(G, degree_type='in', title='max_connected_component_graph',x_min=0, x_max=3, color='blue')
    degree_distributions(G, degree_type='out', title='max_connected_component_graph', x_min=0, x_max=3, color='blue')
    degree_distributions(G, degree_type='total', title='max_connected_component_graph', x_min=0, x_max=3, color='blue')

    degree_distributions(G, degree_type='in', title='max_connected_component_graph',x_min=0, x_max=3, color='red')
    degree_distributions(G, degree_type='out', title='max_connected_component_graph', x_min=0, x_max=3, color='red')
    degree_distributions(G, degree_type='total', title='max_connected_component_graph', x_min=0, x_max=3, color='red')

    degree_distributions(G, degree_type='in', title='max_connected_component_graph',x_min=0, x_max=3, color='yellow')
    degree_distributions(G, degree_type='out', title='max_connected_component_graph', x_min=0, x_max=3, color='yellow')
    degree_distributions(G, degree_type='total', title='max_connected_component_graph', x_min=0, x_max=3, color='yellow')

    degree_distributions(G, degree_type='in', title='max_connected_component_graph',x_min=0, x_max=3)
    degree_distributions(G, degree_type='out', title='max_connected_component_graph', x_min=0, x_max=3)
    degree_distributions(G, degree_type='total', title='max_connected_component_graph', x_min=0, x_max=3)

def centrality(G):
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    def draw_centrality(centrality, title, min_exp, max_exp):

        values = np.array(list(centrality.values()))
        values = values[values > 0]

        bins = np.logspace(min_exp, max_exp, num=20)

        plt.figure(figsize=(12, 6), facecolor='black')
        plt.hist(values, bins=bins, color='blueviolet', edgecolor='cyan')
        plt.xscale('log')
        plt.xlabel(f"{title} Centrality (log scale)", color='white')
        plt.ylabel("Number of Nodes", color='white')
        plt.title(f"Distribution of {title} Centrality", color='white')
        plt.tick_params(colors='white')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

    draw_centrality(closeness_centrality, "Closeness", -1, 0)
    draw_centrality(betweenness_centrality, "Betweenness", -5, -1)

def compare_centrality(G):
    # Calculate centrality measures
    in_deg_centrality = nx.in_degree_centrality(G)
    out_deg_centrality = nx.out_degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # Get the top 10 nodes with the highest values for each centrality measure
    top_in = sorted(in_deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_out = sorted(out_deg_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

    # Print the top 10 nodes for each centrality measure, sorted by centrality value (highest to lowest)
    print("\nTop 10 In-Degree Centrality Nodes:")
    print([node for node, _ in top_in])

    print("\nTop 10 Out-Degree Centrality Nodes:")
    print([node for node, _ in top_out])

    print("\nTop 10 Betweenness Centrality Nodes:")
    print([node for node, _ in top_betweenness])

    print("\nTop 10 Closeness Centrality Nodes:")
    print([node for node, _ in top_closeness])

    # Plot the Venn diagram showing overlaps between the top 10 nodes for each centrality measure
    plt.figure(figsize=(5, 3))
    venn = venn3([set(node for node, _ in top_out),
                  set(node for node, _ in top_betweenness),
                  set(node for node, _ in top_closeness)],
                 set_labels=("Out-Degree", "Betweenness", "Closeness"))

    # Define the colors for the Venn diagram with stronger contrast
    venn.get_patch_by_id('100').set_facecolor('#1f77b4')  # Dark blue
    venn.get_patch_by_id('010').set_facecolor('#ff7f0e')  # Orange
    venn.get_patch_by_id('001').set_facecolor('#2ca02c')  # Green
    venn.get_patch_by_id('110').set_facecolor('#9467bd')  # Purple
    venn.get_patch_by_id('101').set_facecolor('#8c564b')  # Brown
    venn.get_patch_by_id('011').set_facecolor('#e377c2')  # Pink
    venn.get_patch_by_id('111').set_facecolor('#7f7f7f')  # Grey

    # Remove the numbers inside the Venn diagram
    for v in venn.subset_labels:
        v.set_text('')

    plt.title("Top 10 Centrality Nodes Overlap")
    plt.tight_layout()
    plt.show()

def power_law_no_binning(G, show_fit=False, color=None):
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
            import powerlaw
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

def power_law_binning_logarithm(G, bins=20, show_fit=False, color=None):
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
            import powerlaw
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
# TODO check num of edges w weight = 0
def compute_symmetric_edge_percentages_by_color(G):
    # TODO check sum of edges sum to to total edges
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

if __name__ == '__main__':

    G = build_original_graph()
    max_connected_component_graph = build_max_connected_component_graph(G)

    coloredG = build_cross_color_edges_graph(max_connected_component_graph)
    degree_distributions(coloredG, degree_type='in', title='max_connected_component_graph',x_min=2, x_max=3)
    degree_distributions(coloredG, degree_type='out', title='max_connected_component_graph', x_min=2, x_max=3)
    degree_distributions(coloredG, degree_type='total', title='max_connected_component_graph', x_min=2, x_max=3)

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


    # pre = build_preferential_attachment_model(max_connected_component_graph)
    #
    # spreading_mode(max_connected_component_graph)
    #
    # spreading_mode(pre)
