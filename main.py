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


def compute_fixed_colors_by_ranges(G, node_avg_rating):
    colors = []
    for node in G.nodes():
        avg = node_avg_rating.get(node, 0)
        if avg < -2:
            colors.append('#0000FF')  # כחול
        elif avg > 2:
            colors.append('#FFFF00')  # צהוב
        else:
            colors.append('#FF0000')  # אדום
    return colors

def draw_graph(G, node_colors, node_avg_rating, min_rating, max_rating):
    # ביטול מצב dark mode
    plt.style.use('default')

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Colormap חדש: כחול → אדום → צהוב
    blue_red_yellow = mcolors.LinearSegmentedColormap.from_list(
        'custom_bry',
        ['blue', 'red', 'yellow']
    )

    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors,
        node_size=40, cmap=blue_red_yellow, ax=ax
    )

    # קשתות בגוון כחול-אדום
    nx.draw_networkx_edges(
        G, pos, edge_color='darkblue',
        alpha=0.3, arrows=False, ax=ax
    )

    ax.set_title("Bitcoin OTC Trust Graph – Colored by Avg. Incoming Rating",
                 fontsize=14, color='black')
    ax.axis("off")

    # Colorbar מותאם
    sm = plt.cm.ScalarMappable(cmap=blue_red_yellow)
    sm.set_array([min_rating, max_rating])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Average Incoming Rating", color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')

    plt.tight_layout()
    plt.show()

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

def min_max_rating(node_avg_rating):

    # Normalize ratings for color mapping
    min_rating = min(node_avg_rating.values())
    max_rating = max(node_avg_rating.values())

    return min_rating, max_rating

def normalize_rating(val, min_rating, max_rating):
    return (val - min_rating) / (max_rating - min_rating) if max_rating != min_rating else 0.5

def compute_node_colors(node_avg_rating, G,  min_rating, max_rating):
    node_colors = [normalize_rating(node_avg_rating[n],  min_rating, max_rating) for n in G.nodes()]
    return node_colors

def draw_graph(G, node_colors, node_avg_rating, min_rating, max_rating):
    # Dark mode style
    plt.style.use('dark_background')

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')

    # Custom colormap: deep purple to pink, no yellow
    deep_purple_pink = mcolors.LinearSegmentedColormap.from_list(
        'custom_purple_pink',
        ['#3f007d', '#8e24aa', '#d81b60']  # dark violet → purple → deep pink
    )

    nodes = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors,
        node_size=40, cmap=deep_purple_pink, ax=ax
    )

    # Edges: טורקיז מעושן (עמוק אך חי)
    nx.draw_networkx_edges(
        G, pos, edge_color='#20b2aa',
        alpha=0.4, arrows=False, ax=ax
    )

    ax.set_title("Bitcoin OTC Trust Graph – Colored by Avg. Incoming Rating",
                 fontsize=14, color='white')
    ax.axis("off")

    # Colorbar matching the custom colormap
    sm = plt.cm.ScalarMappable(cmap=deep_purple_pink)
    sm.set_array([min_rating, max_rating])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Average Incoming Rating", color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    plt.show()

def plot_rating_histogram(node_avg_rating):
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

# def compute_fixed_colors_by_ranges(G, node_avg_rating):
#     colors = []
#     for node in G.nodes():
#         avg = node_avg_rating.get(node, 0)
#         if avg < -2:
#             colors.append('#9b4d96')  # סגול כהה
#         elif avg > 2:
#             colors.append('#ffb6c1')  # ורוד בהיר (לבן ורדרד)
#         else:
#             colors.append('#d32f7f')  # ורוד-סגול
#     return colors

# def draw_graph_by_fixed_colors(G, node_colors):
#     plt.style.use('dark_background')
#     pos = nx.spring_layout(G, seed=42)
#
#     fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
#
#     # צביעת הקודקודים לפי הצבעים שנבחרו
#     nx.draw_networkx_nodes(
#         G, pos, node_color=node_colors,
#         node_size=40, ax=ax
#     )
#
#     # צביעת הקשתות
#     nx.draw_networkx_edges(
#         G, pos, edge_color='#20b2aa',
#         alpha=0.4, arrows=False, ax=ax
#     )
#
#     # כותרת
#     ax.set_title("Bitcoin OTC Trust Graph – Colored by Rating Ranges",
#                  fontsize=14, color='white')
#     ax.axis("off")
#
#     # יצירת מקרא
#     legend_labels = {
#         '#9b4d96': '< -2',  # סגול כהה
#         '#d32f7f': '= 2',  # ורוד-סגול
#         '#ffb6c1': '> 2'  # ורוד בהיר (לבן/ורדרד)
#     }
#
#     for color, label in legend_labels.items():
#         ax.scatter([], [], c=color, label=label, s=40)
#
#     ax.legend(frameon=False, labelcolor='white')
#
#     plt.tight_layout()
#     plt.show()

def degree_histogram(G):
    # אוסף את דרגות הקלט והפלט
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # חישוב ההתפלגות של כל דרגה (מספר הצמתים עם דרגה מסוימת)
    in_degree_counts = Counter(in_degrees.values())
    out_degree_counts = Counter(out_degrees.values())

    return in_degree_counts, out_degree_counts

def draw_degree_histogram(in_degree_counts, out_degree_counts):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # היסטוגרמה - In-Degree
    ax1.bar(in_degree_counts.keys(), in_degree_counts.values(), color='lightcoral', edgecolor='black')
    ax1.set_title("In-Degree Distribution", fontsize=14)
    ax1.set_xlabel("In-Degree")
    ax1.set_ylabel("Number of Nodes")

    # היסטוגרמה - Out-Degree
    ax2.bar(out_degree_counts.keys(), out_degree_counts.values(), color='mediumseagreen', edgecolor='black')
    ax2.set_title("Out-Degree Distribution", fontsize=14)
    ax2.set_xlabel("Out-Degree")
    ax2.set_ylabel("Number of Nodes")

    plt.tight_layout()
    plt.show()

def plot_distribution(degrees, title, filename, color, max_degree=None):
    # Count degree frequencies
    count = Counter(degrees)
    count = {k: v for k, v in count.items() if k > 0}  # remove zero degrees (log undefined)
    if max_degree:
        count = {k: v for k, v in count.items() if k <= max_degree}

    total = sum(count.values())
    degs = np.array(sorted(count.keys()))
    freqs = np.array([count[d] / total for d in degs])

    # Take log of degrees
    log_degs = np.log(degs)

    # Fit a linear regression on log(degree) vs frequency
    coeffs = np.polyfit(log_degs, freqs, 1)  # linear fit: y = a*log(x) + b
    a, b = coeffs

    # Generate smooth curve for fitted log function
    x_smooth = np.linspace(min(degs), max(degs), 500)
    y_smooth = a * np.log(x_smooth) + b

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    plt.scatter(degs, freqs, color=color, edgecolors='white', label='Data')
    plt.plot(x_smooth, y_smooth, color='gold', linewidth=2, label='Log fit')

    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Relative Frequency")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

def plot_normalized_degree_distributions_fixed(G):
    # Collect the in-degree and out-degree values
    in_degrees = [deg for _, deg in G.in_degree()]
    out_degrees = [deg for _, deg in G.out_degree()]

    # Plot the normalized degree distributions for both in-degree and out-degree
    plot_distribution(in_degrees, "Normalized In-Degree Distribution", "normalized_in_degree_distribution.png",
                      'royalblue') # , max_degree=50
    plot_distribution(out_degrees, "Normalized Out-Degree Distribution", "normalized_out_degree_distribution.png",
                      'tomato') # , max_degree=30

def compute_degree_centrality(G):

    in_deg_centrality = nx.in_degree_centrality(G)
    out_deg_centrality = nx.out_degree_centrality(G)
    return in_deg_centrality, out_deg_centrality

def plot_log_smoothed_centrality(centrality_values, title, color, edge_color):
    # הסרת ערכים אפסיים
    values = np.array(list(centrality_values.values()))
    values = values[values > 0]

    # חישוב תדירויות
    unique_vals, counts = np.unique(values, return_counts=True)
    freqs = counts / counts.sum()

    # רגרסיה לוגריתמית
    log_x = np.log(unique_vals)
    coeffs = np.polyfit(log_x, freqs, 1)
    a, b = coeffs

    # יצירת עקומה חלקה
    x_smooth = np.linspace(min(unique_vals), max(unique_vals), 500)
    y_smooth = a * np.log(x_smooth) + b

    # ציור
    plt.style.use('dark_background')
    plt.figure(figsize=(6, 4))
    plt.scatter(unique_vals, freqs, color=color, edgecolor=edge_color, label='Data')
    plt.plot(x_smooth, y_smooth, color='white', linestyle='--', label=f'y = {a:.2f} log(x) + {b:.2f}')

    plt.title(title, fontsize=14, color='white')
    plt.xlabel("Degree Centrality", color='white')
    plt.ylabel("Relative Frequency", color='white')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def compute_degree_centrality(G):

    in_deg_centrality = nx.in_degree_centrality(G)
    out_deg_centrality = nx.out_degree_centrality(G)
    return in_deg_centrality, out_deg_centrality

def compute_and_plot_degree_centrality(in_deg_centrality, out_deg_centrality):
    # Dark mode settings
    plt.style.use('dark_background')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='black')
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')

    # In-Degree Centrality - turquoise tone
    ax1.hist(in_deg_centrality.values(), bins=20, color='#40E0D0', edgecolor='cyan')
    ax1.set_title("In-Degree Centrality Distribution", fontsize=14, color='white')
    ax1.set_xlabel("In-Degree Centrality", color='white')
    ax1.set_ylabel("Number of Nodes", color='white')
    ax1.tick_params(colors='white')

    # Out-Degree Centrality - light blue tone
    ax2.hist(out_deg_centrality.values(), bins=20, color='#00BFFF', edgecolor='deepskyblue')
    ax2.set_title("Out-Degree Centrality Distribution", fontsize=14, color='white')
    ax2.set_xlabel("Out-Degree Centrality", color='white')
    ax2.set_ylabel("Number of Nodes", color='white')
    ax2.tick_params(colors='white')

    plt.tight_layout()
    plt.show()

def compute_and_plot_smoothed_degree_centrality(in_deg_centrality, out_deg_centrality):
    plot_log_smoothed_centrality(in_deg_centrality, "In-Degree Centrality (Log-Smoothed)", '#40E0D0', 'cyan')
    plot_log_smoothed_centrality(out_deg_centrality, "Out-Degree Centrality (Log-Smoothed)", '#00BFFF', 'deepskyblue')

def plot_centrality(centrality, label):
    # Dark mode
    plt.style.use('dark_background')

    plt.figure(figsize=(8, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')

    plt.hist(centrality.values(), bins=20, color='#6246dc', edgecolor='cyan')  # lightseagreen shade
    plt.title(f'{label} Centrality Distribution', fontsize=14, color='white')
    plt.xlabel(f'{label} Centrality', color='white')
    plt.ylabel("Number of Nodes", color='white')
    plt.xticks(color='white')
    plt.yticks(color='white')

    plt.tight_layout()
    plt.show()

def compute_closeness_centrality(G):
    return nx.closeness_centrality(G)

def compute_betweenness_centrality(G):
    return nx.betweenness_centrality(G)

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

def draw_original_graph():

    G = build_original_graph()

    nodes= compute_average_rating(G)

    mini, maxi = min_max_rating(nodes)

    colors = compute_node_colors(nodes, G, mini, maxi)

    draw_graph(G, colors, nodes, mini, maxi)

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

def triangle_ratio(graph):
    """
    מחשבת את אחוז המשולשים מתוך כלל השלשות האפשריות בגרף נתון.

    Parameters:
        graph (networkx.Graph): גרף לא מכוון

    Returns:
        tuple: (total_triangles, possible_triplets, ratio_percent)
    """

    # סופרים כמה משולשים יש בכל קודקוד
    triangles_dict = nx.triangles(graph)

    # סכום כל המשולשים חלקי 3 (כל משולש נספר 3 פעמים)
    total_triangles = sum(triangles_dict.values()) // 3

    # שלשות אפשריות (כל זוג קשתות שיוצאות מאותו צומת)
    possible_triplets = sum(deg * (deg - 1) / 2 for _, deg in graph.degree())

    # חישוב אחוז
    if possible_triplets == 0:
        ratio_percent = 0.0
    else:
        ratio_percent = 100 * total_triangles / possible_triplets

    print(f"Total triangles: {total_triangles}")
    print(f"Possible triplets: {int(possible_triplets)}")
    print(f"Triangle ratio: {ratio_percent:.2f}%")

    return total_triangles, possible_triplets, ratio_percent

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

def count_directed_cycles(G):
    if not G.is_directed():
        raise ValueError("הגרף חייב להיות מכוון")

    cycles = list(nx.simple_cycles(G))
    print(f"מספר המעגלים בגרף: {len(cycles)}")
    return cycles

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

def create_orders_and_draw(G):
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

# # Function to plot the neighborhood overlap vs. edge weight with a trend line

def plot_neighborhood_overlap(G, title, filename):
    overlaps, weights = calculate_neighborhood_overlap(G)  # Get overlap and weight data

    # Calculate the linear regression trend line (slope and intercept)
    slope, intercept, _, _, _ = linregress(weights, overlaps)
    trend_y = [slope * w + intercept for w in weights]  # Calculate the trend line values

    # Create the plot with white background
    plt.style.use('default')  # Use default (white) style
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')  # Set axis background to white

    # Scatter plot of data points with colorful points
    plt.scatter(weights, overlaps, alpha=0.7, color='blue', edgecolors='green', linewidths=0.3, label='Edges')

    # Plot the trend line with red color
    plt.plot(weights, trend_y, linestyle='--', color='red', linewidth=2, label='Trend Line')

    # Set titles and labels with black color (since background is white)
    plt.title(title, fontsize=14, weight='bold', color='black')
    plt.xlabel("Weight", fontsize=12, color='black')
    plt.ylabel("Overlap", fontsize=12, color='black')

    # Set the ticks color to black for contrast
    plt.xticks(color='black')
    plt.yticks(color='black')

    # Enable grid with dashed lines and light alpha
    plt.grid(True, linestyle='--', alpha=0.6)

    # Display legend
    plt.legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot to a file with high resolution
    plt.savefig(filename, dpi=300)

    # Show the plot
    plt.show()


# def plot_neighborhood_overlap(G, title, filename):
#     overlaps, weights = calculate_neighborhood_overlap(G)  # Get overlap and weight data
#
#     # Calculate the linear regression trend line (slope and intercept)
#     slope, intercept, _, _, _ = linregress(weights, overlaps)
#     trend_y = [slope * w + intercept for w in weights]  # Calculate the trend line values
#
#     # Create the plot with black background
#     plt.style.use('dark_background')  # Set dark background for the plot
#     plt.figure(figsize=(10, 6), facecolor='white')
#     ax = plt.gca()
#     ax.set_facecolor('black')  # Set axis background to black
#
#     # Scatter plot of data points with glowing blue and purple shades
#     plt.scatter(weights, overlaps, alpha=0.7, color='#6246dc', edgecolors='cyan', linewidths=0.3, label='Edges')
#
#     # Plot the trend line with a glowing red effect
#     plt.plot(weights, trend_y, linestyle='--', color='magenta', linewidth=2, label='Trend Line')
#
#     # Set titles and labels with white color
#     plt.title(title, fontsize=14, weight='bold', color='white')
#     plt.xlabel("Weight", fontsize=12, color='white')
#     plt.ylabel("Overlap", fontsize=12, color='white')
#
#     # Set the ticks color to white for contrast
#     plt.xticks(color='white')
#     plt.yticks(color='white')
#
#     # Enable grid with dashed lines and light alpha
#     plt.grid(True, linestyle='--', alpha=0.4)
#
#     # Display legend
#     plt.legend()
#
#     # Adjust layout for better spacing
#     plt.tight_layout()
#
#     # Save the plot to a file with high resolution
#     plt.savefig(filename, dpi=300)
#
#     # Show the plot
#     plt.show()
#
#





if __name__ == '__main__':

    G = build_original_graph()

    max_connected_component_graph = build_max_connected_component_graph(G)

    node_avg_rating = compute_average_rating(max_connected_component_graph)

    node_colors_fixed = compute_fixed_colors_by_ranges(max_connected_component_graph, node_avg_rating)
    #
    # plot_rating_histogram(node_avg_rating)
    #
    # min_rating, max_rating = min_max_rating(node_avg_rating)
    #
    # node_colors = compute_node_colors(node_avg_rating, max_connected_component_graph, min_rating, max_rating)
    #
    draw_graph_by_fixed_colors(max_connected_component_graph, node_colors_fixed)
    #
    # node_avg_rating = compute_average_rating(max_connected_component_graph)
    #
    # min_rating, max_rating = min_max_rating(node_avg_rating)
    #
    # node_colors = compute_node_colors(node_avg_rating, max_connected_component_graph, min_rating, max_rating)
    #
    # normalized_in, normalized_out = degree_histogram(max_connected_component_graph)
    #
    # compute_and_plot_smoothed_degree_centrality(*compute_degree_centrality(max_connected_component_graph))
    #
    # normalized_in, normalized_out = degree_histogram(max_connected_component_graph)
    #
    # draw_degree_histogram(normalized_in, normalized_out)
    #
    # plot_normalized_degree_distributions_fixed(max_connected_component_graph)
    #
    # compute_and_plot_degree_centrality(*compute_degree_centrality(max_connected_component_graph))
    #
    # plot_centrality(compute_closeness_centrality(max_connected_component_graph), "closeness")
    #
    # plot_centrality(compute_betweenness_centrality(max_connected_component_graph), "betweeness")
    #
    # compare_centrality(max_connected_component_grapSh)
    #
    # density(max_connected_component_graph)
    #
    # small_world(max_connected_component_graph)
    #
    # create_orders_and_draw()

    # create_orders_and_draw(G)

    plot_neighborhood_overlap(max_connected_component_graph, "Overlap and Weight", "neighborhood_overlap.png")




