import os
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import math

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

# max_strong_component = max(nx.strongly_connected_components(G), key=len)
# strongly_components_G = list(nx.strongly_connected_components(G))
# print(f"Number of strongly connected components in G: {len(strongly_components_G)}")
# print(f"MAX Stringly C COMPONENT: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

def draw_graph(G, node_colors, node_avg_rating, min_rating, max_rating):
    # Dark mode style
    plt.style.use('dark_background')

    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')

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

# def draw_graph(G, node_colors, node_avg_rating, min_rating, max_rating):
#     # Dark mode style
#     plt.style.use('dark_background')
#
#     pos = nx.spring_layout(G, seed=42)
#
#     fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
#
#     # קודקודים בגווני מג’נטה/סגול עמוקים (plasma נותן עומק ודרמה)
#     nodes = nx.draw_networkx_nodes(
#         G, pos, node_color=node_colors,
#         node_size=40, cmap=plt.cm.plasma, ax=ax
#     )
#
#     # צלעות בטורקיז כהה (DarkCyan)
#     nx.draw_networkx_edges(
#         G, pos, edge_color='#008b8b',
#         alpha=0.4, arrows=False, ax=ax
#     )
#
#     ax.set_title("Bitcoin OTC Trust Graph – Colored by Avg. Incoming Rating",
#                  fontsize=14, color='white')
#     ax.axis("off")
#
#     # Colorbar תואם לגווני הקודקודים
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
#     sm.set_array([min_rating, max_rating])
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label("Average Incoming Rating", color='white')
#     cbar.ax.yaxis.set_tick_params(color='white')
#     plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
#
#     plt.tight_layout()
#     plt.show()


# def draw_graph(G, node_colors, node_avg_rating, min_rating, max_rating):
#     # Dark mode style
#     plt.style.use('dark_background')
#
#     pos = nx.spring_layout(G, seed=42)
#
#     fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
#
#     # קודקודים בגווני ורוד-סגול (Purples)
#     nodes = nx.draw_networkx_nodes(
#         G, pos, node_color=node_colors,
#         node_size=40, cmap=plt.cm.Purples, ax=ax
#     )
#
#     # צלעות בצבע טורקיז
#     nx.draw_networkx_edges(
#         G, pos, edge_color='#40E0D0',  # טורקיז (Turquoise)
#         alpha=0.3, arrows=False, ax=ax
#     )
#
#     ax.set_title("Bitcoin OTC Trust Graph – Colored by Avg. Incoming Rating",
#                  fontsize=14, color='white')
#     ax.axis("off")
#
#     # Colorbar תואם לצבעי הקודקודים
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.Purples)
#     sm.set_array([min_rating, max_rating])
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label("Average Incoming Rating", color='white')
#     cbar.ax.yaxis.set_tick_params(color='white')
#     plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
#
#     plt.tight_layout()
#     plt.show()


# def draw_graph(G, node_colors, node_avg_rating, min_rating, max_rating):
#     # Dark mode style
#     plt.style.use('dark_background')
#
#     pos = nx.spring_layout(G, seed=42)
#
#     fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
#
#     nodes = nx.draw_networkx_nodes(
#         G, pos, node_color=node_colors,
#         node_size=40, cmap=plt.cm.RdYlGn, ax=ax
#     )
#     nx.draw_networkx_edges(G, pos, alpha=0.05, arrows=False, ax=ax)
#     ax.set_title("Bitcoin OTC Trust Graph – Colored by Avg. Incoming Rating", fontsize=14, color='white')
#     ax.axis("off")
#
#     # Colorbar
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
#     sm.set_array([min_rating, max_rating])
#     cbar = plt.colorbar(sm, ax=ax)
#     cbar.set_label("Average Incoming Rating", color='white')
#     cbar.ax.yaxis.set_tick_params(color='white')
#     plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
#
#     plt.tight_layout()
#     plt.show()

# def draw_graph(G, node_colors, node_avg_rating, min_rating, max_rating):
#
#     # Layout and plotting
#     pos = nx.spring_layout(G, seed=42)
#
#     fig, ax = plt.subplots(1, 2, figsize=(18, 8))
#
#     # Graph drawing with color
#     nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
#                            node_size=40, cmap=plt.cm.RdYlGn, ax=ax[0])
#     nx.draw_networkx_edges(G, pos, alpha=0.05, arrows=False, ax=ax[0])
#     ax[0].set_title("Bitcoin OTC Trust Graph – Colored by Avg. Incoming Rating", fontsize=14)
#     ax[0].axis("off")
#
#     # Add colorbar for interpretation
#     sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
#     sm.set_array([min_rating, max_rating])
#     cbar = plt.colorbar(sm, ax=ax[0])
#     cbar.set_label("Average Incoming Rating")
#
#     # Histogram of avg incoming ratings
#     # ax[1].hist(node_avg_rating.values(), bins=20, color='skyblue', edgecolor='black')
#     # ax[1].set_title("Histogram of Average Incoming Ratings", fontsize=14)
#     # ax[1].set_xlabel("Avg. Rating")
#     # ax[1].set_ylabel("Number of Nodes")
#     #
#     # plt.tight_layout()
#     plt.show()

# def degree_histogram(G):
#     # אוסף את דרגות הקלט והפלט
#     in_degrees = dict(G.in_degree())
#     out_degrees = dict(G.out_degree())
#
#     # חישוב הדרגות המקסימליות
#     # max_in = max(in_degrees.values())
#     # max_out = max(out_degrees.values())
#
#     # חישוב התדרים היחסיים (נירמול) לכל דרגה
#     total_in = sum(in_degrees.values())  # סך כל הצמתים עם דרגות קלט
#     total_out = sum(out_degrees.values())  # סך כל הצמתים עם דרגות פלט
#
#     # חישוב ההתפלגות היחסית של כל דרגה
#     normalized_in = [deg / total_in for deg in in_degrees.values()] if total_in != 0 else [0] * len(in_degrees)
#     normalized_out = [deg / total_out for deg in out_degrees.values()] if total_out != 0 else [0] * len(out_degrees)
#
#     return normalized_in, normalized_out
#
# def draw_degree_histogram(normalized_in, normalized_out):
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#
#     # Histogram - In-Degree
#     ax1.hist(normalized_in, bins=20, color='lightcoral', edgecolor='black')
#     ax1.set_title("Normalized In-Degree Distribution", fontsize=14)
#     ax1.set_xlabel("Normalized In-Degree")
#     ax1.set_ylabel("Number of Nodes")
#
#     # Histogram - Out-Degree
#     ax2.hist(normalized_out, bins=20, color='mediumseagreen', edgecolor='black')
#     ax2.set_title("Normalized Out-Degree Distribution", fontsize=14)
#     ax2.set_xlabel("Normalized Out-Degree")
#     ax2.set_ylabel("Number of Nodes")
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
    # Count the frequency of each degree
    count = Counter(degrees)

    # Apply a filter if a maximum degree is specified
    if max_degree:
        count = {k: v for k, v in count.items() if k <= max_degree}

    # Calculate the total number of occurrences
    total = sum(count.values())

    # Sort degrees and calculate the relative frequency
    degs = sorted(count.keys())
    freqs = [count[d] / total for d in degs]

    # Create a bar plot for the degree distribution
    plt.figure(figsize=(10, 6))
    plt.bar(degs, freqs, width=0.8, color=color, edgecolor='black', align='center')
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Relative Frequency")

    # Set the x-axis ticks based on the number of degrees
    plt.xticks(degs if len(degs) < 30 else range(0, max(degs) + 1, max(1, max(degs) // 15)))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save the plot to a file and close it
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
## TODO
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

if __name__ == '__main__':

    # draw_original_graph()

    G = build_original_graph()

    max_connected_component_graph = build_max_connected_component_graph(G)

    # node_avg_rating = compute_average_rating(max_connected_component_graph)
    #
    # min_rating, max_rating = min_max_rating(node_avg_rating)
    #
    # node_colors = compute_node_colors(node_avg_rating, max_connected_component_graph, min_rating, max_rating)
    #
    # draw_graph(max_connected_component_graph, node_colors, node_avg_rating, min_rating, max_rating)
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
    # compare_centrality(max_connected_component_graph)
    #
    # density(max_connected_component_graph)
    #
    # small_world(max_connected_component_graph)

    calculate_directed_triangle_percentage(max_connected_component_graph)

    count_directed_cycles(max_connected_component_graph)
