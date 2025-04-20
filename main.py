import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.ticker import MaxNLocator
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



# creating a sub graph for max strongly connected component
def build_max_connected_component_graph(G):

    max_strong_component = max(nx.strongly_connected_components(G), key=len)
    max_strong_component_subgraph = G.subgraph(max_strong_component).copy()
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
    print(f"Min rating: {min_rating}")
    print(f"Max rating: {max_rating}")

    return min_rating, max_rating

def normalize_rating(val, min_rating, max_rating):
    return (val - min_rating) / (max_rating - min_rating) if max_rating != min_rating else 0.5

def compute_node_colors(node_avg_rating, G,  min_rating, max_rating):
    node_colors = [normalize_rating(node_avg_rating[n],  min_rating, max_rating) for n in G.nodes()]
    return node_colors


def draw_graph(G, node_colors, node_avg_rating, min_rating, max_rating):

    # Layout and plotting
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(1, 2, figsize=(18, 8))

    # Graph drawing with color
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=40, cmap=plt.cm.RdYlGn, ax=ax[0])
    nx.draw_networkx_edges(G, pos, alpha=0.05, arrows=False, ax=ax[0])
    ax[0].set_title("Bitcoin OTC Trust Graph – Colored by Avg. Incoming Rating", fontsize=14)
    ax[0].axis("off")

    # Add colorbar for interpretation
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn)
    sm.set_array([min_rating, max_rating])
    cbar = plt.colorbar(sm, ax=ax[0])
    cbar.set_label("Average Incoming Rating")

    # Histogram of avg incoming ratings
    ax[1].hist(node_avg_rating.values(), bins=20, color='skyblue', edgecolor='black')
    ax[1].set_title("Histogram of Average Incoming Ratings", fontsize=14)
    ax[1].set_xlabel("Avg. Rating")
    ax[1].set_ylabel("Number of Nodes")

    plt.tight_layout()
    plt.show()

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


if __name__ == '__main__':

    G = build_original_graph()

    max_connected_component_graph = build_max_connected_component_graph(G)

    node_avg_rating = compute_average_rating(max_connected_component_graph)

    min_rating, max_rating = min_max_rating(node_avg_rating)

    node_colors = compute_node_colors(node_avg_rating, max_connected_component_graph, min_rating, max_rating)

    # draw_graph(max_connected_component_graph, node_colors, node_avg_rating, min_rating, max_rating)

    # normalized_in, normalized_out = degree_histogram(max_connected_component_graph)

    # draw_degree_histogram(normalized_in, normalized_out)

    plot_normalized_degree_distributions_fixed(max_connected_component_graph)























"""


def plot_distribution(degrees, title, filename, color, max_degree=None):
    from matplotlib.ticker import MaxNLocator
    count = Counter(degrees)

    # חיתוך לפי דרגה מקסימלית אם ביקשו
    if max_degree:
        count = {k: v for k, v in count.items() if k <= max_degree}

    total = sum(count.values())
    degs = sorted(count.keys())
    freqs = [count[d] / total for d in degs]

    plt.figure(figsize=(10, 6))
    plt.bar(degs, freqs, width=0.8, color='yellow', edgecolor='black', align='center')
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Relative Frequency")
    plt.xticks(degs if len(degs) < 30 else range(0, max(degs)+1, max(1, max(degs)//15)))  # לא יותר מדי X ticks
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

in_degrees = [deg for _, deg in G_sub.in_degree()]
out_degrees = [deg for _, deg in G_sub.out_degree()]

plot_distribution(in_degrees, "Normalized In-Degree Distribution", "normalized_in_degree_distribution.png", 'royalblue', max_degree=50)
plot_distribution(out_degrees, "Normalized Out-Degree Distribution", "normalized_out_degree_distribution.png", 'tomato', max_degree=30)


"""
