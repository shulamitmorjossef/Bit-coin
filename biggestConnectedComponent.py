import os
import networkx as nx
import matplotlib.pyplot as plt

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

def degree_histogram(G):

    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    max_in = max(in_degrees.values())
    max_out = max(out_degrees.values())

    normalized_in = [deg / max_in if max_in != 0 else 0 for deg in in_degrees.values()]
    normalized_out = [deg / max_out if max_out != 0 else 0 for deg in out_degrees.values()]

    return normalized_in, normalized_out

def draw_degree_histogram(normalized_in, normalized_out):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram - In-Degree
    ax1.hist(normalized_in, bins=20, color='lightcoral', edgecolor='black')
    ax1.set_title("Normalized In-Degree Distribution", fontsize=14)
    ax1.set_xlabel("Normalized In-Degree")
    ax1.set_ylabel("Number of Nodes")

    # Histogram - Out-Degree
    ax2.hist(normalized_out, bins=20, color='mediumseagreen', edgecolor='black')
    ax2.set_title("Normalized Out-Degree Distribution", fontsize=14)
    ax2.set_xlabel("Normalized Out-Degree")
    ax2.set_ylabel("Number of Nodes")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    G = build_original_graph()

    max_connected_component_graph = build_max_connected_component_graph(G)

    node_avg_rating = compute_average_rating(max_connected_component_graph)

    min_rating, max_rating = min_max_rating(node_avg_rating)

    node_colors = compute_node_colors(node_avg_rating, max_connected_component_graph, min_rating, max_rating)

    draw_graph(max_connected_component_graph, node_colors, node_avg_rating, min_rating, max_rating)

    normalized_in, normalized_out = degree_histogram(max_connected_component_graph)

    draw_degree_histogram(normalized_in, normalized_out)



