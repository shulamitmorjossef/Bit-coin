import os
import networkx as nx
import matplotlib.pyplot as plt

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


print(f"G: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

strongly_components_G = list(nx.strongly_connected_components(G))
print(f"Number of strongly connected components in G: {len(strongly_components_G)}")

max_strong_component = max(nx.strongly_connected_components(G), key=len)
print("Max strongly connected component size: " + str(len(max_strong_component)))


# --- Degree Distribution Histogram ---



# חשב דרגת כניסה, יציאה, ודרגה כוללת
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())
total_degrees = {node: in_degrees[node] + out_degrees[node] for node in G.nodes()}

# נרמול של הדרגות בין 0 ל-1
def normalize_dict(d):
    min_val = min(d.values())
    max_val = max(d.values())
    return {k: (v - min_val) / (max_val - min_val) if max_val != min_val else 0.5 for k, v in d.items()}

norm_in_degrees = normalize_dict(in_degrees)
norm_out_degrees = normalize_dict(out_degrees)
norm_total_degrees = normalize_dict(total_degrees)

# ציור היסטוגרמות של הדרגות
fig, axs = plt.subplots(1, 3, figsize=(20, 5))

axs[0].hist(norm_in_degrees.values(), bins=20, color='orange', edgecolor='black')
axs[0].set_title("Normalized In-Degree Distribution")
axs[0].set_xlabel("Normalized In-Degree")
axs[0].set_ylabel("Number of Nodes")

axs[1].hist(norm_out_degrees.values(), bins=20, color='green', edgecolor='black')
axs[1].set_title("Normalized Out-Degree Distribution")
axs[1].set_xlabel("Normalized Out-Degree")
axs[1].set_ylabel("Number of Nodes")

axs[2].hist(norm_total_degrees.values(), bins=20, color='purple', edgecolor='black')
axs[2].set_title("Normalized Total Degree Distribution")
axs[2].set_xlabel("Normalized Total Degree")
axs[2].set_ylabel("Number of Nodes")

plt.tight_layout()
plt.show()

# הצגת 10 הצמתים עם דרגת כניסה הכי גבוהה
top_in = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 in-degrees:")
for node, deg in top_in:
    print(f"Node {node}: in-degree {deg}")

# אותו דבר עבור דרגת יציאה
top_out = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 out-degrees:")
for node, deg in top_out:
    print(f"Node {node}: out-degree {deg}")


# diameter
# שלב 1: מציאת הרכיב החלש הכי גדול
largest_weak_component = max(nx.weakly_connected_components(G), key=len)

# שלב 2: תת-גרף רק על הרכיב הזה
subgraph = G.subgraph(largest_weak_component).copy()

# שלב 3: הפוך לגרף לא מכוון כדי לא להיתקל בבעיות חוסר קשירות
undirected_subgraph = subgraph.to_undirected()

# שלב 4: חישוב הקוטר
try:
    diameter = nx.diameter(undirected_subgraph)
    print(f"Diameter of the largest weakly connected component in G: {diameter}")
except nx.NetworkXError as e:
    print(f"Could not compute diameter: {e}")



# Compute average incoming rating per node
node_avg_rating = {}
for node in G.nodes():
    in_edges = G.in_edges(node, data=True)
    if in_edges:
        avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
        node_avg_rating[node] = avg
    else:
        node_avg_rating[node] = 0

# Normalize ratings for color mapping
min_rating = min(node_avg_rating.values())
max_rating = max(node_avg_rating.values())
print(f"Min rating: {min_rating}")
print(f"Max rating: {max_rating}")

def normalize(val):
    return (val - min_rating) / (max_rating - min_rating) if max_rating != min_rating else 0.5

node_colors = [normalize(node_avg_rating[n]) for n in G.nodes()]

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

# import math
#
# print("\n--- Small-World Property Check ---")
#
# # Convert the directed graph to undirected for small-world analysis
# G_undirected = G.to_undirected()
#
# # Step 1: Check connectivity and select the largest connected component
# if nx.is_connected(G_undirected):
#     G_lcc = G_undirected
#     print("The graph is connected.")
# else:
#     print("The graph is not fully connected.")
#     largest_cc = max(nx.connected_components(G_undirected), key=len)
#     G_lcc = G_undirected.subgraph(largest_cc).copy()
#     print(f"Largest connected component has {G_lcc.number_of_nodes()} nodes and {G_lcc.number_of_edges()} edges.")
#
# # Step 2: Compute the average shortest path length
# print("Calculating average shortest path length...")
# try:
#     avg_path_length = nx.average_shortest_path_length(G_lcc)
#     print(f"Average shortest path length: {avg_path_length:.4f}")
# except Exception as e:
#     print(f"Could not compute average shortest path length: {e}")
#     avg_path_length = None
#
# # Step 3: Compute the average clustering coefficient
# try:
#     avg_clustering = nx.average_clustering(G_lcc)
#     print(f"Average clustering coefficient: {avg_clustering:.4f}")
# except Exception as e:
#     print(f"Could not compute average clustering coefficient: {e}")
#     avg_clustering = None
#
# # Step 4: Compute the graph diameter
# try:
#     diameter = nx.diameter(G_lcc)
#     print(f"Graph diameter: {diameter}")
# except Exception as e:
#     print(f"Could not compute diameter: {e}")
#     diameter = None
#
# # Step 5: Compute log(N) as baseline for path length
# expected_log_n = math.log(G_lcc.number_of_nodes())
# print(f"Expected log(N) ≈ {expected_log_n:.4f}, where N = {G_lcc.number_of_nodes()}")
#
# # Step 6: Interpretation of small-world properties
# print("\n--- Interpretation ---")
# print("Small-world networks usually have:")
# print("1. Short average path length close to log(N)")
# print("2. High clustering coefficient (much higher than random graphs)")
#
# if avg_path_length and avg_clustering:
#     if avg_path_length <= expected_log_n * 1.5 and avg_clustering > 0.1:
#         print("✅ The graph likely exhibits small-world properties.")
#     else:
#         print("❌ The graph likely does NOT exhibit small-world properties.")
# else:
#     print("⚠️ Could not fully evaluate small-world properties due to missing data.")

