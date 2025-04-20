# import os
# import networkx as nx
# import matplotlib.pyplot as plt
#
# # path to the extracted folder
# folder_path = "soc-sign-bitcoinotc.csv"  # or full path if needed
#
# G = nx.DiGraph()
#
# # Traverse all files in the folder (each file name is a record)
# for filename in os.listdir(folder_path):
#     try:
#         parts = filename.strip().split(',')
#         if len(parts) != 4:
#             continue  # skip malformed names
#         source = int(parts[0])
#         target = int(parts[1])
#         rating = int(parts[2])
#         time = float(parts[3])
#         G.add_edge(source, target, weight=rating, time=time)
#     except Exception as e:
#         print(f"Skipping file {filename} due to error: {e}")
#
# print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
#
# # Compute average incoming rating per node
# node_avg_rating = {}
# for node in G.nodes():
#     in_edges = G.in_edges(node, data=True)
#     if in_edges:
#         avg = sum(data['weight'] for _, _, data in in_edges) / len(in_edges)
#         node_avg_rating[node] = avg
#     else:
#         node_avg_rating[node] = 0
#
# # Normalize ratings for color mapping
# min_rating = min(node_avg_rating.values())
# print(f"Min rating: {min_rating}")
# max_rating = max(node_avg_rating.values())
# print(f"Max rating: {max_rating}")
#
#
# # todo: update normelize and add histogram
# def normalize(val):
#     return (val - min_rating) / (max_rating - min_rating) if max_rating != min_rating else 0.5
#
# node_colors = [normalize(node_avg_rating[n]) for n in G.nodes()]
#
# pos = nx.spring_layout(G, seed=42)
# plt.figure(figsize=(12, 10))
# nx.draw_networkx_nodes(G, pos, node_color=node_colors,
#                        node_size=40, cmap=plt.cm.RdYlGn)
# nx.draw_networkx_edges(G, pos, alpha=0.05, arrows=False)
# plt.title("Bitcoin OTC Trust Subgraph – Colored by Avg. Incoming Rating", fontsize=14)
# plt.axis("off")
# plt.show()
#
#
#
#




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

print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

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
