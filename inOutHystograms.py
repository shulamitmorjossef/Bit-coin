import os
import networkx as nx
import matplotlib.pyplot as plt

# path to the folder containing the files
folder_path = "soc-sign-bitcoinotc.csv"  # ודא שהקובץ נמצא באותה תיקייה או ציין נתיב מלא

G = nx.DiGraph()

# יצירת הגרף מהקובץ (שם הקובץ הוא תוכן השורה עצמה)
for filename in os.listdir(folder_path):
    try:
        parts = filename.strip().split(',')
        if len(parts) != 4:
            continue
        source = int(parts[0])
        target = int(parts[1])
        rating = int(parts[2])
        time = float(parts[3])
        G.add_edge(source, target, weight=rating, time=time)
    except Exception as e:
        print(f"Skipping file {filename} due to error: {e}")

print(f"G: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# מציאת הרכיב הקשיר החזק ביותר
max_strong_component = max(nx.strongly_connected_components(G), key=len)
subgraph = G.subgraph(max_strong_component).copy()

print(f"Max Strongly Connected Component: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

# חישוב דרגות כניסה ויציאה
in_degrees = dict(subgraph.in_degree())
out_degrees = dict(subgraph.out_degree())

# נרמול הדרגות
max_in = max(in_degrees.values())
max_out = max(out_degrees.values())

normalized_in = [deg / max_in if max_in != 0 else 0 for deg in in_degrees.values()]
normalized_out = [deg / max_out if max_out != 0 else 0 for deg in out_degrees.values()]


# draw
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
