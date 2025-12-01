import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

file_path = "./result/edge_score_gaussian_mean.csv"
df = pd.read_csv(file_path)

G = nx.Graph()

pos = {}

for index, row in df.iterrows():
    u = row['INT_ID_FROM']
    v = row['INT_ID_TO']
    weight = row['edge_score_scaled']
    G.add_edge(u, v, weight=weight)
    
    if u not in pos:
        pos[u] = (row['LAT_FROM'], row['LON_FROM'])
    if v not in pos:
        pos[v] = (row['LAT_TO'], row['LON_TO'])

result_data = []

for node in G.nodes():
    edges = G.edges(node, data=True)
    if edges:
        avg_score = sum(data['weight'] for _, _, data in edges) / len(edges)
    else:
        avg_score = 0
    
    if node in pos:
        x, y = pos[node]
    else:
        x, y = (None, None)
        
    result_data.append({
        'CL_NODE_ID': node,
        'node_score': avg_score,
        'X': x,
        'Y': y
    })

node_scores_df = pd.DataFrame(result_data)
node_scores_df.to_csv('./result/intersection_score.csv', index=False)
