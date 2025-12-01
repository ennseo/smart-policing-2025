import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

file_path_edge = './result/edge_score_gaussian_mean.csv'
file_path_cluster = './result/hdbscan_cluster_result.csv'

df_edge = pd.read_csv(file_path_edge)
df_cluster = pd.read_csv(file_path_cluster)

# 그래프 재구성 (인접 노드 정보를 위해 필요)
G = nx.Graph()
pos = {}

for index, row in df_edge.iterrows():
    u = row['INT_ID_FROM']
    v = row['INT_ID_TO'] 
    weight = row['edge_score_scaled']
    G.add_edge(u, v, weight=weight)
    
    if u not in pos:
        pos[u] = (row['LON_FROM'], row['LAT_FROM'])
    if v not in pos:
        pos[v] = (row['LON_TO'], row['LAT_TO'])

# 클러스터 파일에서 점수와 클러스터 정보 직접 로드 및 병합
node_data = df_cluster[['CL_NODE_ID', 'node_score', 'cluster']].copy()
node_scores = pd.Series(node_data.node_score.values, index=node_data.CL_NODE_ID).to_dict() # 딕셔너리 형태로

# 클러스터별 노드 선택 및 마스킹
all_selected_nodes = []
cluster_ids = node_data['cluster'].unique()

for cluster_id in cluster_ids:
    if cluster_id == -1:
        print(f"--- Skipping Noise Cluster {-1} ---")
        continue
    
    if pd.isna(cluster_id):
        print(f"--- Skipping NaN Cluster ID ---")
        continue

    print(f"--- Processing Cluster {int(cluster_id)} ---")
    
    # 해당 클러스터에 속하는 노드들만 선택
    cluster_nodes_df = node_data[node_data['cluster'] == cluster_id]
    cluster_node_list = cluster_nodes_df['CL_NODE_ID'].tolist()
    
    # 해당 클러스터 노드로 이루어진 서브 그래프 추출
    subgraph = G.subgraph(cluster_node_list)
    
    # 서브 그래프 내의 노드 점수를 내림차순으로 정렬
    subgraph_scores = {node: node_scores.get(node, 0) for node in subgraph.nodes()}
    sorted_subgraph_nodes = sorted(subgraph_scores.items(), key=lambda x: x[1], reverse=True)
    
    selected_nodes = []
    masked_nodes = set()
    
    # 클러스터 내부 마스킹 알고리즘
    while len(masked_nodes) < len(subgraph.nodes()):
        best_node = None
        for node, score in sorted_subgraph_nodes:
            if node not in masked_nodes:
                best_node = node
                break
        
        if best_node is None:
            break
            
        selected_nodes.append(best_node)
        masked_nodes.add(best_node)
        
        # 1단계 이웃 노드 마스킹
        for neighbor in subgraph.neighbors(best_node):
            masked_nodes.add(neighbor)

            # 2단계 이웃 노드 마스킹
            for neighbor_of_neighbor in subgraph.neighbors(neighbor):
                masked_nodes.add(neighbor_of_neighbor)

                # 3단계 이웃 노드 마스킹
                for neighbor_of_neighbor_of_neighbor in subgraph.neighbors(neighbor_of_neighbor):
                    masked_nodes.add(neighbor_of_neighbor_of_neighbor)

    print(f"Cluster {int(cluster_id)}: Selected {len(selected_nodes)} nodes out of {len(subgraph.nodes())}")
    all_selected_nodes.extend(selected_nodes)


print(f"\nTotal Selected Nodes (3-step Masking): {len(all_selected_nodes)}")

final_selected_df = node_data[node_data['CL_NODE_ID'].isin(all_selected_nodes)].copy()
final_selected_df['Is_Selected'] = True
final_selected_df.to_csv('./result/masking_3step.csv', index=False)

# 시각화
plt.figure(figsize=(15, 15))

# 배경: 모든 노드와 엣지
nx.draw_networkx_nodes(
    G, 
    pos, 
    node_size=10, 
    node_color='lightgray', 
    alpha=0.5,
    label='All Nodes'
)
nx.draw_networkx_edges(
    G, 
    pos, 
    width=0.5, 
    edge_color='lightgray', 
    alpha=0.3
)

# 선택된 노드(노드 크기를 점수에 비례하게 설정)
"""
if all_selected_nodes:
    selected_scores = [node_scores[n] for n in all_selected_nodes]
    
    min_score = min(selected_scores) if selected_scores else 0
    max_score = max(selected_scores) if selected_scores else 1
    
    if max_score == min_score:
        node_sizes = [50] * len(selected_scores)
    else:
        node_sizes = [10 + (score - min_score) / (max_score - min_score) * 90 for score in selected_scores]

    nodes = nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist=all_selected_nodes,
        node_size=node_sizes, # 크기 조정된 노드 크기 사용
        node_color=selected_scores, 
        cmap=plt.cm.RdYlGn, # 선택된 노드를 점수에 따라 색칠
        edgecolors='black',
        linewidths=0.5,
        alpha=0.9
    )

    # 컬러바
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=min_score, vmax=max_score))
    sm.set_array([])
    plt.colorbar(sm, label='Node Score (Cluster Representative)', ax=plt.gca(), fraction=0.03, pad=0.04)
else:
    print("No nodes were selected after filtering.")


plt.title('Selected Nodes Colored by Score (3-step Masking)')
plt.axis('off')
plt.savefig('./result_pic/masking_colered by score_3step.png')
"""



# 선택된 노드(노드 크기 일정, 클러스터 색상 구분)

if all_selected_nodes:
    selected_clusters = final_selected_df.set_index('CL_NODE_ID').loc[all_selected_nodes]['cluster'].tolist()
    
    nodes = nx.draw_networkx_nodes(
        G, 
        pos, 
        nodelist=all_selected_nodes,
        node_size=50,
        node_color=selected_clusters, 
        cmap=plt.cm.get_cmap('tab20', len(set(selected_clusters))), # 클러스터별 색상 구분
        edgecolors='black',
        linewidths=0.5,
        alpha=0.9
    )

    # 컬러바
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('tab20', len(set(selected_clusters))), norm=plt.Normalize(vmin=min(selected_clusters), vmax=max(selected_clusters)))
    sm.set_array([])
    plt.colorbar(sm, label='Cluster ID', ax=plt.gca(), fraction=0.03, pad=0.04)

    plt.title('Selected Nodes Colored by Cluster ID (3-step Masking)')
    plt.axis('off')
    plt.savefig('./result_pic/masking_colered by cluster id_3step.png')

