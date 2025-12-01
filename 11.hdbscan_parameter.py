# 클러스터 결과 .png 확인 용도

import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import hdbscan
import warnings
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", category=FutureWarning)

def plot_clusters(df, out_image, n_clusters, noise_ratio, min_cluster_size, min_samples):
    labels = df["cluster"].values
    unique_labels = sorted(set(labels))

    normal_clusters = [l for l in unique_labels if l != -1] # noise 제외한 클러스터 목록
    n_normal = len(normal_clusters)

    cmap_base = plt.cm.get_cmap("tab20b", n_normal)
    normal_colors = [cmap_base(i) for i in range(n_normal)]

    noise_color = (0.7, 0.7, 0.7, 1.0)

    # cluster ID를 0 ~ N 범위에 매핑
    cluster_map = {cid: idx for idx, cid in enumerate(normal_clusters)}

    # 전체 cmap 구성
    final_colors = []
    for cid in unique_labels:
        if cid == -1:
            final_colors.append(noise_color)
        else:
            final_colors.append(normal_colors[cluster_map[cid]])

    cmap = mcolors.ListedColormap(final_colors)

    # 시각화
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        df["Y"], df["X"],
        c=labels,
        cmap=cmap,
        s=12,
        alpha=0.85
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(
        f"HDBSCAN Node Clustering\n"
        f"Clusters = {n_clusters}, Noise = {noise_ratio:.2f}%\n"
        f"min_cluster_size = {min_cluster_size}, min_samples = {min_samples}"
    )
    plt.ticklabel_format(style='plain', axis='both')
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)

    cbar = plt.colorbar(scatter)
    cbar.set_ticks(unique_labels)
    cbar.set_ticklabels([str(l) for l in unique_labels])
    cbar.set_label("Cluster ID")

    plt.savefig(out_image, dpi=300)
    plt.close()
    print(f"클러스터 시각화 이미지 저장 완료: {out_image}")

def run_hdbscan(df, min_cluster_size, min_samples, image_name=None):
    print(f"\n HDBSCAN 실행: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    # feature 준비 (X,Y,node_score)
    features = df[['Y', 'X', 'node_score']].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # HDBSCAN 실행
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X_scaled)

    df["cluster"] = labels

    # 클러스터 개수 계산
    unique_clusters = set(labels)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

    noise_ratio = (labels == -1).sum() / len(labels) * 100

    print(f"클러스터 개수: {n_clusters}")
    print(f"Noise 비율: {noise_ratio:.2f}%")
    print(f"Cluster labels: {df['cluster'].value_counts().to_dict()}")

    # 클러스터 산점도 저장
    if image_name:
        plot_clusters(df, image_name, n_clusters, noise_ratio, min_cluster_size, min_samples)

    return n_clusters, noise_ratio


def main():
    ap = argparse.ArgumentParser(description="HDBSCAN 파라미터 튜닝 및 시각화 도구 (Node-based)")
    ap.add_argument("--input", default="./result/intersection_score.csv", help="intersection score CSV 경로")
    ap.add_argument("--min_cluster_size", type=int, default=30, help="HDBSCAN 최소 클러스터 크기")
    ap.add_argument("--min_samples", type=int, default=5, help="노이즈 강도")
    ap.add_argument("--image", default="./result_pic/clusters.png", help="클러스터 시각화 이미지 파일명")
    ap.add_argument("--auto", action="store_true", help="파라미터 자동 탐색 실행")

    args = ap.parse_args()

    print("CSV 로드 중...")
    df = pd.read_csv(args.input)

    # Auto Mode
    if args.auto:
        print("\nAuto Parameter Sweep 시작")
        MIN_CLUSTER_LIST = [11, 12, 13, 14, 15]
        MIN_SAMPLES_LIST = [3, 4, 5, 6]

        results = []

        for mcs in MIN_CLUSTER_LIST:
            for ms in MIN_SAMPLES_LIST:
                img_name = f"./result_auto2/clusters_mcs{mcs}_ms{ms}.png"
                n_clusters, noise_ratio = run_hdbscan(df.copy(), mcs, ms, img_name)
                results.append([mcs, ms, n_clusters, noise_ratio])

        res_df = pd.DataFrame(results, columns=["min_cluster_size", "min_samples", "clusters", "noise_ratio"])
        res_df.to_csv("./result_auto2/hdbscan_param_results.csv", index=False)
        print("\n파라미터 탐색 결과 저장: hdbscan_param_results.csv")
        return

    # Single Run
    run_hdbscan(df, args.min_cluster_size, args.min_samples, args.image)



if __name__ == "__main__":
    main()

