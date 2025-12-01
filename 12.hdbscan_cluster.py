import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
import hdbscan
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def run_hdbscan_and_export(input_csv, output_csv, min_cluster_size, min_samples):
    print(f"입력 파일 로드: {input_csv}")
    df = pd.read_csv(input_csv)

    features = df[['Y', 'X', 'node_score']].values 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # HDBSCAN 실행
    print(f"HDBSCAN 실행: min_cluster_size={min_cluster_size}, min_samples={min_samples}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom'
    )

    labels = clusterer.fit_predict(X_scaled)
    df["cluster"] = labels


    unique_clusters = sorted(set(labels))
    n_clusters = len([c for c in unique_clusters if c != -1])
    noise_ratio = (labels == -1).sum() / len(labels) * 100

    print(f"클러스터 개수: {n_clusters}")
    print(f"Noise 비율: {noise_ratio:.2f}%")
    print(f"cluster 라벨 분포:")
    print(df["cluster"].value_counts())

    # cluster_id 기준 정렬 후 저장
    df_sorted = df.sort_values(by="cluster")
    df_sorted.to_csv(output_csv, index=False)

    print(f"결과 저장 완료 → {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="HDBSCAN 클러스터 CSV Export 도구")

    parser.add_argument("--input", default="./result/intersection_score.csv",
                        help="입력 node_scores CSV 경로")

    parser.add_argument("--output", default="./result/hdbscan_cluster_result.csv",
                        help="결과 클러스터 CSV 경로")

    parser.add_argument("--min_cluster_size", type=int, default=30,
                        help="HDBSCAN 최소 클러스터 크기")

    parser.add_argument("--min_samples", type=int, default=5,
                        help="HDBSCAN min_samples 파라미터")

    args = parser.parse_args()

    run_hdbscan_and_export(
        input_csv=args.input,
        output_csv=args.output,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples
    )


if __name__ == "__main__":
    main()
