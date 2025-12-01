import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

EXCLUDE_SUFFIXES = ("_count",)

# ---------- 지리 좌표 변환 ----------
EARTH_RADIUS_M = 6371000.0
def to_rad(lat, lon): 
    return np.radians(np.c_[lat, lon])

def radius_rad(m):    
    return m / EARTH_RADIUS_M


def main():
    data_path = "./data/soft_counting_final.csv"
    output_path = "./result/spearman_local.csv"
    summary_path = "./result/spearman_local_summary.csv"
    radius_m = 510   # 반경 (미터 단위)

    print("데이터 로딩 중..")
    df = pd.read_csv(data_path)
    print(f"데이터 크기: {df.shape}")

    # 좌표와 타겟 변수
    node_lat = df["latitude"].to_numpy()
    node_lon = df["longitude"].to_numpy()
    nodes_rad = to_rad(node_lat, node_lon)
    target_col = "crime_count"

    # 분석 제외 컬럼
    exclude_cols = ['grid_row', 'grid_col', 'latitude', 'longitude', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.endswith(EXCLUDE_SUFFIXES)]

    print(f"분석 대상 피처 수: {len(feature_cols)}")
    print(f"타겟 변수: {target_col}")

    # BallTree 생성
    tree = BallTree(nodes_rad, metric='haversine')
    ind_list = tree.query_radius(nodes_rad, r=radius_rad(radius_m))

    neighbor_counts = np.array([len(ind) for ind in ind_list])
    few_neighbors_10 = np.sum(neighbor_counts < 10)
    few_neighbors_50 = np.sum(neighbor_counts < 50)
    few_neighbors_100 = np.sum(neighbor_counts < 100)

    print("\n===== 반경 510m 내 이웃 노드 개수 요약 =====")
    print(f"총 노드 수: {len(df)}")
    print(f"이웃노드가 10개 미만인 노드 수: {few_neighbors_10}")
    print(f"이웃노드가 50개 미만인 노드 수: {few_neighbors_50}")
    print(f"이웃노드가 100개 미만인 노드 수: {few_neighbors_100}")
    print(f"평균 이웃 노드 수: {np.mean(neighbor_counts):.1f}")
    print(f"최소: {np.min(neighbor_counts)}, 최대: {np.max(neighbor_counts)}\n")

    # 각 노드별 상관계수 계산
    results = []
    for node_id, neighbors in enumerate(ind_list):
        neighbor_df = df.iloc[neighbors]
        row_result = {"node_id": node_id}

        for col in feature_cols:
            try:
                corr, _ = spearmanr(neighbor_df[col], neighbor_df[target_col])
                row_result[col] = corr
            except Exception:
                row_result[col] = np.nan

        row_result["neighbor_count"] = len(neighbor_df)
        results.append(row_result)

        if node_id < 2:  # 앞 2개 노드만 샘플 출력
            print(f"node_id={node_id}, neighbors={len(neighbor_df)}")
            sample_corrs = {f: row_result[f] for f in feature_cols[:3]}
            print("  샘플 상관계수:", sample_corrs)

    # 결과 DataFrame
    result_df = pd.DataFrame(results)

    # CSV 저장 (updated: neighbor_count col 추가)
    result_df.to_csv(output_path, index=False)
    print(f"\n로컬 Spearman 상관계수 + neighbor_count 저장 완료: {output_path}")
    print(f"결과 크기: {result_df.shape} (노드 × 피처 매트릭스 + neighbor_count)")

    print("\n===== 노드별 요약 통계 생성 중 =====")
    result_df_indexed = result_df.set_index("node_id")

    summary = []
    for node_id, row in result_df_indexed.iterrows():
        abs_vals = row[feature_cols].abs()
        mean_abs_corr = abs_vals.mean()
        top_feature = abs_vals.idxmax()
        top_corr = row[top_feature]
        summary.append([node_id, mean_abs_corr, top_feature, top_corr, row["neighbor_count"]])

    summary_df = pd.DataFrame(summary, 
                              columns=["node_id", "mean_abs_corr", "top_feature", "top_corr", "neighbor_count"])

    summary_df.to_csv(summary_path, index=False)
    print(f"노드별 요약 통계 저장: {summary_path}")
    print(f"요약 데이터 크기: {summary_df.shape} (노드 × 5)")
    print("\n샘플 5개:")
    print(summary_df.head())


if __name__ == "__main__":
    main()
