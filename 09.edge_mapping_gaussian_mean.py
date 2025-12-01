import argparse
import pandas as pd
import numpy as np
from math import exp, sqrt
import matplotlib.pyplot as plt
import seaborn as sns

EARTH_RADIUS_M = 6_371_000.0  # 지구 반경 (미터 단위)


# ---------- 좌표 변환 ----------
def to_rad(lat, lon):
    return np.radians(np.c_[lat, lon])


def haversine_distance(lat1, lon1, lat2, lon2):
    """두 점 사이 거리 (미터)"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# ---------- 가우시안 가중 함수 ----------
def gaussian_weight(distance_m, sigma):
    """가우시안 커널 가중치 계산"""
    return np.exp(- (distance_m ** 2) / (2 * sigma ** 2))


# ---------- 로그 + 시그모이드 스케일링 ----------
def log_sigmoid_scale(series: pd.Series) -> pd.Series:
    """
    로그 압축 + 시그모이드 정규화
    음수/양수 모두 (0,1) 범위로 매핑
    """
    arr = np.sign(series) * np.log1p(np.abs(series))  # 로그 압축 (음수 포함)
    scaled = 1 / (1 + np.exp(-arr))                   # 시그모이드 변환
    return pd.Series(scaled, index=series.index)


def main():
    # ---------- argparse ----------
    ap = argparse.ArgumentParser(description="거리 기반 가우시안 가중치로 엣지 점수 계산 (mean 모드, log-sigmoid 스케일 적용)")
    ap.add_argument("--edges", default="./data/road_network_final.csv", help="도로망 엣지 CSV 경로")
    ap.add_argument("--scores", default="./result/score_total.csv", help="노드 점수 파일 경로")
    ap.add_argument("--out", default="./result/edge_score_gaussian_mean.csv", help="출력 파일 경로")
    ap.add_argument("--sigma", type=float, default=255.0, help="가우시안 거리 스케일 sigma (미터 단위)")
    args = ap.parse_args()

    mode = "mean"

    print(f"=== 설정 ===")
    print(f"mode: {mode}")
    print(f"sigma: {args.sigma}m")

    # ---------- 데이터 로딩 ----------
    edges = pd.read_csv(args.edges)
    nodes = pd.read_csv(args.scores)[["latitude", "longitude", "updated_score"]]
    print(f"\n엣지 개수: {len(edges):,}, 노드 개수: {len(nodes):,}")

    # 엣지 중심점 계산
    edges["mid_lat"] = (edges["LAT_FROM"] + edges["LAT_TO"]) / 2
    edges["mid_lon"] = (edges["LON_FROM"] + edges["LON_TO"]) / 2

    edge_coords = edges[["mid_lat", "mid_lon"]].to_numpy()
    node_coords = nodes[["latitude", "longitude"]].to_numpy()
    node_scores = nodes["updated_score"].to_numpy()

    edge_scores = np.zeros(len(edges), dtype=float)

    # ---------- 거리 기반 가우시안 가중치 계산 ----------
    print("거리 기반 가우시안 가중치 계산 중..")
    for i, (elat, elon) in enumerate(edge_coords):
        dists = haversine_distance(elat, elon, node_coords[:, 0], node_coords[:, 1])
        weights = gaussian_weight(dists, args.sigma)

        weighted_sum = np.sum(weights * node_scores)
        weight_total = np.sum(weights)
        edge_scores[i] = weighted_sum / weight_total if weight_total > 0 else 0

    # ---------- 결과 저장 ----------
    edges["edge_score_gaussian"] = edge_scores
    edges["edge_score_scaled"] = log_sigmoid_scale(edges["edge_score_gaussian"])

    out_path = args.out
    edges.to_csv(out_path, index=False)
    print(f"saved edge score_gaussian_mean ->{out_path}")

    # ---------- 상위 20% 엣지 추출 ----------
    threshold = edges["edge_score_scaled"].quantile(0.8)
    top_20pct = edges[edges["edge_score_scaled"] >= threshold]
    top20pct_path = out_path.replace(".csv", "_top20pct.csv")
    top_20pct.to_csv(top20pct_path, index=False)
    print(f"saved top20pct ({len(top_20pct):,}개) ->{top20pct_path}")

    # ---------- 통계 요약 ----------
    mean_raw = edges["edge_score_gaussian"].mean()
    median_raw = edges["edge_score_gaussian"].median()
    min_raw = edges["edge_score_gaussian"].min()
    max_raw = edges["edge_score_gaussian"].max()

    mean_scaled = edges["edge_score_scaled"].mean()
    median_scaled = edges["edge_score_scaled"].median()
    min_scaled = edges["edge_score_scaled"].min()
    max_scaled = edges["edge_score_scaled"].max()

    print("\n엣지 가중치 통계 — no scaling")
    print(f"평균: {mean_raw:.4f} | 중앙값: {median_raw:.4f} | 최소: {min_raw:.4f} | 최대: {max_raw:.4f}")
    print("\n엣지 가중치 통계 — scaling (0~1)")
    print(f"평균: {mean_scaled:.4f} | 중앙값: {median_scaled:.4f} | 최소: {min_scaled:.4f} | 최대: {max_scaled:.4f}")

    # ---------- 시각화 (1) 원본 ----------
    plt.figure(figsize=(8, 5))
    sns.histplot(edges["edge_score_gaussian"], bins=50, kde=True, color="gray")
    plt.title("Distribution of Raw Edge Risk Scores", fontsize=13)
    plt.xlabel("Raw Edge Risk Score (unscaled)", fontsize=11)
    plt.ylabel("Edge Count", fontsize=11)
    plt.tight_layout()

    # hist_raw_path = out_path.replace(".csv", "_hist_raw.png")
    hist_raw_path = "./result_pic/" + out_path.split("/")[-1].replace(".csv", "_hist_raw.png")
    plt.savefig(hist_raw_path, dpi=300)
    plt.close()
    print(f"saved .png -> {hist_raw_path}")

    # ---------- 시각화 (2) 스케일된 ----------
    plt.figure(figsize=(8, 5))
    sns.histplot(edges["edge_score_scaled"], bins=50, kde=True, color="steelblue")
    plt.title("Distribution of Scaled Edge Risk Scores", fontsize=13)
    plt.xlabel("Scaled Edge Risk Score (0–1)", fontsize=11)
    plt.ylabel("Edge Count", fontsize=11)
    plt.tight_layout()

    # hist_scaled_path = out_path.replace(".csv", "_hist_scaled.png")
    hist_scaled_path = "./result_pic/" + out_path.split("/")[-1].replace(".csv", "_hist_scaled.png")
    plt.savefig(hist_scaled_path, dpi=300)
    plt.close()
    print(f"saved .png -> {hist_scaled_path}")


if __name__ == "__main__":
    main()
