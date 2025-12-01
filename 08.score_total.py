import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6_371_000.0
EPS = 1e-12


# ---------- 좌표 유틸 ----------
def to_rad(lat, lon):
    return np.radians(np.c_[lat, lon])


def radius_rad(m):
    return m / EARTH_RADIUS_M


def softmax_stable(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / max(ex.sum(), EPS)


# ---------- 좌표 로딩 ----------
def load_coords(final_path):
    df = pd.read_csv(final_path)
    if "node_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "node_id"})
    keep = ["node_id", "latitude", "longitude"]
    for extra in ["grid_row", "grid_col"]:
        if extra in df.columns:
            keep.append(extra)
    return df[keep].drop_duplicates("node_id")


# ---------- 역거리 softmax 전파 ----------
def propagate_invdist_softmax(nodes_df, scores_df, radius_m=255.0, add_self=False):
    need_cols = {"node_id", "latitude", "longitude"}
    assert need_cols.issubset(nodes_df.columns), f"nodes_df에 {need_cols} 필요"
    assert {"node_id", "local_score"}.issubset(
        scores_df.columns
    ), "scores_df에 node_id, local_score 필요"

    nodes = nodes_df.sort_values("node_id").reset_index(drop=True)
    scores = scores_df.sort_values("node_id").reset_index(drop=True)

    if not np.array_equal(nodes["node_id"].to_numpy(), scores["node_id"].to_numpy()):
        merged = nodes.merge(scores, on="node_id", how="left")
        if merged["local_score"].isna().any():
            raise ValueError("scores에 없는 node_id 존재")
        nodes = merged.drop(columns=["local_score"])
        scores = merged[["node_id", "local_score"]]

    local = scores["local_score"].to_numpy(float)
    N = len(nodes)

    coords_rad = to_rad(nodes["latitude"], nodes["longitude"])
    tree = BallTree(coords_rad, metric="haversine")
    ind_list, dist_list = tree.query_radius(
        coords_rad, r=radius_rad(radius_m), return_distance=True, sort_results=True
    )

    updated = local.copy()
    nn_counts = np.zeros(N, dtype=int)

    for i, (nbr_idx, d_rad) in enumerate(zip(ind_list, dist_list)):
        if len(nbr_idx) == 0:
            continue

        d_m = np.maximum(d_rad * EARTH_RADIUS_M, EPS)
        idx = nbr_idx

        if not add_self:
            mask = idx != i
            idx = idx[mask]
            d_m = d_m[mask]

        if len(idx) == 0:
            continue

        # 역거리
        g = 1.0 / (d_m + EPS)
        # softmax 정규화
        k = g / (1.0 + g)
        w = softmax_stable(k)

        # 이웃 가중합
        nn_sum = float((w * local[idx]).sum())
        updated[i] = local[i] + nn_sum
        nn_counts[i] = len(idx)

    out = nodes.copy()
    out["local_score"] = local
    out["updated_score"] = updated
    out["neighbor_count"] = nn_counts
    return out


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="역거리 softmax 이웃 전파")
    ap.add_argument("--final", default="./data/soft_counting_final.csv")
    ap.add_argument("--scores", default="./result/score_node_shrinkage.csv")
    ap.add_argument("--out", default="./result/score_total.csv")
    ap.add_argument("--radius", type=float, default=510.0, help="반경(m)")
    ap.add_argument(
        "--add_self", action="store_true", help="자기 자신도 softmax에 포함"
    )
    args = ap.parse_args()

    nodes_df = load_coords(args.final)
    scores_df = pd.read_csv(args.scores)

    out_df = propagate_invdist_softmax(
        nodes_df, scores_df, radius_m=args.radius, add_self=args.add_self
    )
    out_df.to_csv(args.out, index=False)
    print(f"[saved] {args.out}  shape={out_df.shape}")
    print(out_df[["node_id", "local_score", "updated_score", "neighbor_count"]].head())


if __name__ == "__main__":
    main()
 