import argparse
import numpy as np
import pandas as pd

EXCLUDE_EXACT = {"grid_row", "grid_col", "latitude", "longitude", "crime_count", "node_id"}
EXCLUDE_SUFFIXES = ("_count",)
TIME_BINS = ["0_3","3_6","6_9","9_12","12_15","15_18","18_21","21_24"]

def rank_percentile_to_pm1(x: pd.Series) -> pd.Series:
    """랭크-퍼센타일을 [-1,1]로 변환"""
    mask = ~x.isna()
    out = pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    if mask.sum() > 0:
        vals = x[mask].to_numpy()
        order = np.argsort(vals, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = (np.arange(len(order)) + 0.5) / len(order)
        out.loc[mask] = 2.0 * ranks - 1.0
    out.loc[~mask] = 0.0
    return out


def compute_wi(neighbor_counts: pd.Series) -> pd.Series:
    """w_i = (n_i - n_min) / (n_max - n_min)"""
    n_min, n_max = neighbor_counts.min(), neighbor_counts.max()
    if n_max == n_min:
        return pd.Series(np.ones_like(neighbor_counts), index=neighbor_counts.index)
    return (neighbor_counts - n_min) / (n_max - n_min)


def main():
    ap = argparse.ArgumentParser(description="Bayesian Shrinkage 기반 점수 계산")
    ap.add_argument("--features", default="./data/soft_counting_final.csv", help="피처 데이터")
    ap.add_argument("--local_r", default="./result/spearman_local.csv", help="로컬 Spearman 상관계수 CSV (neighbor_count col 포함)")
    ap.add_argument("--global_r", default="./result/spearman_global.csv", help="글로벌 Spearman 상관계수 CSV")
    ap.add_argument("--out", default="./result/score_node_shrinkage.csv", help="출력 파일 경로")
    args = ap.parse_args()

    # 1. 데이터 로드
    df = pd.read_csv(args.features)
    R_local = pd.read_csv(args.local_r)
    R_global = pd.read_csv(args.global_r)

    if "node_id" not in R_local.columns:
        raise ValueError("spearman_local.csv에 node_id 컬럼이 필요합니다.")
    if "neighbor_count" not in R_local.columns:
        raise ValueError("spearman_local.csv에 neighbor_count 컬럼이 필요합니다.")
    
    R_local = R_local.set_index("node_id")

    if "feature" not in R_global.columns or "spearman_corr" not in R_global.columns:
        raise ValueError("spearman_global.csv에는 feature, spearman_corr 컬럼이 필요합니다.")
    global_dict = dict(zip(R_global["feature"], R_global["spearman_corr"])) # 글로벌 상관계수 딕셔너리화

    print(f"로컬 노드 수: {len(R_local)}, 글로벌 피처 수: {len(global_dict)}")

    # 2. 사용할 피처 선택 (local, global 모두 존재하는 피처만 사용)
    feat_cols_raw = [c for c in df.columns if c not in EXCLUDE_EXACT and not c.endswith(EXCLUDE_SUFFIXES)]
    feat_cols = [c for c in feat_cols_raw if c in R_local.columns and c in global_dict]
    print(f"사용 피처 수: {len(feat_cols)}")

    # 3. 가중치 w_i 계산
    w = compute_wi(R_local["neighbor_count"])
    print(f"가중치 w_i 범위: {w.min():.3f} ~ {w.max():.3f}")

    # 4. Bayesian Shrinkage 상관계수 결합
    R_shrink = pd.DataFrame(index=R_local.index)
    for c in feat_cols:
        r_local = R_local[c].astype(float)
        r_global = global_dict[c]
        R_shrink[c] = w * r_local + (1 - w) * r_global

    # 5.  피처 정규화
    Xnorm = pd.DataFrame(index=df.index)
    for c in feat_cols:
        Xnorm[c] = rank_percentile_to_pm1(df[c].astype(float))
    Xnorm.index.name = "node_id"
    Xnorm_reset = Xnorm.reset_index()
    Xnorm_reset["node_id"] = Xnorm_reset["node_id"].astype(int)

    # 6. Shrinkage r 기반 노드 점수(local_score) 계산
    merged = Xnorm_reset.merge(R_shrink.reset_index(), on="node_id", how="inner", suffixes=("_s", "_r"))
    contribs = []
    for c in feat_cols:
        s_col = f"{c}_s"
        r_col = f"{c}_r"
        merged[f"contrib__{c}"] = merged[s_col] * merged[r_col]
        contribs.append(f"contrib__{c}")

    merged["local_score"] = merged[contribs].sum(axis=1)
    print(f"노드별 점수 계산 완료 (shape={merged.shape})")

    # 7. score only 저장
    score_df = merged[["node_id", "local_score"]]
    score_df.to_csv(args.out, index=False)
    print(f"saved score -> {args.out} (노드별 점수)")

    # 8. breakdown 저장
    breakdown_path = args.out.replace(".csv", "_breakdown.csv")
    merged[["node_id"] + contribs + ["local_score"]].to_csv(breakdown_path, index=False)
    print(f"saved breakdown -> {breakdown_path} (피처별 기여도)")

    # 9. 그룹별 합산 (교통사고/체포는 시간대별로 각각 합계)
    groups = {
        "alcohol":   [c for c in contribs if c.startswith("contrib__alcohol")],
        "bus":       [c for c in contribs if c.startswith("contrib__bus")],
        "metro":     [c for c in contribs if "contrib__metro_dist_sum" in c or "contrib__metro_invdv" in c or "contrib__metro_" in c],
        "metro_portal": [c for c in contribs if c.startswith("contrib__metro_portal")],
        "school":    [c for c in contribs if c.startswith("contrib__school")],
        "population":[c for c in contribs if c.startswith("contrib__POP23")],
        "poverty":   [c for c in contribs if c.startswith("contrib__POV23")],
        "park":      [c for c in contribs if c.startswith("contrib__park")],
        "building":  [c for c in contribs if c.startswith("contrib__building_area_ratio")],
    }
    # 교통사고/체포: 시간대별로 별도 컬럼 생성
    for tb in TIME_BINS:
        groups[f"traffic_{tb}"] = [c for c in contribs if c.startswith(f"contrib__traffic_time_{tb}")]
        groups[f"arrest_{tb}"]  = [c for c in contribs if c.startswith(f"contrib__arrest_time_{tb}")]

    grp_df = pd.DataFrame({"node_id": merged["node_id"]})
    for gname, cols in groups.items():
        if cols:
            grp_df[f"{gname}_score"] = merged[cols].sum(axis=1)
    grp_df["local_score"] = merged["local_score"]

    grouped_path = args.out.replace(".csv", "_grouped.csv")
    grp_df.to_csv(grouped_path, index=False)
    print(f"saved grouped -> {grouped_path} (그룹별 합산 완료)")


if __name__ == "__main__":
    main()
