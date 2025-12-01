# build_building_ratio_features.py
import argparse
import pandas as pd
import numpy as np
from itertools import product

def invdist(dr: int, dc: int, cell_size_m: float = 85.0, center_eps_frac: float = 0.25) -> float:
    """
    그리드 중심 간 거리(미터). 중심셀(dr=dc=0)은 가상거리(cell_size*center_eps_frac) 사용.
    """
    d = np.hypot(dr * cell_size_m, dc * cell_size_m)
    if dr == 0 and dc == 0:
        d = cell_size_m * center_eps_frac
    return float(d)

def build_building_ratio_features(crime_df: pd.DataFrame,
                                  ratio_df: pd.DataFrame,
                                  cell_size_m: float = 85.0,
                                  center_eps_frac: float = 0.25) -> pd.DataFrame:
    """
    building_area_ratio (0~1)을 3x3 inverse-distance 평균으로 피처화:
      - building_area_ratio_center
      - building_area_ratio_invdv_mean_3x3 = (1/9) * Σ( area_ratio / dist )
    입력:
      crime_df: grid_row, grid_col 포함
      ratio_df: (row_index,col_index) 또는 (grid_row,grid_col) + area_ratio 포함
    """
    # 컬럼명 정규화
    ratio_df = ratio_df.copy()
    if {"row_index","col_index"}.issubset(ratio_df.columns):
        ratio_df.rename(columns={"row_index":"grid_row", "col_index":"grid_col"}, inplace=True)

    need_crime = {"grid_row","grid_col"}
    need_ratio = {"grid_row","grid_col","area_ratio"}
    if not need_crime.issubset(crime_df.columns):
        raise ValueError("crime_df에 grid_row, grid_col 컬럼이 필요합니다.")
    if not need_ratio.issubset(ratio_df.columns):
        raise ValueError("ratio_df에 grid_row, grid_col, area_ratio 컬럼이 필요합니다.")

    # (row,col) -> area_ratio 맵
    ratio_map = {(int(r), int(c)): float(v)
                 for r, c, v in zip(ratio_df["grid_row"], ratio_df["grid_col"], ratio_df["area_ratio"])}

    offsets = list(product([-1,0,1], [-1,0,1]))
    invdv_mean_list, center_vals = [], []

    for _, row in crime_df.iterrows():
        r0, c0 = int(row["grid_row"]), int(row["grid_col"])
        s = 0.0
        center_val = ratio_map.get((r0, c0), 0.0)

        for dr, dc in offsets:
            rr, cc = r0 + dr, c0 + dc
            val = ratio_map.get((rr, cc), 0.0)  # 0~1
            d   = invdist(dr, dc, cell_size_m=cell_size_m, center_eps_frac=center_eps_frac)
            s  += (val / d) if d > 0 else 0.0

        invdv_mean_list.append(s / 9.0)  # 3x3 평균
        # center_vals.append(center_val)

    return pd.DataFrame({
        # "building_area_ratio_center": center_vals,
        "building_area_ratio_invdv_mean_3x3": invdv_mean_list
    })

def main():
    ap = argparse.ArgumentParser(description="Building area ratio 3x3 inverse-distance feature generator")
    ap.add_argument("--crime", default="./data/crime_3.csv")
    ap.add_argument("--ratio", default="./data/building_area_ratio.csv")
    ap.add_argument("--out",   default="./data/soft_counting_final.csv")
    ap.add_argument("--cell_size", type=float, default=85.0, help="그리드 셀 한 변 길이(m)")
    ap.add_argument("--center_eps_frac", type=float, default=0.25, help="중심 가상거리 비율 (cell_size * frac)")
    args = ap.parse_args()

    print("Loading...")
    crime = pd.read_csv(args.crime)
    ratio = pd.read_csv(args.ratio)

    print(f"crime: {crime.shape}, ratio: {ratio.shape}")
    feats = build_building_ratio_features(
        crime_df=crime,
        ratio_df=ratio,
        cell_size_m=args.cell_size,
        center_eps_frac=args.center_eps_frac
    )

    out = pd.concat([crime, feats], axis=1)
    out.to_csv(args.out, index=False)
    print(f"Saved -> {args.out}  shape={out.shape}")

    # 간단 요약
    # cols = ["building_area_ratio_center", "building_area_ratio_invdv_mean_3x3"]
    cols = ["building_area_ratio_invdv_mean_3x3"]
    print(out[cols].describe())

if __name__ == "__main__":
    main()
    