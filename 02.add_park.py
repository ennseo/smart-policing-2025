import pandas as pd
import numpy as np
from itertools import product

def invdist(dr, dc, cell_size_m=85.0, center_eps_frac=0.25):
    """
    3x3 오프셋(dr,dc)에서 그리드 중심 간 거리(미터)를 반환.
    중심셀(dr=dc=0)은 d=cell_size_m*center_eps_frac 로 처리.
    """
    d = np.hypot(dr * cell_size_m, dc * cell_size_m)
    if dr == 0 and dc == 0:
        d = cell_size_m * center_eps_frac
    return float(d)

def build_park_3x3_features_presence(crime_df, park_grid_df, cell_size_m=85.0):
    """
    공원 존재 여부(1/0) 기반 3x3 inverse-distance 평균 피처 생성.
    feature = (1/9) * Σ_{dr,dc∈{-1,0,1}} [ presence(rr,cc) / dist((r0,c0),(rr,cc)) ]
    presence(rr,cc) = (park_grid_df에 (rr,cc) 있으면 1, 없으면 0)
    """
    # 공원 격자 좌표를 set으로 (존재=1, 미존재=0)
    park_set = set(zip(park_grid_df['grid_row'].astype(int),
                       park_grid_df['grid_col'].astype(int)))

    offsets = list(product([-1,0,1], [-1,0,1]))
    invdv_mean = []
    park_center = []

    for _, row in crime_df.iterrows():
        r0, c0 = int(row['grid_row']), int(row['grid_col'])
        s = 0.0
        center_presence = 1.0 if (r0, c0) in park_set else 0.0

        for dr, dc in offsets:
            rr, cc = r0 + dr, c0 + dc
            presence = 1.0 if (rr, cc) in park_set else 0.0
            d = invdist(dr, dc, cell_size_m=cell_size_m)
            s += presence / d  # d>0 (중심은 가상거리)라 괜찮

        invdv_mean.append(s / 9.0)
        # park_center.append(center_presence)

    return pd.DataFrame({
        # "park_center": park_center,          # 중심 셀 공원 존재(1/0)
        "park_invdv_mean_3x3": invdv_mean    # (presence / distance)의 3x3 평균
    })

def main():
    crime_path = "./data/crime_1.csv"   # grid_row, grid_col 포함
    park_path  = "./data/park.csv"      # 공원 있는 셀만 (grid_row, grid_col)
    output_path = "./data/crime_2.csv"
    cell_size_m = 85.0

    print("데이터 로딩")
    crime_df = pd.read_csv(crime_path)
    park_df  = pd.read_csv(park_path)

    need = {"grid_row","grid_col"}
    if not need.issubset(crime_df.columns):
        raise ValueError("crime_df에 grid_row, grid_col 필요")
    if not need.issubset(park_df.columns):
        raise ValueError("park_df에 grid_row, grid_col 필요")

    print("공원 피처 생성 (presence 기반 1/d 평균)")
    park_feat = build_park_3x3_features_presence(crime_df, park_df, cell_size_m=cell_size_m)

    print("병합 & 저장")
    out = pd.concat([crime_df, park_feat], axis=1)
    out.to_csv(output_path, index=False)
    print(f"완료: {output_path}  shape={out.shape}")

if __name__ == "__main__":
    main()
    