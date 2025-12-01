import pandas as pd
import numpy as np
from itertools import product

def invdist(dr, dc, cell_size_m=85.0, center_eps_frac=0.25):
    d = np.hypot(dr * cell_size_m, dc * cell_size_m)
    if dr == 0 and dc == 0:
        d = cell_size_m * center_eps_frac
    return float(d)

def build_population_grid_features(crime_df, pop_df, cell_size_m=85.0):
    """
    각 col에 대해:
      invdv_mean_3x3 = (1/9) * Σ_{3x3} [ col(r,c) / dist((r0,c0),(r,c)) ]
      col_center     = 중심 셀의 값 (참고용)
    """
    pop_idx = pop_df.set_index(["grid_row","grid_col"])
    pop_cols = [c for c in pop_df.columns if c not in ["idx","grid_row","grid_col"]]
    offsets = list(product([-1,0,1], [-1,0,1]))

    out = {}
    # 미리 dict 캐시: (row,col)->값
    maps = {
        col: {(int(r), int(c)): float(v) for (r,c), v in pop_idx[col].dropna().items()}
        for col in pop_cols
    }

    for col in pop_cols:
        invdv_mean, centers = [], []
        cmap = maps[col]
        for _, row in crime_df.iterrows():
            r0, c0 = int(row["grid_row"]), int(row["grid_col"])
            s = 0.0
            center_v = cmap.get((r0, c0), 0.0)
            for dr, dc in offsets:
                rr, cc = r0+dr, c0+dc
                val = cmap.get((rr, cc), 0.0)
                d = invdist(dr, dc, cell_size_m=cell_size_m)
                s += (val / d) if d > 0 else 0.0
            invdv_mean.append(s / 9.0)
            # centers.append(center_v)

        # out[f"{col}_center"]        = centers
        out[f"{col}_invdv_mean_3x3"] = invdv_mean

    return pd.DataFrame(out)

def main():
    crime_path = "./data/crime_2.csv"
    pop_path   = "./data/pop.csv"
    output_path = "./data/crime_3.csv"
    cell_size_m = 85.0

    crime_df = pd.read_csv(crime_path)
    pop_df   = pd.read_csv(pop_path)

    req = {"grid_row","grid_col"}
    if not req.issubset(crime_df.columns): raise ValueError("crime: grid_row, grid_col 필요")
    if not req.issubset(pop_df.columns):   raise ValueError("pop: grid_row, grid_col 필요")

    pop_feat = build_population_grid_features(crime_df, pop_df, cell_size_m=cell_size_m)
    out = pd.concat([crime_df, pop_feat], axis=1)
    out.to_csv(output_path, index=False)
    print(f"saved -> {output_path}  shape={out.shape}")

if __name__ == "__main__":
    main()
    