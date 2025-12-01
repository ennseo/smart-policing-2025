import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6371000.0
EPS = 1e-12

# ---------- 좌표 변환 ----------
def to_rad(lat, lon):
    return np.radians(np.c_[lat, lon])

def radius_rad(m):
    return m / EARTH_RADIUS_M

# ---------- 반경 내 거리합 + 개수 ----------
def distance_sum_and_count(node_lat, node_lon, pt_lat, pt_lon, radius_m=255):
    n = len(node_lat)
    if len(pt_lat) == 0:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=int)

    tree = BallTree(to_rad(pt_lat, pt_lon), metric="haversine")
    nodes_rad = to_rad(node_lat, node_lon)

    ind_list, dist_list = tree.query_radius(
        nodes_rad, r=radius_rad(radius_m),
        return_distance=True, sort_results=True
    )

    dist_sum = np.zeros(n, dtype=float)
    count    = np.zeros(n, dtype=int)

    for i, (nds, d_rad) in enumerate(zip(ind_list, dist_list)):
        if len(nds) == 0:
            continue
        # d_m = d_rad * EARTH_RADIUS_M
        # dist_sum[i] = d_m.sum()
        # count[i]    = len(nds)

        d_m = d_rad * EARTH_RADIUS_M

        # === custom kernel 직접 삽입 ===
        # W(d) = 1 / (1 + d^p)
        p = 1.0   # 필요하면 함수 인자로
        k = 1.0 / (EPS + d_m ** p)
        w = k / (1.0 + k)

        dist_sum[i] = w.sum()       # 거리 대신 가중치 합
        count[i]    = len(nds)      # 개수 그대로

    return dist_sum, count

# ---------- 시간대 분류 ----------
def get_time_category(time_str):
    if pd.isna(time_str): return None
    s = ''.join(ch for ch in str(time_str) if ch.isdigit())
    if not s: return None
    try:
        hour = int(s.zfill(4)[:2])
    except:
        return None
    if   0 <= hour < 3:   return 'time_0_3'
    elif 3 <= hour < 6:   return 'time_3_6'
    elif 6 <= hour < 9:   return 'time_6_9'
    elif 9 <= hour < 12:  return 'time_9_12'
    elif 12 <= hour < 15: return 'time_12_15'
    elif 15 <= hour < 18: return 'time_15_18'
    elif 18 <= hour < 21: return 'time_18_21'
    elif 21 <= hour <= 23:return 'time_21_24'
    return None

# ---------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crime", default="./data/crime_count_grid.csv")
    ap.add_argument("--alcohol", default="./data/alcohol.csv")
    ap.add_argument("--bus", default="./data/bus.csv")
    ap.add_argument("--metro_portal", default="./data/metro_portal.csv")
    ap.add_argument("--metro", default="./data/metro.csv")
    ap.add_argument("--school", default="./data/school.csv")
    ap.add_argument("--traffic", default="./data/traffic_collision.csv")
    ap.add_argument("--arrest", default="./data/arrest_data.csv")
    ap.add_argument("--output", default="./data/crime_1.csv")
    ap.add_argument("--radius", type=int, default=127.5)
    args = ap.parse_args()

    def load_or_empty(path):
        try:
            return pd.read_csv(path)
        except:
            return pd.DataFrame(columns=["latitude","longitude"])

    print("데이터 로딩...")
    crime_df = pd.read_csv(args.crime)
    node_lat = crime_df["latitude"].to_numpy()
    node_lon = crime_df["longitude"].to_numpy()

    alcohol = load_or_empty(args.alcohol)
    bus     = load_or_empty(args.bus)
    mport   = load_or_empty(args.metro_portal)
    metro   = load_or_empty(args.metro)
    school  = load_or_empty(args.school)
    traffic = load_or_empty(args.traffic)
    arrest  = load_or_empty(args.arrest)

    print(f"nodes={len(crime_df)} | alcohol={len(alcohol)} bus={len(bus)} "
          f"metro_portal={len(mport)} metro={len(metro)} school={len(school)} "
          f"traffic={len(traffic)} arrest={len(arrest)}")

    # --- 시설: 거리합 + 개수 ---
    sources = {
        "alcohol": alcohol,
        "bus": bus,
        "metro_portal": mport,
        "metro": metro,
        "school": school,
    }
    for name, df_src in sources.items():
        dist_sum, cnt = distance_sum_and_count(
            node_lat, node_lon,
            df_src.get("latitude", pd.Series([])).to_numpy(),
            df_src.get("longitude", pd.Series([])).to_numpy(),
            radius_m=args.radius
        )
        crime_df[f"{name}_dist_sum"] = dist_sum
        crime_df[f"{name}_count"]    = cnt
        print(f"{name:12s}  mean dist_sum={dist_sum.mean():.2f}, mean count={cnt.mean():.2f}")

    # --- 교통사고: 시간대별 ---
    if not traffic.empty and {"latitude","longitude","Time_Occurred"}.issubset(traffic.columns):
        traffic["_time_cat"] = traffic["Time_Occurred"].map(get_time_category)
        times = ['time_0_3','time_3_6','time_6_9','time_9_12',
                 'time_12_15','time_15_18','time_18_21','time_21_24']
        for tcat in times:
            sub = traffic[traffic["_time_cat"] == tcat]
            dsum, ccnt = distance_sum_and_count(
                node_lat, node_lon,
                sub.get("latitude", pd.Series([])).to_numpy(),
                sub.get("longitude", pd.Series([])).to_numpy(),
                radius_m=args.radius
            )
            cname = f"traffic_{tcat}"
            crime_df[f"{cname}_dist_sum"] = dsum
            crime_df[f"{cname}_count"]    = ccnt
            print(f"{cname:18s} mean dist_sum={dsum.mean():.2f}, mean count={ccnt.mean():.2f}")

    # --- 체포: 시간대별 ---
    if not arrest.empty and {"latitude","longitude","Time"}.issubset(arrest.columns):
        arrest["_time_cat"] = arrest["Time"].map(get_time_category)
        times = ['time_0_3','time_3_6','time_6_9','time_9_12',
                 'time_12_15','time_15_18','time_18_21','time_21_24']
        for tcat in times:
            sub = arrest[arrest["_time_cat"] == tcat]
            dsum, ccnt = distance_sum_and_count(
                node_lat, node_lon,
                sub.get("latitude", pd.Series([])).to_numpy(),
                sub.get("longitude", pd.Series([])).to_numpy(),
                radius_m=args.radius
            )
            cname = f"arrest_{tcat}"
            crime_df[f"{cname}_dist_sum"] = dsum
            crime_df[f"{cname}_count"]    = ccnt
            print(f"{cname:18s} mean dist_sum={dsum.mean():.2f}, mean count={ccnt.mean():.2f}")

    # --- 저장 ---
    crime_df.to_csv(args.output, index=False)
    print(f"\n저장 완료: {args.output}")
    print(f"최종 데이터 shape: {crime_df.shape}")

if __name__ == "__main__":
    main()
    