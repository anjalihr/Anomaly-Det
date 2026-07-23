
"""
features.py
============
Converts raw synthetic_logs.csv into a feature table where every event has
~15 numeric behavioral signals, each answering: "how unusual is this event
relative to THIS entity's own history?" This is the layer that determines
the system's false-positive rate, so the golden rule enforced throughout is:
 
    STRICT CAUSALITY: every feature for event[t] is computed using only
    events with timestamp < t for that entity. Nothing from the future is
    ever used. This mirrors how a real-time detector actually operates -
    it never gets to see what happens after the event it's scoring.
 
Feature groups -> attack types they target
-------------------------------------------
  hour_zscore                  -> credential_misuse
  geo_velocity_kmh              -> impossible_travel
  is_new_device                 -> credential_misuse, brute_force
  fingerprint_mismatch          -> device_spoofing
  is_new_host, foreign_dept_*   -> lateral_movement (graph-based, not a
                                    naive host-count rule - see notes below)
  failure_burst_count           -> brute_force
  entity_history_* / cold_start -> drives the cold-start blending logic
 
Cold-start handling
--------------------
For entities with thin history (< MIN_HISTORY_EVENTS), personal baselines
are unreliable (e.g. a mean/std from 2 data points is meaningless). We
blend each entity's own baseline with a POPULATION-LEVEL baseline (typical
behavior for their department), weighted by how much personal history they
have. The population baseline itself is computed only from established
(non cold-start) users' full history - this represents pre-existing
organizational knowledge (e.g. "engineering typically logs in 9-6"), not
information leaked from any individual entity's own future.
 
Lateral movement (graph-based)
-------------------------------
Instead of "flag if user touches > N hosts," we use the access-topology
graph from Day 1 to track, in a rolling window, how many DISTINCT FOREIGN
DEPARTMENT CLUSTERS an entity has touched that it doesn't normally touch.
Crossing cluster boundaries the entity has never crossed is a structurally
stronger signal than raw host count, and it's what keeps false positives
down for legitimately broad-access roles (e.g. IT admins).
"""
 
import json
from collections import Counter, defaultdict, deque
 
import numpy as np
import pandas as pd
 
DATA_DIR = "/Users/anj/Projects/anomaly_det/data"
MIN_HISTORY_EVENTS = 15       # below this, an entity is treated as "cold start"
LATERAL_WINDOW_MIN = 15       # rolling window for lateral movement burst detection
BRUTE_FORCE_WINDOW_MIN = 5    # rolling window for failed-login burst detection
 
 
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
 
 
def circular_hour_stats(hours):
    """Mean/std of a list of hour-of-day values (0-23), handling the wraparound
    at midnight (23:00 and 00:00 should be treated as close, not far apart)."""
    if len(hours) == 0:
        return None, None
    angles = np.array(hours) / 24.0 * 2 * np.pi
    sin_mean = np.mean(np.sin(angles))
    cos_mean = np.mean(np.cos(angles))
    mean_angle = np.arctan2(sin_mean, cos_mean)
    mean_hour = (mean_angle / (2 * np.pi) * 24) % 24
    r = np.sqrt(sin_mean ** 2 + cos_mean ** 2)  # mean resultant length (1=no spread, 0=max spread)
    circular_std_hour = np.sqrt(-2 * np.log(max(r, 1e-6))) * (24 / (2 * np.pi))
    return mean_hour, max(circular_std_hour, 0.5)  # floor std to avoid div-by-~0
 
 
def circular_hour_distance(hour, mean_hour):
    """Shortest distance (in hours) between `hour` and `mean_hour` on a 24h clock."""
    diff = abs(hour - mean_hour) % 24
    return min(diff, 24 - diff)
 
 
def load_data():
    df = pd.read_csv(f"{DATA_DIR}/synthetic_logs.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    with open(f"{DATA_DIR}/entities.json") as f:
        entities = json.load(f)
    return df, entities
 
 
def build_host_dept_map(entities):
    host_dept = {}
    for dept, hosts in entities["dept_hosts"].items():
        for h in hosts:
            host_dept[h] = dept
    for h in entities["shared_hosts"]:
        host_dept[h] = "shared"
    return host_dept
 
 
def get_device_true_fingerprint(entities, device_id):
    dev = entities["devices"].get(device_id)
    if dev is None:
        return None  # unknown/rogue device_id (e.g. credential_misuse's fake device)
    fp = dev["fingerprint"]
    return (fp["os"], fp["browser"], fp["screen"])
 
 
def compute_population_hour_priors_seed():
    """
    Fix for future-leakage bug: population priors are now computed
    CAUSALLY, incrementally, inside engineer_features() itself (see
    `dept_hour_hist`) - updated only from established (non cold-start)
    users' events as they are processed in chronological order. This
    function only supplies a generic, data-free starting default for the
    very first events of the simulation, before any department has
    accumulated real history yet (there is genuinely no leakage-free data
    to use at that point).
    """
    return {"mean_hour": 13.0, "std_hour": 4.0}
 
 
def build_access_graph(entities):
    """Rebuild the Day-1 access-topology graph from entities.json so we can
    compute graph-distance features here (used for intra-department lateral
    movement detection, not just cross-department detection)."""
    import networkx as nx
    G = nx.Graph()
    for u, v, attrs in entities["graph_edges"]:
        G.add_edge(u, v, **attrs)
    return G
 
 
def engineer_features(df, entities):
    import networkx as nx
 
    host_dept = build_host_dept_map(entities)
    users_meta = entities["users"]
    cold_start_ids = {u for u, v in users_meta.items() if v["is_cold_start"]}
    G = build_access_graph(entities)
    # precompute shortest-path lengths lazily per host (cached) - full
    # all-pairs would be expensive and unnecessary
    _sp_cache = {}
 
    def graph_dist(host_a, host_b):
        if host_a == host_b:
            return 0
        key = tuple(sorted((host_a, host_b)))
        if key not in _sp_cache:
            try:
                _sp_cache[key] = nx.shortest_path_length(G, host_a, host_b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                _sp_cache[key] = 99  # effectively "unreachable" via normal topology
        return _sp_cache[key]
 
    # per-user rolling state (all causal: updated AFTER processing each event)
    user_hour_hist = defaultdict(list)
    user_devices_seen = defaultdict(set)
    user_hosts_seen = defaultdict(set)
    user_depts_seen = defaultdict(set)
    user_last_event = {}  # user_id -> (timestamp, lat, lon)
    user_first_seen = {}  # user_id -> timestamp
    user_event_count = defaultdict(int)
    device_fp_hist = defaultdict(Counter)  # device_id -> Counter of fingerprint tuples actually observed
 
    # FIX (future-leakage bug): population hour priors are now built up
    # causally, department by department, ONLY from established
    # (non-cold-start) users' events, as they are processed in chronological
    # order - exactly like the personal baselines are. No event ever
    # contributes to a prior used to score an earlier event.
    dept_hour_hist = defaultdict(list)
 
    # rolling windows (deques of timestamps) for burst detection
    user_foreign_dept_events = defaultdict(deque)   # (timestamp, dept) touched, for lateral movement burst
    user_new_host_events = defaultdict(deque)       # (timestamp,) new-host touches, for lateral movement burst
    user_failure_events = defaultdict(deque)        # (timestamp,) failed logins, for brute-force burst
 
    rows_out = []
 
    for row in df.itertuples(index=False):
        uid = row.user_id
        ts = row.timestamp
        dept = row.dept
        host = row.host
        device_id = row.device_id
        status = row.status
 
        meta = users_meta.get(uid, {})
        is_declared_cold_start = meta.get("is_cold_start", False)
        n_hist = user_event_count[uid]
        entity_history_days = (ts - user_first_seen[uid]).total_seconds() / 86400 if uid in user_first_seen else 0.0
        # an entity is "cold" if it lacks enough OBSERVED history so far, regardless
        # of whether it was a "new hire" in the generator - this makes the feature
        # general enough to also apply mid-simulation to anyone with thin history
        is_cold_now = n_hist < MIN_HISTORY_EVENTS
        cold_start_blend_weight = min(1.0, n_hist / MIN_HISTORY_EVENTS)  # 0=fully population prior, 1=fully personal
 
        # ---------- 1. Time-of-day deviation (credential_misuse) ----------
        personal_mean_h, personal_std_h = circular_hour_stats(user_hour_hist[uid])
        home_dept_for_prior = meta.get("dept", dept)
        prior_mean_h, prior_std_h = circular_hour_stats(dept_hour_hist[home_dept_for_prior])
        if prior_mean_h is None:
            prior_mean_h, prior_std_h = 13.0, 4.0  # generic default: genuinely no leakage-free data exists yet
        if personal_mean_h is None:
            mean_h, std_h = prior_mean_h, prior_std_h
        else:
            # blend personal and population estimates by history depth
            mean_h = cold_start_blend_weight * personal_mean_h + (1 - cold_start_blend_weight) * prior_mean_h
            std_h = cold_start_blend_weight * personal_std_h + (1 - cold_start_blend_weight) * prior_std_h
        hour = ts.hour + ts.minute / 60
        hour_zscore = circular_hour_distance(hour, mean_h) / max(std_h, 0.5)
 
        # ---------- 2. Geo-velocity (impossible_travel) ----------
        if uid in user_last_event:
            last_ts, last_lat, last_lon = user_last_event[uid]
            dist_km = haversine_km(last_lat, last_lon, row.geo_lat, row.geo_lon)
            dt_hr = max((ts - last_ts).total_seconds() / 3600, 1 / 60)  # floor at 1 min
            geo_velocity_kmh = dist_km / dt_hr
            time_since_last_event_min = (ts - last_ts).total_seconds() / 60
        else:
            dist_km, geo_velocity_kmh, time_since_last_event_min = 0.0, 0.0, np.nan
 
        # ---------- 3. Device novelty (credential_misuse, brute_force) ----------
        is_new_device = int(device_id not in user_devices_seen[uid])
 
        # ---------- 4. Fingerprint mismatch (device_spoofing) ----------
        claimed_fp = None
        if isinstance(getattr(row, "spoofed_fingerprint", None), str):
            try:
                fp = json.loads(row.spoofed_fingerprint)
                claimed_fp = (fp["os"], fp["browser"], fp["screen"])
            except (json.JSONDecodeError, KeyError, TypeError):
                claimed_fp = None
        if claimed_fp is None:
            claimed_fp = get_device_true_fingerprint(entities, device_id)
 
        hist_fps = device_fp_hist[device_id]
        if len(hist_fps) == 0 or claimed_fp is None:
            fingerprint_mismatch = 0  # nothing to compare against yet
        else:
            most_common_fp, _ = hist_fps.most_common(1)[0]
            fingerprint_mismatch = int(claimed_fp != most_common_fp)
 
        # ---------- 5. Host / graph-based lateral movement ----------
        # IMPORTANT: use the HOST's actual owning department (from the access
        # graph), not the log's `dept` column - that column records the USER's
        # home department and is therefore constant per-user, so it can never
        # reveal a foreign-department access on its own.
        is_new_host = int(host not in user_hosts_seen[uid])
        home_dept = meta.get("dept", dept)
        host_owner_dept = host_dept.get(host, "shared")
        is_foreign_dept = int(host_owner_dept != home_dept and host_owner_dept != "shared")
        is_first_time_in_this_dept = int(host_owner_dept not in user_depts_seen[uid] and is_foreign_dept)
 
        # purge rolling windows to LATERAL_WINDOW_MIN, then measure burst BEFORE adding this event
        dq_fd = user_foreign_dept_events[uid]
        while dq_fd and (ts - dq_fd[0][0]).total_seconds() > LATERAL_WINDOW_MIN * 60:
            dq_fd.popleft()
        foreign_dept_burst_distinct = len({d for _, d in dq_fd})
        # (dq_fd stores (timestamp, host_owner_dept) tuples - appended below after scoring)
 
        dq_nh = user_new_host_events[uid]
        while dq_nh and (ts - dq_nh[0]).total_seconds() > LATERAL_WINDOW_MIN * 60:
            dq_nh.popleft()
        new_host_burst_count = len(dq_nh)
 
        # ---------- 5b. Graph-distance novelty (catches INTRA-department
        # lateral movement too, not just cross-department jumps) ----------
        # For each event, measure the shortest-path graph distance from the
        # accessed host to the NEAREST host this entity has ever visited
        # before. A compromised account moving between unfamiliar hosts
        # inside its own department (e.g. an engineer's account suddenly
        # reaching deep into engineering hosts it has never touched) still
        # shows up here even though is_foreign_dept would stay 0.
        visited = user_hosts_seen[uid]
        if not visited:
            graph_dist_from_history = 0  # first-ever event: no history to compare against
        elif host in visited:
            graph_dist_from_history = 0
        else:
            graph_dist_from_history = min(graph_dist(host, h) for h in list(visited)[-20:])  # cap for speed
 
        # ---------- 6. Brute-force burst ----------
        dq_fail = user_failure_events[uid]
        while dq_fail and (ts - dq_fail[0]).total_seconds() > BRUTE_FORCE_WINDOW_MIN * 60:
            dq_fail.popleft()
        failure_burst_count = len(dq_fail)
 
        rows_out.append({
            "event_id": row.event_id,
            "timestamp": ts,
            "user_id": uid,
            "dept": dept,
            "host": host,
            "device_id": device_id,
            "status": status,
            "label": row.label,
            "attack_type": row.attack_type,
            "attack_id": row.attack_id,
            # --- engineered features ---
            "hour_zscore": round(hour_zscore, 3),
            "geo_distance_km": round(dist_km, 2),
            "geo_velocity_kmh": round(geo_velocity_kmh, 2),
            "time_since_last_event_min": round(time_since_last_event_min, 2) if not np.isnan(time_since_last_event_min) else np.nan,
            "is_new_device": is_new_device,
            "fingerprint_mismatch": fingerprint_mismatch,
            "is_new_host": is_new_host,
            "is_foreign_dept": is_foreign_dept,
            "is_first_time_in_this_dept": is_first_time_in_this_dept,
            "foreign_dept_burst_distinct": foreign_dept_burst_distinct,
            "new_host_burst_count": new_host_burst_count,
            "graph_dist_from_history": graph_dist_from_history,
            "failure_burst_count": failure_burst_count,
            "entity_history_days": round(entity_history_days, 2),
            "entity_history_event_count": n_hist,
            "is_cold_start_entity": int(is_cold_now),
            "cold_start_blend_weight": round(cold_start_blend_weight, 2),
        })
 
        # ---------- update rolling state AFTER scoring (causality preserved) ----------
        user_hour_hist[uid].append(hour)
        if len(user_hour_hist[uid]) > 500:
            user_hour_hist[uid] = user_hour_hist[uid][-500:]
        # population prior update - ONLY from established users, and only
        # AFTER this event has been scored, so it can never leak into its
        # own or any earlier event's score
        if uid not in cold_start_ids:
            dept_hour_hist[home_dept_for_prior].append(hour)
            if len(dept_hour_hist[home_dept_for_prior]) > 2000:
                dept_hour_hist[home_dept_for_prior] = dept_hour_hist[home_dept_for_prior][-2000:]
        user_devices_seen[uid].add(device_id)
        user_hosts_seen[uid].add(host)
        user_depts_seen[uid].add(host_owner_dept)
        user_last_event[uid] = (ts, row.geo_lat, row.geo_lon)
        if uid not in user_first_seen:
            user_first_seen[uid] = ts
        user_event_count[uid] += 1
        if claimed_fp is not None:
            device_fp_hist[device_id][claimed_fp] += 1
        if is_foreign_dept:
            dq_fd.append((ts, host_owner_dept))
        if is_new_host:
            dq_nh.append(ts)
        if status == "failure":
            dq_fail.append(ts)
 
    return pd.DataFrame(rows_out)
 
 
def main():
    print("Loading data...")
    df, entities = load_data()
    print(f"  {len(df)} raw events")
 
    print("Engineering features (strictly causal, per-entity)...")
    feat_df = engineer_features(df, entities)
 
    out_path = f"{DATA_DIR}/features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"Saved {len(feat_df)} rows x {len(feat_df.columns)} cols -> {out_path}")
 
    print("\nFeature summary by label:")
    numeric_cols = ["hour_zscore", "geo_velocity_kmh", "is_new_device", "fingerprint_mismatch",
                     "is_new_host", "foreign_dept_burst_distinct", "new_host_burst_count",
                     "graph_dist_from_history", "failure_burst_count"]
    print(feat_df.groupby("label")[numeric_cols].mean().round(3))
 
 
if __name__ == "__main__":
    main()
 
