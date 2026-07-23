"""generate_logs.py
================
Synthetic access-log generator for the behavioral anomaly detection system.
 
Produces:
  - data/synthetic_logs.csv        : the full event log (normal + attack events)
  - data/entities.json              : ground-truth user/device/graph metadata
                                       (used by features.py for baselining)
  - data/attack_sessions.json       : ground truth of injected attacks (for eval)
 
Design notes
------------
- ~50 users across 5 departments, each with a stable "normal" behavioral
  profile (typical hours, typical geo, typical devices, typical hosts).
- ~90 hosts/resources organized into an access-topology GRAPH (networkx),
  so lateral movement can later be detected as "traversal outside the
  entity's normal reachable subgraph" rather than a naive host-count rule.
- A concept-drift event is baked in around day 20/30: a subset of users
  permanently shifts to a new work-from-home pattern (new geo + new hours),
  simulating a real policy change. This lets us DEMO drift detection later,
  since we control exactly when and for whom the shift happens.
- 5 attack types are injected at low frequency (~3-4% of total events) to
  produce realistic class imbalance:
    1. brute_force        - rapid repeated failed logins from one source
    2. impossible_travel   - two logins geographically infeasible in the gap
    3. lateral_movement    - fast traversal across hosts outside the
                              entity's normal reachable subgraph
    4. credential_misuse    - odd-hour + new-device + unusual-resource combo
    5. device_spoofing      - claimed device_id doesn't match its known
                              fingerprint history
- Cold-start is simulated by having several users NOT appear at all until
  after day 20 (brand-new hires with zero history for the model to learn from).
 
INCIDENT-COUNT NOTE (revised after Day 4 review): the 4 rarer attack types
were bumped from 20 -> 38 incidents each (brute_force stays at 25 since it
already yields 200+ events). With a 60/20/20 session-aware split, 20
incidents put only ~4 in the test fold - a single flipped example swings
recall by 25 points, too noisy to report honestly. 38 incidents -> ~7-8 in
test, roughly halving that swing. This nudges overall attack traffic from
~3% to ~4% of events, and N_USERS was raised 50 -> 70 (non-cold-start pool
45 -> 65) alongside it so attacks don't concentrate on nearly the whole
eligible population - a deliberate, disclosed trade-off of a bit of
realism for statistically defensible per-class recall numbers.
"""
 
import json
import os
import random
import uuid
from datetime import datetime, timedelta
 
import networkx as nx
import numpy as np
import pandas as pd
from faker import Faker
 
# Output goes to a "data" folder next to this script (created automatically
# if it doesn't exist) — so this works no matter where you put the script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
 
fake = Faker()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)
 
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_USERS = 70
HOSTS_PER_DEPT = 16
SHARED_HOSTS = 10           # hosts everyone can legitimately touch (email, VPN, wiki)
SIM_DAYS = 30
EVENTS_PER_USER_PER_DAY = (4, 14)
DRIFT_DAY = 20                        # day the WFH policy shift kicks in
DRIFT_USER_FRACTION = 0.35
COLD_START_NEW_HIRE_DAY = 22          # brand-new users appear only from this day on
N_COLD_START_USERS = 5
START_DATE = datetime(2026, 6, 15, 0, 0, 0)
 
DEPARTMENTS = ["engineering", "sales", "finance", "hr", "it_admin"]
 
CITY_POOL = [
    ("Bengaluru", 12.9716, 77.5946, "IN"),
    ("Mumbai", 19.0760, 72.8777, "IN"),
    ("Delhi", 28.7041, 77.1025, "IN"),
    ("Pune", 18.5204, 73.8567, "IN"),
    ("Hyderabad", 17.3850, 78.4867, "IN"),
    ("Singapore", 1.3521, 103.8198, "SG"),
    ("London", 51.5072, -0.1276, "GB"),
    ("New York", 40.7128, -74.0060, "US"),
    ("San Francisco", 37.7749, -122.4194, "US"),
    ("Frankfurt", 50.1109, 8.6821, "DE"),
    ("Tokyo", 35.6762, 139.6503, "JP"),
    ("Sydney", -33.8688, 151.2093, "AU"),
    ("Dubai", 25.2048, 55.2708, "AE"),
    ("Moscow", 55.7558, 37.6173, "RU"),   # rare/hostile geo -> for impossible travel / misuse
    ("Lagos", 6.5244, 3.3792, "NG"),      # rare/hostile geo
]
 
OS_LIST = ["Windows 11", "macOS 15", "Ubuntu 24.04", "iOS 18", "Android 15"]
BROWSER_LIST = ["Chrome 126", "Edge 126", "Firefox 128", "Safari 18"]
 
 
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))
 
 
# ---------------------------------------------------------------------------
# 1. Access-topology graph
# ---------------------------------------------------------------------------
def build_access_graph():
    G = nx.Graph()
    dept_hosts = {}
 
    for dept in DEPARTMENTS:
        hosts = [f"{dept}-host-{i:02d}" for i in range(HOSTS_PER_DEPT)]
        dept_hosts[dept] = hosts
        G.add_nodes_from(hosts, dept=dept)
        for h in hosts:
            peers = random.sample(hosts, k=min(4, len(hosts) - 1))
            for p in peers:
                if p != h:
                    G.add_edge(h, p, kind="intra_dept")
 
    shared_hosts = [f"shared-host-{i:02d}" for i in range(SHARED_HOSTS)]
    G.add_nodes_from(shared_hosts, dept="shared")
    for dept, hosts in dept_hosts.items():
        for h in random.sample(hosts, k=min(5, len(hosts))):
            for s in random.sample(shared_hosts, k=3):
                G.add_edge(h, s, kind="shared_link")
 
    return G, dept_hosts, shared_hosts
 
 
# ---------------------------------------------------------------------------
# 2. User & device profiles
# ---------------------------------------------------------------------------
def build_entities(dept_hosts, shared_hosts):
    users = {}
    devices = {}
    user_ids = [f"user_{i:03d}" for i in range(N_USERS)]
    cold_start_ids = set(user_ids[-N_COLD_START_USERS:])  # last N users = new hires
 
    for i, user_id in enumerate(user_ids):
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        home_city = random.choice(CITY_POOL[:5])
        start_hr = random.choice([7, 8, 9, 10])
        end_hr = start_hr + random.choice([8, 9, 10])
        n_devices = random.choice([1, 1, 2])
        user_devices = []
        for d in range(n_devices):
            device_id = f"{user_id}-dev{d}"
            fingerprint = {
                "os": random.choice(OS_LIST),
                "browser": random.choice(BROWSER_LIST),
                "screen": random.choice(["1920x1080", "2560x1440", "1366x768", "2880x1800"]),
            }
            devices[device_id] = {"owner": user_id, "fingerprint": fingerprint}
            user_devices.append(device_id)
 
        normal_hosts = random.sample(dept_hosts[dept], k=min(6, len(dept_hosts[dept])))
        normal_hosts += random.sample(shared_hosts, k=3)
 
        # --- Realism fix: some roles genuinely have legitimate broad access ---
        # (e.g. IT admins / some managers routinely touch other departments'
        # hosts as part of their job). Without this, "foreign department
        # access" would be a perfect attack tell, which is unrealistic and
        # would make the lateral-movement signal look artificially clean.
        broad_access = (dept == "it_admin") or (random.random() < 0.30)
        broad_access_hosts = []
        if broad_access:
            foreign_pool = [h for d, hs in dept_hosts.items() if d != dept for h in hs]
            broad_access_hosts = random.sample(foreign_pool, k=min(4, len(foreign_pool)))
 
        # --- Realism fix: ~20% of users legitimately get a NEW device partway
        # through the simulation (a real device upgrade), so `is_new_device`
        # is not a perfect attack tell either - some normal events genuinely
        # trigger it too.
        upgrade_device = None
        upgrade_day = None
        if random.random() < 0.20:
            upgrade_day = random.randint(10, 25)
            upgrade_device_id = f"{user_id}-dev-upgrade"
            fingerprint = {
                "os": random.choice(OS_LIST),
                "browser": random.choice(BROWSER_LIST),
                "screen": random.choice(["1920x1080", "2560x1440", "1366x768", "2880x1800"]),
            }
            devices[upgrade_device_id] = {"owner": user_id, "fingerprint": fingerprint}
            upgrade_device = upgrade_device_id
 
        users[user_id] = {
            "dept": dept,
            "role": random.choice(["ic", "ic", "manager", "senior_ic"]),
            "home_city": home_city,
            "work_start_hr": start_hr,
            "work_end_hr": end_hr,
            "devices": user_devices,
            "normal_hosts": normal_hosts,
            "drift_group": False,
            "is_cold_start": user_id in cold_start_ids,
            "broad_access": broad_access,
            "broad_access_hosts": broad_access_hosts,
            "upgrade_device": upgrade_device,
            "upgrade_day": upgrade_day,
        }
 
    non_cold_start = [u for u in users if not users[u]["is_cold_start"]]
    drift_users = random.sample(non_cold_start, k=int(len(non_cold_start) * DRIFT_USER_FRACTION))
    for u in drift_users:
        users[u]["drift_group"] = True
        new_city = random.choice(CITY_POOL[:6])
        new_start = random.choice([11, 12, 13, 22, 23])
        users[u]["post_drift_city"] = new_city
        users[u]["post_drift_start_hr"] = new_start
        users[u]["post_drift_end_hr"] = new_start + random.choice([8, 9])
 
    return users, devices
 
 
# ---------------------------------------------------------------------------
# 3. Normal event generation
# ---------------------------------------------------------------------------
def random_timestamp_on_day(day_offset, work_start, work_end):
    day = START_DATE + timedelta(days=day_offset)
    if random.random() < 0.92:
        hr = random.randint(work_start, max(work_start, work_end - 1))
    else:
        hr = random.randint(0, 23)
    minute = random.randint(0, 59)
    sec = random.randint(0, 59)
    return day.replace(hour=hr % 24, minute=minute, second=sec)
 
 
def jitter_geo(lat, lon, km=15):
    dlat = (km / 111.0) * (random.random() - 0.5) * 2
    dlon = (km / (111.0 * np.cos(np.radians(lat)))) * (random.random() - 0.5) * 2
    return lat + dlat, lon + dlon
 
 
def make_event(user_id, users, timestamp, host, device_id, status="success",
               label="normal", attack_type=None, attack_id=None, source_ip=None,
               override_city=None):
    profile = users[user_id]
    city = override_city or profile["home_city"]
    city_name, lat, lon, country = city
    jlat, jlon = jitter_geo(lat, lon)
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": timestamp.isoformat(),
        "user_id": user_id,
        "dept": profile["dept"],
        "device_id": device_id,
        "source_ip": source_ip or fake.ipv4_public(),
        "geo_city": city_name,
        "geo_country": country,
        "geo_lat": round(jlat, 4),
        "geo_lon": round(jlon, 4),
        "host": host,
        "status": status,
        "label": label,
        "attack_type": attack_type,
        "attack_id": attack_id,
    }
 
 
def generate_normal_events(users):
    events = []
    for user_id, profile in users.items():
        start_day = COLD_START_NEW_HIRE_DAY if profile["is_cold_start"] else 0
        for day in range(start_day, SIM_DAYS):
            drifted = profile["drift_group"] and day >= DRIFT_DAY
            work_start = profile["post_drift_start_hr"] if drifted else profile["work_start_hr"]
            work_end = profile["post_drift_end_hr"] if drifted else profile["work_end_hr"]
            city = profile["post_drift_city"] if drifted else profile["home_city"]
 
            # devices actually available to this user today (includes the
            # upgrade device only from its introduction day onward - this is
            # what creates genuine, legitimate "new device" events in normal
            # traffic, not just in attacks)
            available_devices = list(profile["devices"])
            if profile["upgrade_device"] and day >= profile["upgrade_day"]:
                available_devices.append(profile["upgrade_device"])
 
            n_events = random.randint(*EVENTS_PER_USER_PER_DAY)
            for _ in range(n_events):
                ts = random_timestamp_on_day(day, work_start, work_end)
                device_id = random.choice(available_devices)
 
                # Realism fix: broad-access roles (IT admins, some managers)
                # legitimately touch other departments' hosts sometimes -
                # this is what keeps "foreign department access" from being
                # a perfect, unrealistic attack tell.
                if profile["broad_access"] and profile["broad_access_hosts"] and random.random() < 0.06:
                    host = random.choice(profile["broad_access_hosts"])
                else:
                    host = random.choice(profile["normal_hosts"])
 
                # Realism fix: occasionally a normal login has one preceding
                # failed attempt (mistyped password) before succeeding - real
                # users do this constantly, and without it, ANY failed login
                # would trivially mean "attack," which is unrealistic.
                if random.random() < 0.03:
                    typo_ts = ts - timedelta(seconds=random.randint(3, 15))
                    events.append(make_event(user_id, users, typo_ts, host, device_id,
                                              status="failure", label="normal", override_city=city))
 
                ev = make_event(user_id, users, ts, host, device_id,
                                 status="success", label="normal", override_city=city)
                events.append(ev)
    return events
 
 
# ---------------------------------------------------------------------------
# 4. Attack injection (only among non-cold-start users with real history,
#    so attacks are genuinely "abnormal relative to an established baseline")
# ---------------------------------------------------------------------------
def eligible_victims(users, k):
    pool = [u for u in users if not users[u]["is_cold_start"]]
    return random.sample(pool, k=min(k, len(pool)))
 
 
def inject_brute_force(users, n_attacks=25):
    events, meta = [], []
    for user_id in eligible_victims(users, n_attacks):
        profile = users[user_id]
        day = random.randint(0, SIM_DAYS - 1)
        base_ts = random_timestamp_on_day(day, 0, 23)
        attack_id = f"bf_{user_id}_{day}"
        n_attempts = random.randint(4, 20)
        src_ip = fake.ipv4_public()
        device_id = random.choice(profile["devices"])
        for i in range(n_attempts):
            ts = base_ts + timedelta(seconds=i * random.randint(2, 8))
            success = (i == n_attempts - 1) and random.random() < 0.4
            ev = make_event(user_id, users, ts, random.choice(profile["normal_hosts"]),
                             device_id, status="success" if success else "failure",
                             label="attack", attack_type="brute_force",
                             attack_id=attack_id, source_ip=src_ip)
            events.append(ev)
        meta.append({"attack_id": attack_id, "type": "brute_force",
                      "user_id": user_id, "n_events": n_attempts})
    return events, meta
 
 
def inject_impossible_travel(users, n_attacks=38):
    events, meta = [], []
    for user_id in eligible_victims(users, n_attacks):
        profile = users[user_id]
        day = random.randint(0, SIM_DAYS - 1)
        ts1 = random_timestamp_on_day(day, profile["work_start_hr"], profile["work_end_hr"])
        far_city = random.choice(CITY_POOL[6:])
        gap_minutes = random.randint(5, 150)
        ts2 = ts1 + timedelta(minutes=gap_minutes)
        attack_id = f"it_{user_id}_{day}"
        device_id = random.choice(profile["devices"])
 
        ev1 = make_event(user_id, users, ts1, random.choice(profile["normal_hosts"]),
                          device_id, status="success", label="normal")
        ev2 = make_event(user_id, users, ts2, random.choice(profile["normal_hosts"]),
                          device_id, status="success", label="attack",
                          attack_type="impossible_travel", attack_id=attack_id,
                          override_city=far_city)
        events.extend([ev1, ev2])
        meta.append({"attack_id": attack_id, "type": "impossible_travel",
                      "user_id": user_id, "n_events": 2})
    return events, meta
 
 
def inject_lateral_movement(users, dept_hosts, n_attacks=38):
    events, meta = [], []
    for user_id in eligible_victims(users, n_attacks):
        profile = users[user_id]
        day = random.randint(0, SIM_DAYS - 1)
        base_ts = random_timestamp_on_day(day, 0, 23)
        attack_id = f"lm_{user_id}_{day}"
        device_id = random.choice(profile["devices"])
        foreign_depts = [d for d in DEPARTMENTS if d != profile["dept"]]
        n_depts = random.randint(1, 3)
        chosen_depts = random.sample(foreign_depts, k=min(n_depts, len(foreign_depts)))
        foreign_hosts = []
        for d in chosen_depts:
            foreign_hosts.extend(random.sample(dept_hosts[d], k=random.randint(1, 2)))
        # Realism fix: sometimes mix in one host from the user's OWN
        # department mid-session (a compromised account often still touches
        # some legitimate resources too) - dilutes what would otherwise be
        # a uniformly 100% foreign-host session.
        session_hosts = foreign_hosts
        if random.random() < 0.25 and profile["normal_hosts"]:
            insert_at = random.randint(0, len(session_hosts))
            session_hosts = session_hosts[:insert_at] + [random.choice(profile["normal_hosts"])] + session_hosts[insert_at:]
 
        for i, host in enumerate(session_hosts):
            ts = base_ts + timedelta(minutes=i * random.randint(1, 4))
            ev = make_event(user_id, users, ts, host, device_id, status="success",
                             label="attack", attack_type="lateral_movement",
                             attack_id=attack_id)
            events.append(ev)
        meta.append({"attack_id": attack_id, "type": "lateral_movement",
                      "user_id": user_id, "n_events": len(session_hosts)})
    return events, meta
 
 
def inject_credential_misuse(users, n_attacks=38):
    events, meta = [], []
    for user_id in eligible_victims(users, n_attacks):
        profile = users[user_id]
        day = random.randint(0, SIM_DAYS - 1)
        odd_hr = (profile["work_end_hr"] + random.randint(2, 6)) % 24
        ts = (START_DATE + timedelta(days=day)).replace(hour=odd_hr, minute=random.randint(0, 59))
        attack_id = f"cm_{user_id}_{day}"
        # Realism fix: ~35% of the time the attacker uses a device the user
        # HAS used before (e.g. a stolen/compromised laptop already trusted
        # on the network) - so is_new_device isn't a perfect tell either.
        # The odd-hour timing remains the actual anomaly signal in that case.
        if random.random() < 0.35 and profile["devices"]:
            rogue_device = random.choice(profile["devices"])
        else:
            rogue_device = f"unknown-dev-{uuid.uuid4().hex[:6]}"
        rare_host = random.choice(profile["normal_hosts"])
        ev = make_event(user_id, users, ts, rare_host, rogue_device, status="success",
                         label="attack", attack_type="credential_misuse", attack_id=attack_id)
        events.append(ev)
        meta.append({"attack_id": attack_id, "type": "credential_misuse",
                      "user_id": user_id, "n_events": 1})
    return events, meta
 
 
def inject_device_spoofing(users, n_attacks=38):
    events, meta = [], []
    for user_id in eligible_victims(users, n_attacks):
        profile = users[user_id]
        day = random.randint(0, SIM_DAYS - 1)
        ts = random_timestamp_on_day(day, profile["work_start_hr"], profile["work_end_hr"])
        attack_id = f"ds_{user_id}_{day}"
        real_device_id = random.choice(profile["devices"])
        ev = make_event(user_id, users, ts, random.choice(profile["normal_hosts"]),
                         real_device_id, status="success", label="attack",
                         attack_type="device_spoofing", attack_id=attack_id)
        ev["spoofed_fingerprint"] = json.dumps({
            "os": random.choice(OS_LIST),
            "browser": random.choice(BROWSER_LIST),
            "screen": random.choice(["1920x1080", "1366x768"]),
        })
        events.append(ev)
        meta.append({"attack_id": attack_id, "type": "device_spoofing",
                      "user_id": user_id, "n_events": 1})
    return events, meta
 
 
# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    print("Building access-topology graph...")
    G, dept_hosts, shared_hosts = build_access_graph()
 
    print("Building user/device profiles...")
    users, devices = build_entities(dept_hosts, shared_hosts)
 
    print("Generating normal events...")
    normal_events = generate_normal_events(users)
    print(f"  {len(normal_events)} normal events")
 
    print("Injecting attacks...")
    bf_events, bf_meta = inject_brute_force(users)
    it_events, it_meta = inject_impossible_travel(users)
    lm_events, lm_meta = inject_lateral_movement(users, dept_hosts)
    cm_events, cm_meta = inject_credential_misuse(users)
    ds_events, ds_meta = inject_device_spoofing(users)
 
    attack_events = bf_events + it_events + lm_events + cm_events + ds_events
    attack_meta = bf_meta + it_meta + lm_meta + cm_meta + ds_meta
    print(f"  {len(attack_events)} attack events across {len(attack_meta)} attack sessions")
 
    all_events = normal_events + attack_events
    df = pd.DataFrame(all_events)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
 
    n_total = len(df)
    n_attack = (df["label"] == "attack").sum()
    print(f"\nTotal events: {n_total} | Attack events: {n_attack} ({100*n_attack/n_total:.2f}%)")
    print(df["attack_type"].value_counts(dropna=False))
 
    df.to_csv("/Users/anj/Projects/anomaly_det/data/synthetic_logs.csv", index=False)
 
    entities_out = {
        "users": users,
        "devices": devices,
        "dept_hosts": dept_hosts,
        "shared_hosts": shared_hosts,
        "graph_edges": list(G.edges(data=True)),
        "config": {
            "drift_day": DRIFT_DAY,
            "cold_start_day": COLD_START_NEW_HIRE_DAY,
            "start_date": START_DATE.isoformat(),
            "sim_days": SIM_DAYS,
        },
    }
    with open("/Users/anj/Projects/anomaly_det/data/entities.json", "w") as f:
        json.dump(entities_out, f, indent=2, default=str)
 
    with open("/Users/anj/Projects/anomaly_det/data/attack_sessions.json", "w") as f:
        json.dump(attack_meta, f, indent=2)
 
    print("\nSaved: data/synthetic_logs.csv, data/entities.json, data/attack_sessions.json")
 
 
if __name__ == "__main__":
    main()
 
